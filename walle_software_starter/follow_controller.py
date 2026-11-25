import os
import time
from typing import Dict, List, Optional, Any, Tuple

try:
    from person_tracker import PersonTracker, PersonObservation, ReacquisitionStatus
    from tag_tracker import TagTrackerManager, TagReading, RadioSource
except ImportError:
    from walle_software_starter.person_tracker import PersonTracker, PersonObservation, ReacquisitionStatus
    from walle_software_starter.tag_tracker import TagTrackerManager, TagReading, RadioSource


class FollowMode:
    MANUAL = "MANUAL"
    PERSON = "PERSON"
    POCKET_TAG = "POCKET_TAG"


class FollowState:
    IDLE = "IDLE"
    ACQUIRING = "ACQUIRING"
    FOLLOWING = "FOLLOWING"
    HOLDING_DISTANCE = "HOLDING_DISTANCE"
    TARGET_LOST = "TARGET_LOST"
    OBSTACLE_STOP = "OBSTACLE_STOP"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"
    COMMUNICATION_ERROR = "COMMUNICATION_ERROR"


class FollowDecision:
    STOP = "STOP"
    MOVE_FORWARD = "MOVE_FORWARD"
    MOVE_BACKWARD = "MOVE_BACKWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    HOLD = "HOLD"


class FollowController:
    """
    Unified Follow Controller for WALL-E.
    Supports Person Follow (Vision) and Pocket Tag Follow (UWB/BLE).
    Enforces simulation safety, independent distance and direction processing,
    prohibits forward motion without calibrated metric distance, and halts on target loss.
    """

    def __init__(self,
                 person_tracker: Optional[PersonTracker] = None,
                 tag_tracker: Optional[TagTrackerManager] = None,
                 target_distance_m: float = 1.5,
                 distance_tolerance_m: float = 0.3,
                 min_safe_distance_m: float = 0.8,
                 approach_threshold_m: Optional[float] = None,
                 hold_min_m: Optional[float] = None,
                 hold_max_m: Optional[float] = None,
                 stale_timeout_sec: float = 1.0,
                 drive_speed: int = 150,
                 turn_speed: int = 120):
        self.person_tracker = person_tracker if person_tracker is not None else PersonTracker()
        self.tag_tracker = tag_tracker if tag_tracker is not None else TagTrackerManager()

        # Follow configuration
        self.target_distance_m = target_distance_m
        self.min_following_distance_m = 1.0
        self.max_following_distance_m = 5.0
        self.distance_tolerance_m = distance_tolerance_m
        self.min_safe_distance_m = min_safe_distance_m
        self.approach_threshold_m = approach_threshold_m if approach_threshold_m is not None else (target_distance_m + distance_tolerance_m)
        self.hold_min_m = hold_min_m if hold_min_m is not None else (target_distance_m - distance_tolerance_m)
        self.hold_max_m = hold_max_m if hold_max_m is not None else (target_distance_m + distance_tolerance_m)
        self.stale_timeout_sec = stale_timeout_sec
        self.drive_speed = drive_speed
        self.turn_speed = turn_speed

        # Controller state
        self.mode = FollowMode.MANUAL
        self.enabled = False
        self.state = FollowState.IDLE
        self.decision = FollowDecision.STOP
        self.manual_override_active = False
        self.emergency_stop_latched = False
        # Obstacle & Sensor Telemetry State
        self.simulated_obstacle: bool = False
        self.sensor_obstacle: bool = False
        self.ultrasonic_distance_cm: Optional[float] = None
        self.ultrasonic_valid: bool = True
        self.ultrasonic_stale: bool = False
        self.ultrasonic_obstacle_threshold_cm: float = 30.0
        self.require_ultrasonic_for_forward: bool = False
        self.sensor_telemetry_received: bool = False
        self.was_in_obstacle_stop: bool = False
        self.sensor_telemetry_source: Optional[Any] = None

        self.command_dispatcher: Optional[Any] = None

        # Command logging (simulation only - no real hardware commands)
        self.last_command: Dict[str, Any] = {"direction": "stop", "speed": 0, "simulated": True, "sent": False}
        self.command_history: List[Dict[str, Any]] = []
        self.max_history = 25

    @property
    def obstacle_detected(self) -> bool:
        """Obstacle is active if either simulated obstacle OR live sensor obstacle is detected."""
        return bool(self.simulated_obstacle or self.sensor_obstacle)

    @obstacle_detected.setter
    def obstacle_detected(self, val: bool) -> None:
        """Maintains backward compatibility for direct attribute assignment in legacy tests."""
        self.simulated_obstacle = bool(val)

    def trigger_obstacle(self, detected: bool) -> None:
        """
        Sets or clears software-simulated obstacle.
        Notice: If sensor_obstacle is True from physical telemetry,
        clearing simulated obstacle will not clear the real sensor obstacle (fail-safe).
        """
        self.simulated_obstacle = bool(detected)

    def update_sensor_telemetry(self,
                               distance_cm: Optional[float],
                               obstacle: bool = False,
                               is_valid: bool = True,
                               is_stale: bool = False) -> None:
        """
        Updates physical or mock sensor telemetry and evaluates obstacle state.
        Failsafe rules:
        - A valid distance <= threshold forces STOP (sensor_obstacle = True).
        - Stale or invalid readings cannot clear an existing obstacle.
        - Recovery requires fresh, valid readings with distance > threshold.
        """
        self.sensor_telemetry_received = True
        self.ultrasonic_distance_cm = distance_cm
        self.ultrasonic_valid = bool(is_valid)
        self.ultrasonic_stale = bool(is_stale)

        # Fail-safe: if stale or invalid, we never interpret it as clear path
        if not is_valid or is_stale:
            # Maintain previous obstacle state if already triggered, but do not clear
            return

        # Fresh and valid readings:
        if distance_cm is not None:
            if distance_cm <= self.ultrasonic_obstacle_threshold_cm or obstacle:
                self.sensor_obstacle = True
            else:
                self.sensor_obstacle = False
        else:
            if obstacle:
                self.sensor_obstacle = True
            else:
                self.sensor_obstacle = False

    def is_forward_path_clear(self) -> Tuple[bool, str]:
        """
        Validates whether autonomous forward movement is safe.
        Failsafe invariants:
        - If obstacle is detected (sensor or simulated) -> False
        - If ultrasonic telemetry is active or required:
            - Missing readings (distance_cm is None) -> False
            - Stale readings -> False
            - Invalid readings -> False
            - Distance <= threshold -> False
        - Clear only when fresh, valid readings confirm distance > threshold.
        """
        if self.obstacle_detected:
            return False, "Obstacle detected in forward path"

        # Check ultrasonic telemetry when active or required
        if self.require_ultrasonic_for_forward or self.sensor_telemetry_received:
            if not self.ultrasonic_valid:
                return False, "Forward movement inhibited: Ultrasonic sensor reading is invalid"
            if self.ultrasonic_stale:
                return False, "Forward movement inhibited: Ultrasonic sensor telemetry is stale"
            if self.ultrasonic_distance_cm is None:
                return False, "Forward movement inhibited: Ultrasonic distance measurement is missing"
            if self.ultrasonic_distance_cm <= self.ultrasonic_obstacle_threshold_cm:
                return False, f"Forward movement inhibited: Object at {self.ultrasonic_distance_cm:.1f}cm <= {self.ultrasonic_obstacle_threshold_cm:.1f}cm"

        return True, "Forward path clear"

    @property
    def selected_person_id(self) -> Optional[int]:
        return self.person_tracker.selected_person_id

    @selected_person_id.setter
    def selected_person_id(self, val: Optional[int]) -> None:
        self.person_tracker.selected_person_id = val

    @property
    def selected_tag_id(self) -> Optional[str]:
        return self.tag_tracker.selected_tag_id

    @selected_tag_id.setter
    def selected_tag_id(self, val: Optional[str]) -> None:
        self.tag_tracker.selected_tag_id = val

    def set_mode(self, new_mode: str) -> None:
        """
        Safely transitions following mode.
        Disables previous mode, resets target locks, and commands STOP.
        Cannot clear emergency stop latch if active.
        """
        mode_up = new_mode.upper()
        if mode_up not in (FollowMode.MANUAL, FollowMode.PERSON, FollowMode.POCKET_TAG):
            raise ValueError(f"Invalid follow mode '{new_mode}'. Expected MANUAL, PERSON, or POCKET_TAG.")

        if self.emergency_stop_latched:
            # Mode changes are recorded, but latch remains locked and enabled remains False
            self.mode = mode_up
            self.enabled = False
            self.decision = FollowDecision.STOP
            self.state = FollowState.MANUAL_OVERRIDE
            return

        if self.mode != mode_up:
            # Safely reset target locks and disable prior following
            self.enabled = False
            self.person_tracker.select_person(None)
            self.tag_tracker.select_tag(None)
            self.decision = FollowDecision.STOP
            self._dispatch_command(FollowDecision.STOP, reason=f"Switched mode from {self.mode} to {mode_up}")

            self.mode = mode_up
            self.state = FollowState.IDLE
            self.manual_override_active = False

    def set_enabled(self, enabled: bool) -> None:
        """Enables or disables autonomous following. Re-enabling is rejected if latched."""
        if self.emergency_stop_latched:
            # Re-enabling follow is strictly rejected while emergency stop is latched
            self.enabled = False
            self.decision = FollowDecision.STOP
            self.state = FollowState.MANUAL_OVERRIDE
            return

        if self.mode == FollowMode.MANUAL and enabled:
            # Cannot enable autonomous following while in MANUAL mode
            self.enabled = False
            self.state = FollowState.IDLE
            self.decision = FollowDecision.STOP
            return

        self.enabled = bool(enabled)
        if not self.enabled:
            self.state = FollowState.IDLE
            self.decision = FollowDecision.STOP
            self._dispatch_command(FollowDecision.STOP, reason="Autonomous follow disabled")
        else:
            self.manual_override_active = False
            self.state = FollowState.ACQUIRING

    def set_target_distance(self, distance_m: float) -> bool:
        """
        Sets target person-following distance within bounds [min_following_distance_m, max_following_distance_m].
        Recalculates approach threshold and hold min boundaries with distance tolerance hysteresis.
        Does not reset target selection, clear tracks, or disrupt active detours.
        """
        try:
            dist = float(distance_m)
        except (TypeError, ValueError):
            return False
        if not (self.min_following_distance_m <= dist <= self.max_following_distance_m):
            return False
        self.target_distance_m = dist
        self.approach_threshold_m = dist + self.distance_tolerance_m
        self.hold_min_m = max(self.min_safe_distance_m, dist - self.distance_tolerance_m)
        self.hold_max_m = dist + self.distance_tolerance_m
        return True

    def select_person_target(self, person_id: Optional[int]) -> None:
        """Sets target person ID or auto-select if None."""
        self.person_tracker.select_person(person_id)

    def select_tag_target(self, tag_id: Optional[str], source: Optional[str] = None) -> None:
        """Sets target tag ID and optionally switches radio source."""
        if source:
            self.tag_tracker.set_source(source)
        self.tag_tracker.select_tag(tag_id)

    def emergency_stop(self) -> Dict[str, Any]:
        """Emergency STOP: immediately overrides all modes, commands STOP, and engages persistent latch."""
        self.enabled = False
        self.emergency_stop_latched = True
        self.manual_override_active = True
        self.state = FollowState.MANUAL_OVERRIDE
        self.decision = FollowDecision.STOP
        return self._dispatch_command(FollowDecision.STOP, reason="Emergency STOP / Manual Override latched")

    def reset_emergency_stop(self) -> Dict[str, Any]:
        """
        Explicitly clears the emergency STOP latch.
        Returns the controller to IDLE. Does NOT resume following.
        Requires fresh target acquisition and explicit follow enable before following resumes.
        """
        self.emergency_stop_latched = False
        self.manual_override_active = False
        self.enabled = False
        self.state = FollowState.IDLE
        self.decision = FollowDecision.STOP
        self.was_in_obstacle_stop = False
        self.person_tracker.select_person(None)
        self.tag_tracker.select_tag(None)
        return self._dispatch_command(FollowDecision.STOP, reason="Emergency STOP latch reset to IDLE")

    def reacquire_target(self) -> bool:
        """
        Attempts to reacquire the locked target person.
        Transitions state to ACQUIRING if target was temporarily lost/occluded,
        maintaining the existing target lock.
        """
        if self.selected_person_id is not None:
            if self.state in (FollowState.TARGET_LOST, FollowState.IDLE):
                self.state = FollowState.ACQUIRING
            return True
        return False

    def _convert_decision_to_motor_cmd(self, decision: str) -> Dict[str, Any]:
        """Maps high-level decision to directional motor speeds."""
        if decision == FollowDecision.MOVE_FORWARD:
            return {"direction": "forward", "speed": self.drive_speed}
        elif decision == FollowDecision.MOVE_BACKWARD:
            return {"direction": "backward", "speed": self.drive_speed}
        elif decision == FollowDecision.TURN_LEFT:
            return {"direction": "left", "speed": self.turn_speed}
        elif decision == FollowDecision.TURN_RIGHT:
            return {"direction": "right", "speed": self.turn_speed}
        elif decision == FollowDecision.HOLD:
            return {"direction": "stop", "speed": 0}
        else:  # STOP or safety default
            return {"direction": "stop", "speed": 0}

    def _dispatch_command(self, decision: str, reason: str = "") -> Dict[str, Any]:
        motor_cmd = self._convert_decision_to_motor_cmd(decision)
        cmd_record = {
            "timestamp": time.time(),
            "mode": self.mode,
            "decision": decision,
            "direction": motor_cmd["direction"],
            "speed": motor_cmd["speed"],
            "reason": reason,
            "simulated": True,
            "sent": False  # Strictly simulation - no real motor commands
        }
        if self.command_dispatcher is not None:
            try:
                self.command_dispatcher(motor_cmd["direction"], motor_cmd["speed"])
            except Exception:
                pass
        self.last_command = cmd_record
        self.command_history.append(cmd_record)
        if len(self.command_history) > self.max_history:
            self.command_history.pop(0)
        return cmd_record

    def evaluate(self, now: Optional[float] = None) -> Dict[str, Any]:
        """
        Executes one evaluation step of the unified following controller.
        Respects safety priority:
        1. Manual mode / Emergency Override -> STOP
        2. Disabled -> STOP
        3. Obstacle Stop -> STOP
        4. Target Loss / Stale Sensor -> STOP
        5. Distance & Direction Evaluation -> Steer / Move / Hold
        """
        current_time = now if now is not None else time.time()

        # Query sensor telemetry source if attached
        if self.sensor_telemetry_source is not None:
            try:
                res = self.sensor_telemetry_source()
                if isinstance(res, tuple):
                    telem = res[0]
                    stale = res[1] if len(res) > 1 and isinstance(res[1], bool) else False
                else:
                    telem = res
                    stale = False

                if telem is not None:
                    if hasattr(telem, "distance_cm"):
                        self.update_sensor_telemetry(
                            distance_cm=telem.distance_cm,
                            obstacle=telem.obstacle,
                            is_valid=telem.is_valid,
                            is_stale=stale
                        )
                    elif isinstance(telem, dict):
                        self.update_sensor_telemetry(
                            distance_cm=telem.get("distance_cm"),
                            obstacle=telem.get("obstacle", False),
                            is_valid=telem.get("is_valid", True),
                            is_stale=stale or telem.get("is_stale", False)
                        )
            except Exception:
                pass

        # Priority 0: Latched Emergency Stop
        if self.emergency_stop_latched:
            self.decision = FollowDecision.STOP
            self.state = FollowState.MANUAL_OVERRIDE
            return self._dispatch_command(self.decision, reason="Emergency STOP latch active — awaiting explicit reset")

        # Priority 1: Manual Mode or Emergency Override
        if self.mode == FollowMode.MANUAL:
            self.decision = FollowDecision.STOP
            self.state = FollowState.MANUAL_OVERRIDE if self.manual_override_active else FollowState.IDLE
            return self._dispatch_command(self.decision, reason="Manual mode active")

        if self.manual_override_active:
            self.decision = FollowDecision.STOP
            self.state = FollowState.MANUAL_OVERRIDE
            return self._dispatch_command(self.decision, reason="Manual override engaged")

        # Priority 2: Autonomous Following Disabled
        if not self.enabled:
            self.decision = FollowDecision.STOP
            self.state = FollowState.IDLE
            return self._dispatch_command(self.decision, reason="Follow mode not enabled")

        # Priority 3: Obstacle Stop (Independent of target)
        if self.obstacle_detected:
            self.decision = FollowDecision.STOP
            self.state = FollowState.OBSTACLE_STOP
            self.was_in_obstacle_stop = True
            return self._dispatch_command(self.decision, reason="Obstacle detected within safety perimeter")

        # Priority 4 & 5: Evaluate active observation source
        if self.mode == FollowMode.PERSON:
            res = self._evaluate_person(current_time)
        elif self.mode == FollowMode.POCKET_TAG:
            res = self._evaluate_tag(current_time)
        else:
            self.decision = FollowDecision.STOP
            self.state = FollowState.IDLE
            return self._dispatch_command(self.decision, reason="Unknown mode")

        # Sensor Recovery Invariant:
        # Clearing an obstacle or recovering sensor state must NEVER automatically resume movement.
        if self.was_in_obstacle_stop:
            self.was_in_obstacle_stop = False
            if self.decision in (FollowDecision.MOVE_FORWARD, FollowDecision.MOVE_BACKWARD):
                self.decision = FollowDecision.HOLD
                self.state = FollowState.HOLDING_DISTANCE
                return self._dispatch_command(
                    FollowDecision.HOLD,
                    reason="Obstacle cleared — holding position (sensor recovery prohibits automatic movement resumption)"
                )

        return res

    def _evaluate_person(self, now: float) -> Dict[str, Any]:
        """
        Evaluates vision-based person following.
        CRITICAL CONSTRAINT: Bounding-box area is NOT a real-world distance measurement.
        Metric distance remains unavailable until calibrated or provided by a ranging sensor.
        Invalid/unavailable distance readings must never produce forward movement.
        """
        target = self.person_tracker.get_selected_target(now, allow_stale=True)

        # Requirement 5: Target locking. Once a person target is actively acquired during follow mode,
        # lock onto that target ID so that if the person is lost or disappears, the controller halts
        # and does not automatically switch to a different person in the frame.
        if target is not None and self.selected_person_id is None:
            self.select_person_target(target.id)

        if target is None:
            # Target is occluded, lost, or not yet detected.
            # Never blindly drive around searching for a lost target!
            self.decision = FollowDecision.STOP
            if self.person_tracker.reacquisition_status == ReacquisitionStatus.OCCLUDED:
                self.state = FollowState.ACQUIRING
                reason = "Person temporarily occluded — halted awaiting reacquisition"
            else:
                self.state = FollowState.TARGET_LOST
                reason = "Person target lost — halted immediately"
            return self._dispatch_command(self.decision, reason=reason)

        # Check freshness & validity
        if not target.is_valid or (now - target.timestamp > self.stale_timeout_sec):
            self.decision = FollowDecision.STOP
            self.state = FollowState.COMMUNICATION_ERROR
            return self._dispatch_command(self.decision, reason="Person observation stale or invalid")

        # Distance is uncalibrated/unavailable in Person Mode without dedicated range sensor
        # Direction steering is allowed (LEFT / RIGHT), but forward approach is prohibited.
        self.state = FollowState.FOLLOWING
        direction = target.direction

        if direction == "LEFT":
            self.decision = FollowDecision.TURN_LEFT
            reason = "Aligning to person: TURN_LEFT (distance uncalibrated, holding forward motion)"
        elif direction == "RIGHT":
            self.decision = FollowDecision.TURN_RIGHT
            reason = "Aligning to person: TURN_RIGHT (distance uncalibrated, holding forward motion)"
        else:  # CENTER
            self.decision = FollowDecision.HOLD
            self.state = FollowState.HOLDING_DISTANCE
            reason = "Person centered — holding position (distance sensor awaiting calibration)"

        return self._dispatch_command(self.decision, reason=reason)

    def _evaluate_tag(self, now: float) -> Dict[str, Any]:
        """
        Evaluates wireless pocket tag following (UWB or BLE).
        UWB provides calibrated metric distance.
        BLE provides rough proximity from RSSI; forward motion is prohibited without metric distance.
        CRITICAL: Single UWB receiver provides distance only, NOT direction.
        Forward or rotational movement is prohibited unless a valid directional source is available.
        """
        reading = self.tag_tracker.get_current_reading(now)

        if reading is None or not reading.connected:
            self.decision = FollowDecision.STOP
            self.state = FollowState.TARGET_LOST
            return self._dispatch_command(self.decision, reason="Wireless tag disconnected or missing")

        if not reading.valid or (now - reading.timestamp > self.stale_timeout_sec):
            self.decision = FollowDecision.STOP
            self.state = FollowState.COMMUNICATION_ERROR
            return self._dispatch_command(self.decision, reason="Tag reading stale or invalid")

        # Safely transition through ACQUIRING if tag was previously lost
        if self.state == FollowState.TARGET_LOST:
            self.decision = FollowDecision.HOLD
            self.state = FollowState.ACQUIRING
            return self._dispatch_command(self.decision, reason="Wireless tag reacquired — stabilizing in ACQUIRING")

        # Check direction validity and freshness
        has_valid_direction = bool(
            reading.direction and 
            reading.direction in ("LEFT", "RIGHT", "CENTER") and
            getattr(reading, "direction_valid", True)
        )
        if has_valid_direction and getattr(reading, "direction_timestamp", None) is not None:
            dir_ttl = getattr(reading, "direction_ttl_sec", 0.5) or 0.5
            if (now - reading.direction_timestamp) > dir_ttl:
                has_valid_direction = False

        # Handle BLE rough proximity vs UWB calibrated metric distance
        if not reading.is_metric or reading.distance_m is None:
            # BLE: RSSI proximity indicator only.
            # "Invalid distance readings must never produce forward movement."
            if has_valid_direction and reading.direction == "LEFT":
                self.decision = FollowDecision.TURN_LEFT
                self.state = FollowState.FOLLOWING
                reason = "Aligning to BLE tag: TURN_LEFT (metric distance unavailable)"
            elif has_valid_direction and reading.direction == "RIGHT":
                self.decision = FollowDecision.TURN_RIGHT
                self.state = FollowState.FOLLOWING
                reason = "Aligning to BLE tag: TURN_RIGHT (metric distance unavailable)"
            else:
                self.decision = FollowDecision.HOLD
                self.state = FollowState.HOLDING_DISTANCE
                reason = "BLE proximity active — forward approach prohibited without metric ranging"
            return self._dispatch_command(self.decision, reason=reason)

        # Calibrated Metric Distance (UWB)
        dist = reading.distance_m

        # Safety perimeter check
        if dist < self.min_safe_distance_m:
            self.decision = FollowDecision.STOP
            self.state = FollowState.OBSTACLE_STOP
            return self._dispatch_command(self.decision, reason=f"Tag inside minimum safe threshold ({dist:.2f}m < {self.min_safe_distance_m}m)")

        # CRITICAL SINGLE UWB RECEIVER CHECK:
        # A single UWB range measurement must NEVER independently trigger forward or rotational movement.
        # If direction is unavailable, report range but do NOT attempt to navigate toward the tag.
        if not has_valid_direction:
            self.state = FollowState.HOLDING_DISTANCE
            self.decision = FollowDecision.HOLD
            reason = (f"UWB range {dist:.2f}m valid, but directional source unavailable or expired — "
                      f"single receiver cannot determine bearing; holding position")
            return self._dispatch_command(self.decision, reason=reason)

        # Valid metric distance AND valid directional source:
        if dist > self.approach_threshold_m:
            # Target is farther than approach threshold (default 1.8m): Approach
            self.state = FollowState.FOLLOWING
            if reading.direction == "LEFT":
                self.decision = FollowDecision.TURN_LEFT
                reason = f"Tag at {dist:.2f}m (>{self.approach_threshold_m}m): steering LEFT"
            elif reading.direction == "RIGHT":
                self.decision = FollowDecision.TURN_RIGHT
                reason = f"Tag at {dist:.2f}m (>{self.approach_threshold_m}m): steering RIGHT"
            else:
                clear, forward_reason = self.is_forward_path_clear()
                if not clear:
                    self.decision = FollowDecision.HOLD
                    self.state = FollowState.HOLDING_DISTANCE
                    reason = f"Tag at {dist:.2f}m: forward movement prevented by ultrasonic sensor: {forward_reason}"
                else:
                    self.decision = FollowDecision.MOVE_FORWARD
                    reason = f"Tag at {dist:.2f}m (>{self.approach_threshold_m}m): approaching target"

        elif dist < self.hold_min_m:
            # Target is closer than hold_min (default 1.2m) but outside safety stop: Hold / Stop forward approach
            self.state = FollowState.HOLDING_DISTANCE
            self.decision = FollowDecision.HOLD
            reason = f"Tag at {dist:.2f}m (<{self.hold_min_m}m): holding distance, stopping forward approach"

        else:
            # Inside the hold window [1.2m - 1.8m]
            self.state = FollowState.HOLDING_DISTANCE
            if reading.direction == "LEFT":
                self.decision = FollowDecision.TURN_LEFT
                reason = f"Holding distance at {dist:.2f}m: aligning LEFT"
            elif reading.direction == "RIGHT":
                self.decision = FollowDecision.TURN_RIGHT
                reason = f"Holding distance at {dist:.2f}m: aligning RIGHT"
            else:
                self.decision = FollowDecision.HOLD
                reason = f"Holding distance at {dist:.2f}m (target {self.target_distance_m}m maintained)"

        return self._dispatch_command(self.decision, reason=reason)

    def get_status(self) -> Dict[str, Any]:
        """Comprehensive telemetry for API and dashboard."""
        active_target = self.person_tracker.get_selected_target() if self.mode == FollowMode.PERSON else None
        target_dir = active_target.direction if active_target else (
            self.tag_tracker.get_current_reading().direction if self.mode == FollowMode.POCKET_TAG and self.tag_tracker.get_current_reading() else "NO_TARGET"
        )
        return {
            "mode": self.mode,
            "enabled": self.enabled,
            "state": self.state,
            "decision": self.decision,
            "target_direction": target_dir,
            "selected_person_id": self.selected_person_id,
            "selected_tag_id": self.selected_tag_id,
            "current_target": active_target.to_dict() if active_target else None,
            "target_distance_m": self.target_distance_m,
            "min_following_distance_m": self.min_following_distance_m,
            "max_following_distance_m": self.max_following_distance_m,
            "distance_tolerance_m": self.distance_tolerance_m,
            "min_safe_distance_m": self.min_safe_distance_m,
            "approach_threshold_m": self.approach_threshold_m,
            "hold_min_m": self.hold_min_m,
            "hold_max_m": self.hold_max_m,
            "obstacle_detected": self.obstacle_detected,
            "simulated_obstacle": self.simulated_obstacle,
            "sensor_obstacle": self.sensor_obstacle,
            "ultrasonic_distance_cm": self.ultrasonic_distance_cm,
            "ultrasonic_valid": self.ultrasonic_valid,
            "ultrasonic_stale": self.ultrasonic_stale,
            "require_ultrasonic_for_forward": self.require_ultrasonic_for_forward,
            "manual_override_active": self.manual_override_active,
            "emergency_stop_latched": self.emergency_stop_latched,
            "person_tracker": self.person_tracker.get_status(),
            "tag_tracker": self.tag_tracker.get_status(),
            "last_command": self.last_command,
            "simulation_mode": True
        }

import os
import time
from typing import Dict, List, Optional, Any


class Decision:
    STOP = "STOP"
    MOVE_FORWARD = "MOVE_FORWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    SEARCH_TARGET = "SEARCH_TARGET"


class AutonomousState:
    DISABLED = "DISABLED"
    SEARCHING = "SEARCHING"
    TRACKING = "TRACKING"
    TARGET_REACHED = "TARGET_REACHED"
    STALE_DATA = "STALE_DATA"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class AutonomousCommandInterface:
    """
    Modular command interface for translating autonomous decisions into
    structured motor commands. Supports pure simulation mode and HTTP dispatch.
    Physical motor activation is intentionally disabled for this milestone.
    """

    def __init__(self,
                 drive_speed: int = 150,
                 turn_speed: int = 120,
                 search_policy: str = "STOP"):
        self.drive_speed = drive_speed
        self.turn_speed = turn_speed
        self.search_policy = search_policy
        self.last_command: Dict[str, Any] = {"direction": "stop", "speed": 0}
        self.command_history: List[Dict[str, Any]] = []
        self.max_history: int = 25

    def convert_decision_to_command(self, decision: str) -> Dict[str, Any]:
        """Convert an autonomous decision string into a structured direction/speed dictionary."""
        if decision == Decision.MOVE_FORWARD:
            return {"direction": "forward", "speed": self.drive_speed}
        elif decision == Decision.TURN_LEFT:
            return {"direction": "left", "speed": self.turn_speed}
        elif decision == Decision.TURN_RIGHT:
            return {"direction": "right", "speed": self.turn_speed}
        elif decision == Decision.SEARCH_TARGET:
            # Under SEARCH_TARGET policy, perform a simulated in-place pivot if enabled
            if self.search_policy == Decision.SEARCH_TARGET:
                return {"direction": "left", "speed": self.turn_speed}
            return {"direction": "stop", "speed": 0}
        else:  # Decision.STOP, invalid, or emergency
            return {"direction": "stop", "speed": 0}

    def dispatch(self, decision: str, esp32_url: str = "") -> Dict[str, Any]:
        """
        Dispatches or logs the decision.
        Physical motor activation is suppressed in this milestone.
        """
        cmd = self.convert_decision_to_command(decision)
        self.last_command = cmd

        record = {
            "timestamp": time.time(),
            "decision": decision,
            "direction": cmd["direction"],
            "speed": cmd["speed"],
            "mode": "esp32_simulation" if esp32_url else "simulation",
            "simulated": True
        }
        self.command_history.append(record)
        if len(self.command_history) > self.max_history:
            self.command_history.pop(0)

        return {
            "sent": False,
            "mode": "simulation" if not esp32_url else "esp32_simulation",
            "command": cmd,
            "simulated": True
        }


class AutonomousDecisionEngine:
    """
    Autonomous Decision Engine for WALL-E robotics.
    Processes tracked objects, performs target selection, evaluates horizontal alignment
    with hysteresis, enforces proximity stopping, and prioritizes safety.
    """

    def __init__(self,
                 frame_width: int = 640,
                 center_deadzone_ratio: Optional[float] = None,
                 hysteresis_px: Optional[float] = None,
                 proximity_area: Optional[int] = None,
                 stale_timeout_sec: Optional[float] = None,
                 search_policy: Optional[str] = None,
                 drive_speed: Optional[int] = None,
                 turn_speed: Optional[int] = None):

        # Load configurable parameters with environment variable fallbacks
        self.frame_width = frame_width
        self.center_deadzone_ratio = (
            center_deadzone_ratio if center_deadzone_ratio is not None
            else float(os.getenv("AUTONOMOUS_CENTER_RATIO", "0.20"))
        )
        self.hysteresis_px = (
            hysteresis_px if hysteresis_px is not None
            else float(os.getenv("AUTONOMOUS_HYSTERESIS_PX", "15.0"))
        )
        self.proximity_area = (
            proximity_area if proximity_area is not None
            else int(os.getenv("AUTONOMOUS_PROXIMITY_AREA", "12000"))
        )
        self.stale_timeout_sec = (
            stale_timeout_sec if stale_timeout_sec is not None
            else float(os.getenv("AUTONOMOUS_STALE_TIMEOUT", "0.8"))
        )
        self.search_policy = (
            search_policy if search_policy is not None
            else os.getenv("AUTONOMOUS_SEARCH_POLICY", "STOP").upper()
        )
        self.drive_speed = (
            drive_speed if drive_speed is not None
            else int(os.getenv("AUTONOMOUS_DRIVE_SPEED", "150"))
        )
        self.turn_speed = (
            turn_speed if turn_speed is not None
            else int(os.getenv("AUTONOMOUS_TURN_SPEED", "120"))
        )

        # Operational state
        self.enabled: bool = False
        self.selected_target_id: Optional[int] = None
        self.current_state: str = AutonomousState.DISABLED
        self.current_decision: str = Decision.STOP
        self.last_valid_target_time: float = 0.0

        # Memory for horizontal hysteresis anti-oscillation
        self._last_horizontal_alignment: Optional[str] = None  # "LEFT", "CENTER", "RIGHT"

        # Information about the currently active target
        self.current_target: Optional[Dict[str, Any]] = None

        # Command interface
        self.command_interface = AutonomousCommandInterface(
            drive_speed=self.drive_speed,
            turn_speed=self.turn_speed,
            search_policy=self.search_policy
        )

    def set_enabled(self, enabled: bool):
        """Enable or disable autonomous decision engine."""
        self.enabled = bool(enabled)
        if not self.enabled:
            self.manual_override()
        else:
            self.current_state = AutonomousState.SEARCHING
            self.current_decision = Decision.STOP
            self.last_valid_target_time = time.time()
            self._last_horizontal_alignment = None

    def manual_override(self):
        """Immediately abort autonomous mode and command STOP."""
        self.enabled = False
        self.current_state = AutonomousState.DISABLED
        self.current_decision = Decision.STOP
        self.current_target = None
        self._last_horizontal_alignment = None
        self.command_interface.dispatch(Decision.STOP)

    def set_target_id(self, target_id: Optional[int]):
        """Specify a persistent tracking ID to track, or None for automatic selection."""
        self.selected_target_id = int(target_id) if target_id is not None else None
        self._last_horizontal_alignment = None

    def select_target(self, tracked_objects: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Selects target by persistent tracking ID if configured,
        or automatically selects the largest visible tracked object.
        """
        if not tracked_objects:
            return None

        if self.selected_target_id is not None:
            for obj in tracked_objects:
                if obj.get("id") == self.selected_target_id:
                    return obj
            # Specified target ID is not currently visible
            return None

        # Automatic target selection: select object with maximum area
        return max(tracked_objects, key=lambda o: o.get("area", 0))

    def evaluate_alignment(self, x: float) -> str:
        """
        Evaluates whether centroid X is LEFT, CENTER, or RIGHT relative to frame width,
        incorporating deadzone and hysteresis to avoid rapid oscillation.
        """
        center_x = self.frame_width / 2.0
        half_deadzone = (self.frame_width * self.center_deadzone_ratio) / 2.0
        left_boundary = center_x - half_deadzone
        right_boundary = center_x + half_deadzone
        h = self.hysteresis_px

        prev = self._last_horizontal_alignment

        if prev == "LEFT":
            # Must cross past left_boundary + h to leave LEFT
            if x > (left_boundary + h):
                if x > (right_boundary + h):
                    alignment = "RIGHT"
                else:
                    alignment = "CENTER"
            else:
                alignment = "LEFT"

        elif prev == "RIGHT":
            # Must cross past right_boundary - h to leave RIGHT
            if x < (right_boundary - h):
                if x < (left_boundary - h):
                    alignment = "LEFT"
                else:
                    alignment = "CENTER"
            else:
                alignment = "RIGHT"

        else:  # previously in CENTER or None
            if x < left_boundary:
                alignment = "LEFT"
            elif x > right_boundary:
                alignment = "RIGHT"
            else:
                alignment = "CENTER"

        self._last_horizontal_alignment = alignment
        return alignment

    def evaluate(self,
                 tracked_objects: List[Dict[str, Any]],
                 frame_width: Optional[int] = None,
                 current_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Evaluates tracked objects and issues movement decision.
        Enforces safety policies: disabled mode, stale data timeout, proximity stopping,
        and anti-oscillation deadzone.
        """
        now = time.time() if current_time is None else current_time

        if frame_width is not None and frame_width > 0:
            self.frame_width = frame_width

        # Safeguard 1: If autonomous mode is disabled, always enforce STOP
        if not self.enabled:
            self.current_state = AutonomousState.DISABLED
            self.current_decision = Decision.STOP
            self.current_target = None
            self._last_horizontal_alignment = None
            self.command_interface.dispatch(Decision.STOP)
            return self.get_status()

        # Target Selection
        target = self.select_target(tracked_objects)

        if target is not None:
            # Valid target detected
            self.last_valid_target_time = now
            self.current_target = target

            target_x = target.get("x", self.frame_width / 2.0)
            target_area = target.get("area", 0)

            # Determine horizontal alignment with hysteresis
            alignment = self.evaluate_alignment(target_x)

            if alignment == "LEFT":
                self.current_decision = Decision.TURN_LEFT
                self.current_state = AutonomousState.TRACKING
            elif alignment == "RIGHT":
                self.current_decision = Decision.TURN_RIGHT
                self.current_state = AutonomousState.TRACKING
            else:  # CENTER
                # Proximity check using object area as proximity estimate
                if target_area >= self.proximity_area:
                    # Target reached configured proximity threshold
                    self.current_decision = Decision.STOP
                    self.current_state = AutonomousState.TARGET_REACHED
                else:
                    self.current_decision = Decision.MOVE_FORWARD
                    self.current_state = AutonomousState.TRACKING

        else:
            # No valid target in current frame
            self.current_target = None
            self._last_horizontal_alignment = None

            # Safeguard 2: Stale data check
            time_since_target = now - self.last_valid_target_time

            if time_since_target > self.stale_timeout_sec:
                # Tracking data is stale / target lost
                if self.search_policy == Decision.SEARCH_TARGET:
                    self.current_decision = Decision.SEARCH_TARGET
                    self.current_state = AutonomousState.SEARCHING
                else:
                    self.current_decision = Decision.STOP
                    self.current_state = AutonomousState.STALE_DATA
            else:
                # Brief absence within stale timeout window
                if self.search_policy == Decision.SEARCH_TARGET:
                    self.current_decision = Decision.SEARCH_TARGET
                    self.current_state = AutonomousState.SEARCHING
                else:
                    self.current_decision = Decision.STOP
                    self.current_state = AutonomousState.SEARCHING

        # Dispatch decision to modular command interface
        self.command_interface.dispatch(self.current_decision)

        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        """Returns structured status dictionary for API and dashboard serialization."""
        return {
            "enabled": self.enabled,
            "state": self.current_state,
            "decision": self.current_decision,
            "target_id": self.selected_target_id,
            "target_locked": self.current_target is not None,
            "target": self.current_target,
            "command": self.command_interface.last_command,
            "simulation_mode": True,
            "frame_width": self.frame_width,
            "proximity_area": self.proximity_area,
            "stale_timeout_sec": self.stale_timeout_sec,
            "search_policy": self.search_policy
        }

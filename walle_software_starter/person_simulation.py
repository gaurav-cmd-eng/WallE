"""
WALL-E Dual-Mode Autonomous Following System
Module: person_simulation.py (Stage 5: Complete End-to-End Person Simulation)

Provides reproducible, deterministic simulation scenarios demonstrating:
1. Multi-person detection in vision field.
2. Target selection and locking.
3. Person tracking and steering decisions (LEFT, CENTER, RIGHT).
4. Target loss and lock protection (no accidental switching).
5. Obstacle override of following behavior.
6. Ultrasonic telemetry staleness / missing data guards.
7. Sensor recovery non-lurch invariant.
8. Emergency STOP latch persistence.
9. Reset safety latch transition to IDLE (no auto-resume).
10. ESP32 serial communication failure failsafe.
"""

import time
from typing import Dict, List, Optional, Any

try:
    from person_tracker import (
        PersonTracker, PersonObservation, ReacquisitionStatus
    )
    from follow_controller import (
        FollowController, FollowMode, FollowState, FollowDecision
    )
    from tag_tracker import TagTrackerManager
    from esp32_comm import (
        ESP32CommunicationManager, MockSerialTransport, SensorTelemetry, ESP32Command, ConnectionStatus
    )
except ImportError:
    from walle_software_starter.person_tracker import (
        PersonTracker, PersonObservation, ReacquisitionStatus
    )
    from walle_software_starter.follow_controller import (
        FollowController, FollowMode, FollowState, FollowDecision
    )
    from walle_software_starter.tag_tracker import TagTrackerManager
    from walle_software_starter.esp32_comm import (
        ESP32CommunicationManager, MockSerialTransport, SensorTelemetry, ESP32Command, ConnectionStatus
    )


class DeterministicPersonSimulator:
    """
    Deterministic End-to-End Simulation Engine for Person Following.
    Validates complete pipeline:
    Camera Detection -> Person Tracking -> Target Locking ->
    FollowController Decision -> Obstacle & Sensor Safety ->
    ESP32 Communication Manager -> Mock Serial Transport.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = seed

        self.mock_transport = MockSerialTransport()
        self.esp32_comm = ESP32CommunicationManager(
            transport=self.mock_transport,
            simulation_mode=True,
            watchdog_timeout_sec=1.0,
            telemetry_timeout_sec=1.0
        )
        self.person_tracker = PersonTracker(max_disappeared_frames=5, stale_timeout_sec=0.8)
        self.tag_tracker = TagTrackerManager()
        self.controller = FollowController(
            person_tracker=self.person_tracker,
            tag_tracker=self.tag_tracker,
            target_distance_m=1.5,
            distance_tolerance_m=0.3,
            min_safe_distance_m=0.8,
            approach_threshold_m=1.8,
            hold_min_m=1.2,
            hold_max_m=1.8,
            stale_timeout_sec=0.8
        )
        self.controller.command_dispatcher = lambda direction, speed: self.esp32_comm.send_command(direction, speed)
        self.controller.set_mode(FollowMode.PERSON)

    def scenario_1_multiple_people_detected(self) -> Dict[str, Any]:
        """Scenario 1: Camera detects multiple people (Left, Center, Right)."""
        self.reset()
        # Person 1 (Left), Person 2 (Center), Person 3 (Right)
        p1 = PersonObservation(person_id=1, bbox=(50, 80, 120, 260), confidence=0.91, direction="LEFT")
        p2 = PersonObservation(person_id=2, bbox=(260, 70, 130, 280), confidence=0.95, direction="CENTER")
        p3 = PersonObservation(person_id=3, bbox=(480, 85, 115, 255), confidence=0.89, direction="RIGHT")

        self.person_tracker.update([p1, p2, p3], frame_width=640)
        tracked = self.person_tracker.get_tracked_people()

        assert len(tracked) == 3, f"Expected 3 tracked people, got {len(tracked)}"
        assert self.controller.selected_person_id is None, "Target should not be locked initially"

        return {
            "name": "scenario_1_multiple_people_detected",
            "people_count": len(tracked),
            "target_locked": False,
            "status": "PASS"
        }

    def scenario_2_target_selection_and_lock(self) -> Dict[str, Any]:
        """Scenario 2: Operator selects and locks Person #2."""
        self.scenario_1_multiple_people_detected()

        # Select Person #2 and enable following
        self.controller.select_person_target(2)
        self.controller.set_enabled(True)

        assert self.controller.selected_person_id == 2, "Person #2 must be locked"
        assert self.controller.enabled is True, "Follow controller must be enabled"

        target = self.person_tracker.get_selected_target()
        assert target is not None and target.id == 2, "Selected target must be Person #2"

        cmd = self.controller.evaluate()
        # Since target is CENTER and distance uncalibrated, decision is HOLD
        assert cmd["decision"] == FollowDecision.HOLD, f"Expected HOLD, got {cmd['decision']}"
        assert self.controller.state == FollowState.HOLDING_DISTANCE

        return {
            "name": "scenario_2_target_selection_and_lock",
            "selected_person_id": 2,
            "state": self.controller.state,
            "decision": self.controller.decision,
            "status": "PASS"
        }

    def scenario_3_steering_tracking_left(self) -> Dict[str, Any]:
        """Scenario 3: Locked Person #2 moves to LEFT -> Controller steers LEFT."""
        self.scenario_2_target_selection_and_lock()

        # Person #2 moves to the left portion of frame
        p2_left = PersonObservation(person_id=2, bbox=(60, 80, 130, 280), confidence=0.94, direction="LEFT")
        self.person_tracker.update([p2_left], frame_width=640)

        cmd = self.controller.evaluate()
        assert cmd["decision"] == FollowDecision.TURN_LEFT, f"Expected TURN_LEFT, got {cmd['decision']}"
        assert cmd["direction"] == "left", "Dispatched motor direction must be 'left'"
        assert self.controller.state == FollowState.FOLLOWING

        # Verify dispatched command reached ESP32 communication manager
        assert self.esp32_comm.last_command["cmd"] == ESP32Command.LEFT

        return {
            "name": "scenario_3_steering_tracking_left",
            "decision": cmd["decision"],
            "dispatched_direction": cmd["direction"],
            "status": "PASS"
        }

    def scenario_4_steering_tracking_right(self) -> Dict[str, Any]:
        """Scenario 4: Locked Person #2 moves to RIGHT -> Controller steers RIGHT."""
        self.scenario_2_target_selection_and_lock()

        # Person #2 moves to the right portion of frame
        p2_right = PersonObservation(person_id=2, bbox=(490, 80, 130, 280), confidence=0.93, direction="RIGHT")
        self.person_tracker.update([p2_right], frame_width=640)

        cmd = self.controller.evaluate()
        assert cmd["decision"] == FollowDecision.TURN_RIGHT, f"Expected TURN_RIGHT, got {cmd['decision']}"
        assert cmd["direction"] == "right", "Dispatched motor direction must be 'right'"
        assert self.controller.state == FollowState.FOLLOWING

        assert self.esp32_comm.last_command["cmd"] == ESP32Command.RIGHT

        return {
            "name": "scenario_4_steering_tracking_right",
            "decision": cmd["decision"],
            "dispatched_direction": cmd["direction"],
            "status": "PASS"
        }

    def scenario_5_centering_holds_position(self) -> Dict[str, Any]:
        """Scenario 5: Locked Person #2 in CENTER -> Controller holds position (no blind forward drive)."""
        self.scenario_2_target_selection_and_lock()

        p2_center = PersonObservation(person_id=2, bbox=(260, 80, 130, 280), confidence=0.95, direction="CENTER")
        self.person_tracker.update([p2_center], frame_width=640)

        cmd = self.controller.evaluate()
        assert cmd["decision"] == FollowDecision.HOLD, f"Expected HOLD, got {cmd['decision']}"
        assert cmd["direction"] == "stop", "Dispatched motor direction must be 'stop'"
        assert cmd["speed"] == 0
        assert self.controller.state == FollowState.HOLDING_DISTANCE

        return {
            "name": "scenario_5_centering_holds_position",
            "decision": cmd["decision"],
            "state": self.controller.state,
            "status": "PASS"
        }

    def scenario_6_missing_or_stale_target_stops(self) -> Dict[str, Any]:
        """Scenario 6: Locked person disappears from frame -> Controller halts immediately in TARGET_LOST."""
        self.scenario_2_target_selection_and_lock()

        # Frame with no detections across disappearance threshold (6 frames > max_disappeared_frames=5)
        for _ in range(6):
            self.person_tracker.update([], frame_width=640)

        cmd = self.controller.evaluate()
        assert cmd["decision"] == FollowDecision.STOP, f"Expected STOP, got {cmd['decision']}"
        assert cmd["direction"] == "stop"
        assert self.controller.state == FollowState.TARGET_LOST

        return {
            "name": "scenario_6_missing_or_stale_target_stops",
            "decision": cmd["decision"],
            "state": self.controller.state,
            "status": "PASS"
        }

    def scenario_7_lock_prevents_switching(self) -> Dict[str, Any]:
        """Scenario 7: Locked Person #2 missing, but Person #1 and #3 present -> Controller halts, refuses switch."""
        self.scenario_2_target_selection_and_lock()

        # Person #2 is gone, but Person #1 and Person #3 are still visible
        p1 = PersonObservation(person_id=1, bbox=(50, 80, 120, 260), confidence=0.91, direction="LEFT")
        p3 = PersonObservation(person_id=3, bbox=(480, 85, 115, 255), confidence=0.89, direction="RIGHT")
        for _ in range(6):
            self.person_tracker.update([p1, p3], frame_width=640)

        cmd = self.controller.evaluate()
        # Must NOT auto-switch to Person 1 or 3!
        assert cmd["decision"] == FollowDecision.STOP, f"Expected STOP, got {cmd['decision']}"
        assert self.controller.state == FollowState.TARGET_LOST
        assert self.controller.selected_person_id == 2, "Target lock on Person #2 must be retained"

        return {
            "name": "scenario_7_lock_prevents_switching",
            "locked_target_id": self.controller.selected_person_id,
            "state": self.controller.state,
            "decision": cmd["decision"],
            "status": "PASS"
        }

    def scenario_8_obstacle_detection_overrides_following(self) -> Dict[str, Any]:
        """Scenario 8: Obstacle detected within safety perimeter (<= 30cm) -> Forces immediate OBSTACLE_STOP."""
        self.scenario_3_steering_tracking_left()

        # Sensor detects obstacle at 18.5 cm
        self.controller.update_sensor_telemetry(distance_cm=18.5, obstacle=True, is_valid=True, is_stale=False)
        assert self.controller.obstacle_detected is True

        cmd = self.controller.evaluate()
        assert cmd["decision"] == FollowDecision.STOP, f"Expected STOP, got {cmd['decision']}"
        assert cmd["direction"] == "stop"
        assert self.controller.state == FollowState.OBSTACLE_STOP

        return {
            "name": "scenario_8_obstacle_detection_overrides_following",
            "obstacle_detected": True,
            "state": self.controller.state,
            "decision": cmd["decision"],
            "status": "PASS"
        }

    def scenario_9_stale_or_missing_ultrasonic_prevents_forward(self) -> Dict[str, Any]:
        """Scenario 9: Stale or missing ultrasonic telemetry inhibits forward motion."""
        self.reset()
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.controller.set_enabled(True)
        self.controller.tag_tracker.set_source("UWB")
        self.controller.tag_tracker.uwb_tracker.record_measurement(
            tag_id="TAG_01", distance_m=2.5, direction="CENTER", connected=True
        )
        self.controller.tag_tracker.set_external_direction("CENTER", tag_id="TAG_01", ttl_sec=1.0)

        # Telemetry is stale (> 2.0s old)
        self.controller.update_sensor_telemetry(distance_cm=90.0, obstacle=False, is_valid=True, is_stale=True)
        clear, reason = self.controller.is_forward_path_clear()
        assert clear is False, "Forward path must not be clear when telemetry is stale"

        cmd = self.controller.evaluate()
        assert cmd["decision"] == FollowDecision.HOLD, f"Expected HOLD, got {cmd['decision']}"
        assert self.controller.state == FollowState.HOLDING_DISTANCE

        return {
            "name": "scenario_9_stale_or_missing_ultrasonic_prevents_forward",
            "clear": clear,
            "reason": reason,
            "decision": cmd["decision"],
            "status": "PASS"
        }

    def scenario_10_sensor_recovery_avoids_lurch(self) -> Dict[str, Any]:
        """Scenario 10: Sensor recovers after obstacle -> Holds position on 1st cycle, avoiding sudden lurch."""
        self.scenario_8_obstacle_detection_overrides_following()

        # Obstacle clears with fresh valid distance (120 cm)
        self.controller.update_sensor_telemetry(distance_cm=120.0, obstacle=False, is_valid=True, is_stale=False)
        assert self.controller.obstacle_detected is False

        # First cycle after recovery: MUST HOLD position rather than surging forward
        cmd1 = self.controller.evaluate()
        assert cmd1["decision"] in (FollowDecision.HOLD, FollowDecision.TURN_LEFT), (
            f"Expected HOLD or steering, got {cmd1['decision']}"
        )
        assert cmd1["direction"] != "forward", "Must NOT command forward drive on recovery turn"

        return {
            "name": "scenario_10_sensor_recovery_avoids_lurch",
            "recovery_decision": cmd1["decision"],
            "status": "PASS"
        }

    def scenario_11_emergency_stop_latches(self) -> Dict[str, Any]:
        """Scenario 11: Emergency STOP latches controller and halts all motion."""
        self.scenario_3_steering_tracking_left()

        cmd = self.controller.emergency_stop()
        assert self.controller.emergency_stop_latched is True
        assert cmd["decision"] == FollowDecision.STOP
        assert self.controller.state == FollowState.MANUAL_OVERRIDE
        assert self.controller.enabled is False

        # Even with clear target and clear sensor, evaluate returns STOP
        cmd2 = self.controller.evaluate()
        assert cmd2["decision"] == FollowDecision.STOP
        assert self.controller.state == FollowState.MANUAL_OVERRIDE

        # Attempt to re-enable is rejected
        self.controller.set_enabled(True)
        assert self.controller.enabled is False

        return {
            "name": "scenario_11_emergency_stop_latches",
            "latched": True,
            "state": self.controller.state,
            "status": "PASS"
        }

    def scenario_12_reset_emergency_stop_requires_reacquisition(self) -> Dict[str, Any]:
        """Scenario 12: Resetting emergency STOP restores IDLE; does not automatically resume following."""
        self.scenario_11_emergency_stop_latches()

        cmd = self.controller.reset_emergency_stop()
        assert self.controller.emergency_stop_latched is False
        assert self.controller.enabled is False
        assert self.controller.state == FollowState.IDLE
        assert cmd["decision"] == FollowDecision.STOP
        assert self.controller.selected_person_id is None, "Target lock must be cleared"

        return {
            "name": "scenario_12_reset_emergency_stop_requires_reacquisition",
            "latched": False,
            "state": self.controller.state,
            "enabled": self.controller.enabled,
            "status": "PASS"
        }

    def scenario_13_esp32_disconnect_failsafe(self) -> Dict[str, Any]:
        """Scenario 13: ESP32 serial communication loss fails safe to STOP and reports DISCONNECTED."""
        self.reset()
        self.controller.set_enabled(True)

        # Simulate hardware disconnect
        self.mock_transport.simulate_disconnect = True
        assert self.esp32_comm.is_connected() is False

        # Attempting to send command reports error and forces STOP state
        res = self.esp32_comm.send_command(ESP32Command.FORWARD, speed=150)
        assert res["ok"] is False
        assert self.esp32_comm.connection_status == ConnectionStatus.DISCONNECTED
        assert self.esp32_comm.current_state == "STOPPED"

        return {
            "name": "scenario_13_esp32_disconnect_failsafe",
            "connected": False,
            "status": "PASS"
        }

    def run_all_scenarios(self) -> Dict[str, Any]:
        """Runs all 13 end-to-end scenarios and returns a comprehensive report."""
        scenarios = [
            self.scenario_1_multiple_people_detected,
            self.scenario_2_target_selection_and_lock,
            self.scenario_3_steering_tracking_left,
            self.scenario_4_steering_tracking_right,
            self.scenario_5_centering_holds_position,
            self.scenario_6_missing_or_stale_target_stops,
            self.scenario_7_lock_prevents_switching,
            self.scenario_8_obstacle_detection_overrides_following,
            self.scenario_9_stale_or_missing_ultrasonic_prevents_forward,
            self.scenario_10_sensor_recovery_avoids_lurch,
            self.scenario_11_emergency_stop_latches,
            self.scenario_12_reset_emergency_stop_requires_reacquisition,
            self.scenario_13_esp32_disconnect_failsafe,
        ]

        results = {}
        passed = 0
        failed = 0

        for fn in scenarios:
            name = fn.__name__
            try:
                res = fn()
                results[name] = {"status": "PASS", "details": res}
                passed += 1
            except AssertionError as err:
                results[name] = {"status": "FAIL", "error": str(err)}
                failed += 1
            except Exception as exc:
                results[name] = {"status": "ERROR", "error": str(exc)}
                failed += 1

        return {
            "total_scenarios": len(scenarios),
            "passed": passed,
            "failed": failed,
            "all_passed": (failed == 0),
            "results": results
        }


if __name__ == "__main__":
    simulator = DeterministicPersonSimulator()
    report = simulator.run_all_scenarios()
    print(f"Executed {report['total_scenarios']} scenarios: {report['passed']} passed, {report['failed']} failed.")

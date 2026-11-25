import os
import sys
import time
import unittest

# Ensure walle_software_starter is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(BASE_DIR, "walle_software_starter")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from person_tracker import (
    PersonTracker, PersonObservation, AppearanceReID, ReacquisitionStatus
)
from tag_tracker import (
    TagTrackerManager, UWBTagTracker, BLETagTracker, RadioSource, TagReading, SimulatedTagDevice
)
from follow_controller import (
    FollowController, FollowMode, FollowState, FollowDecision
)
import main
from main import (
    FollowModeCommand, FollowEnableCommand, FollowPersonTargetCommand,
    FollowTagTargetCommand, FollowTagSourceCommand, SimTagCommand, SimPersonCommand
)


class TestFollowSystem(unittest.TestCase):
    """
    Unit and integration tests for the Dual-Mode Autonomous Following System.
    Validates Person Follow, Pocket Tag Follow, Manual Mode, safety policies,
    and simulation constraints.
    """

    def setUp(self):
        self.person_tracker = PersonTracker(max_disappeared_frames=5, stale_timeout_sec=0.8)
        self.tag_tracker = TagTrackerManager()
        self.controller = FollowController(
            person_tracker=self.person_tracker,
            tag_tracker=self.tag_tracker,
            target_distance_m=1.5,
            distance_tolerance_m=0.3,
            min_safe_distance_m=0.8,
            stale_timeout_sec=0.8
        )
        self.sim_tag = SimulatedTagDevice(self.tag_tracker)

    # 1. Person mode activation and deactivation
    def test_person_mode_activation_and_deactivation(self):
        self.controller.set_mode(FollowMode.PERSON)
        self.assertEqual(self.controller.mode, FollowMode.PERSON)
        self.assertFalse(self.controller.enabled)

        # Enable follow
        self.controller.set_enabled(True)
        self.assertTrue(self.controller.enabled)
        self.assertEqual(self.controller.state, FollowState.ACQUIRING)

        # Deactivate follow
        self.controller.set_enabled(False)
        self.assertFalse(self.controller.enabled)
        self.assertEqual(self.controller.state, FollowState.IDLE)
        self.assertEqual(self.controller.decision, FollowDecision.STOP)

    # 2. Pocket-tag mode activation and deactivation
    def test_pocket_tag_mode_activation_and_deactivation(self):
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.assertEqual(self.controller.mode, FollowMode.POCKET_TAG)
        self.assertFalse(self.controller.enabled)

        # Enable follow
        self.controller.set_enabled(True)
        self.assertTrue(self.controller.enabled)
        self.assertEqual(self.controller.state, FollowState.ACQUIRING)

        # Deactivate follow
        self.controller.set_enabled(False)
        self.assertFalse(self.controller.enabled)
        self.assertEqual(self.controller.state, FollowState.IDLE)
        self.assertEqual(self.controller.decision, FollowDecision.STOP)

    # 3. Switching between person and tag modes
    def test_switching_between_person_and_tag_modes(self):
        self.controller.set_mode(FollowMode.PERSON)
        self.controller.set_enabled(True)
        self.controller.select_person_target(1)
        self.assertEqual(self.person_tracker.selected_person_id, 1)

        # Switch to POCKET_TAG
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.assertEqual(self.controller.mode, FollowMode.POCKET_TAG)
        # Previous person target lock should be safely reset
        self.assertIsNone(self.person_tracker.selected_person_id)
        # Controller should safely disable previous follow and command STOP
        self.assertFalse(self.controller.enabled)
        self.assertEqual(self.controller.decision, FollowDecision.STOP)
        self.assertEqual(self.controller.last_command["direction"], "stop")

    # 4. Switching to manual mode
    def test_switching_to_manual_mode(self):
        self.controller.set_mode(FollowMode.PERSON)
        self.controller.set_enabled(True)
        self.assertTrue(self.controller.enabled)

        # Switch to MANUAL
        self.controller.set_mode(FollowMode.MANUAL)
        self.assertEqual(self.controller.mode, FollowMode.MANUAL)
        self.assertFalse(self.controller.enabled)
        self.assertEqual(self.controller.state, FollowState.IDLE)
        self.assertEqual(self.controller.decision, FollowDecision.STOP)

        # In MANUAL mode, autonomous follow cannot be enabled
        self.controller.set_enabled(True)
        self.assertFalse(self.controller.enabled)
        self.assertEqual(self.controller.decision, FollowDecision.STOP)

    # 5. Target selection and target reset
    def test_target_selection_and_target_reset(self):
        # Person target selection
        self.controller.select_person_target(2)
        self.assertEqual(self.person_tracker.selected_person_id, 2)
        self.controller.select_person_target(None)
        self.assertIsNone(self.person_tracker.selected_person_id)

        # Tag target selection
        self.controller.select_tag_target("TAG_UWB_42", source=RadioSource.UWB)
        self.assertEqual(self.tag_tracker.selected_tag_id, "TAG_UWB_42")
        self.assertEqual(self.tag_tracker.active_source, RadioSource.UWB)
        self.controller.select_tag_target(None)
        self.assertIsNone(self.tag_tracker.selected_tag_id)

    # 6. UWB simulated readings
    def test_uwb_simulated_readings(self):
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.tag_tracker.set_source(RadioSource.UWB)
        self.controller.set_enabled(True)

        # Target far (2.5m > 1.5m + 0.3m deadzone): Should MOVE_FORWARD
        self.sim_tag.simulate_uwb(tag_id="TAG_01", distance_m=2.5, direction="CENTER", connected=True)
        cmd = self.controller.evaluate()
        self.assertEqual(self.controller.decision, FollowDecision.MOVE_FORWARD)
        self.assertEqual(self.controller.state, FollowState.FOLLOWING)
        self.assertEqual(cmd["direction"], "forward")

        # Target in holding window (1.5m): Should HOLD
        self.sim_tag.simulate_uwb(tag_id="TAG_01", distance_m=1.5, direction="CENTER", connected=True)
        cmd = self.controller.evaluate()
        self.assertEqual(self.controller.decision, FollowDecision.HOLD)
        self.assertEqual(self.controller.state, FollowState.HOLDING_DISTANCE)
        self.assertEqual(cmd["direction"], "stop")

        # Target far left: Should TURN_LEFT
        self.sim_tag.simulate_uwb(tag_id="TAG_01", distance_m=2.5, direction="LEFT", connected=True)
        cmd = self.controller.evaluate()
        self.assertEqual(self.controller.decision, FollowDecision.TURN_LEFT)
        self.assertEqual(cmd["direction"], "left")

        # Target within minimum safe perimeter (0.5m < 0.8m): Should OBSTACLE_STOP
        self.sim_tag.simulate_uwb(tag_id="TAG_01", distance_m=0.5, direction="CENTER", connected=True)
        cmd = self.controller.evaluate()
        self.assertEqual(self.controller.decision, FollowDecision.STOP)
        self.assertEqual(self.controller.state, FollowState.OBSTACLE_STOP)
        self.assertEqual(cmd["direction"], "stop")

    # 7. BLE simulated readings
    def test_ble_simulated_readings(self):
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.tag_tracker.set_source(RadioSource.BLE)
        self.controller.set_enabled(True)

        # Simulate BLE reading
        reading = self.sim_tag.simulate_ble(tag_id="BLE_01", rssi=-65.0, direction="CENTER", connected=True)
        self.assertFalse(reading.is_metric, "BLE readings must have is_metric=False")
        self.assertEqual(reading.proximity_level, "NEAR")

        # BLE RSSI is not metric distance: forward movement must NEVER be generated
        cmd = self.controller.evaluate()
        self.assertNotEqual(self.controller.decision, FollowDecision.MOVE_FORWARD)
        self.assertEqual(self.controller.decision, FollowDecision.HOLD)
        self.assertEqual(cmd["direction"], "stop")

        # Steering alignment based on direction
        self.sim_tag.simulate_ble(tag_id="BLE_01", rssi=-65.0, direction="LEFT", connected=True)
        cmd = self.controller.evaluate()
        self.assertEqual(self.controller.decision, FollowDecision.TURN_LEFT)
        self.assertEqual(cmd["direction"], "left")

    # 8. Invalid and stale tag readings
    def test_invalid_and_stale_tag_readings(self):
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.tag_tracker.set_source(RadioSource.UWB)
        self.controller.set_enabled(True)

        # Stale reading (timestamp 2.0s in past, timeout is 0.8s)
        stale_time = time.time() - 2.0
        self.sim_tag.simulate_uwb(tag_id="TAG_01", distance_m=2.5, connected=True, timestamp=stale_time)
        cmd = self.controller.evaluate()
        self.assertEqual(self.controller.decision, FollowDecision.STOP)
        self.assertEqual(self.controller.state, FollowState.COMMUNICATION_ERROR)
        self.assertEqual(cmd["direction"], "stop")

        # Invalid distance (negative distance)
        reading = TagReading(
            tag_id="TAG_01",
            source=RadioSource.UWB,
            distance_m=-1.0,
            is_metric=True,
            signal_quality=50.0,
            connected=True,
            valid=True
        )
        self.tag_tracker.uwb_tracker.process_reading(reading)
        cmd = self.controller.evaluate()
        self.assertEqual(self.controller.decision, FollowDecision.STOP)
        self.assertEqual(cmd["direction"], "stop")

        # Disconnected tag
        self.sim_tag.simulate_uwb(tag_id="TAG_01", distance_m=2.5, connected=False)
        cmd = self.controller.evaluate()
        self.assertEqual(self.controller.decision, FollowDecision.STOP)
        self.assertEqual(self.controller.state, FollowState.TARGET_LOST)

    # 9. Invalid and stale person observations
    def test_invalid_and_stale_person_observations(self):
        self.controller.set_mode(FollowMode.PERSON)
        self.controller.set_enabled(True)

        # Person observation with unavailable metric distance
        p = PersonObservation(person_id=1, bbox=(200, 100, 150, 300), confidence=0.95, direction="CENTER")
        self.assertIsNone(p.to_dict()["distance_m"])
        self.person_tracker.update([p])

        # Person centered: distance uncalibrated -> MUST NOT move forward
        cmd = self.controller.evaluate()
        self.assertNotEqual(self.controller.decision, FollowDecision.MOVE_FORWARD)
        self.assertEqual(self.controller.decision, FollowDecision.HOLD)
        self.assertEqual(cmd["direction"], "stop")

        # Person on left: can steer to align
        p_left = PersonObservation(person_id=1, bbox=(50, 100, 150, 300), confidence=0.95, direction="LEFT")
        self.person_tracker.update([p_left])
        cmd = self.controller.evaluate()
        self.assertEqual(self.controller.decision, FollowDecision.TURN_LEFT)
        self.assertEqual(cmd["direction"], "left")

        # Stale person detection (2.0s ago, timeout 0.8s)
        stale_p = PersonObservation(
            person_id=1,
            bbox=(200, 100, 150, 300),
            confidence=0.95,
            timestamp=time.time() - 2.0
        )
        self.person_tracker.update([stale_p])
        cmd = self.controller.evaluate()
        self.assertEqual(self.controller.decision, FollowDecision.STOP)
        self.assertEqual(cmd["direction"], "stop")

    # 10. Target loss resulting in STOP
    def test_target_loss_resulting_in_stop(self):
        self.controller.set_mode(FollowMode.PERSON)
        self.controller.set_enabled(True)

        # Initially person visible
        p = PersonObservation(person_id=1, bbox=(200, 100, 150, 300), confidence=0.95, direction="CENTER")
        self.person_tracker.update([p])
        self.controller.evaluate()

        # Target disappears across multiple frames
        for _ in range(self.person_tracker.max_disappeared_frames + 2):
            self.person_tracker.update([])
            cmd = self.controller.evaluate()
            # Must strictly halt — NEVER blindly wander or search for target
            self.assertEqual(self.controller.decision, FollowDecision.STOP)
            self.assertEqual(cmd["direction"], "stop")
            self.assertNotEqual(self.controller.decision, "SEARCH_TARGET")

        self.assertEqual(self.controller.state, FollowState.TARGET_LOST)

    # 11. Emergency STOP overriding all modes
    def test_emergency_stop_overriding_all_modes(self):
        # Test in PERSON mode
        self.controller.set_mode(FollowMode.PERSON)
        self.controller.set_enabled(True)
        p = PersonObservation(person_id=1, bbox=(50, 100, 150, 300), confidence=0.95, direction="LEFT")
        self.person_tracker.update([p])
        self.controller.evaluate()

        stop_cmd = self.controller.emergency_stop()
        self.assertFalse(self.controller.enabled)
        self.assertEqual(self.controller.decision, FollowDecision.STOP)
        self.assertEqual(self.controller.state, FollowState.MANUAL_OVERRIDE)
        self.assertEqual(stop_cmd["direction"], "stop")

        # Obstacle stop override
        self.controller.reset_emergency_stop()
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.controller.set_enabled(True)
        self.sim_tag.simulate_uwb(tag_id="TAG_01", distance_m=3.0, direction="CENTER")
        self.controller.trigger_obstacle(True)
        cmd = self.controller.evaluate()
        self.assertEqual(self.controller.decision, FollowDecision.STOP)
        self.assertEqual(self.controller.state, FollowState.OBSTACLE_STOP)
        self.assertEqual(cmd["direction"], "stop")

    # 12. No physical hardware commands generated
    def test_no_physical_hardware_commands_being_generated(self):
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.controller.set_enabled(True)
        self.sim_tag.simulate_uwb(tag_id="TAG_01", distance_m=2.5, direction="CENTER")
        self.controller.evaluate()

        # Check last command and command history
        last = self.controller.last_command
        self.assertTrue(last["simulated"], "Command must be flagged simulated: True")
        self.assertFalse(last["sent"], "Command must be flagged sent: False")

        for cmd in self.controller.command_history:
            self.assertTrue(cmd["simulated"])
            self.assertFalse(cmd["sent"])

    # 13 & 14. API endpoints integration
    def test_api_endpoints_integration(self):
        # 1. Global status includes follow telemetry
        status_res = main.status()
        self.assertIn("follow", status_res)
        self.assertIn("mode", status_res["follow"])

        # 2. Set mode to POCKET_TAG
        mode_res = main.set_follow_mode(FollowModeCommand(mode="POCKET_TAG"))
        self.assertTrue(mode_res["ok"])
        self.assertEqual(mode_res["follow"]["mode"], "POCKET_TAG")

        # 3. Enable follow
        enable_res = main.set_follow_enable(FollowEnableCommand(enabled=True))
        self.assertTrue(enable_res["ok"])
        self.assertTrue(enable_res["follow"]["enabled"])

        # 4. Set tag radio source
        source_res = main.set_follow_tag_source(FollowTagSourceCommand(source="UWB"))
        self.assertTrue(source_res["ok"])

        # 5. Simulate UWB tag reading
        sim_res = main.simulate_tag_reading(SimTagCommand(
            tag_id="TAG_API_01",
            source="UWB",
            distance_m=2.2,
            direction="CENTER",
            connected=True
        ))
        self.assertTrue(sim_res["ok"])
        self.assertEqual(sim_res["reading"]["tag_id"], "TAG_API_01")

        # 6. Follow status evaluation
        follow_status = main.get_follow_status()
        self.assertTrue(follow_status["ok"])
        self.assertEqual(follow_status["follow"]["decision"], FollowDecision.MOVE_FORWARD)

        # 7. Person target selection endpoint
        p_res = main.set_follow_person_target(FollowPersonTargetCommand(person_id=4))
        self.assertTrue(p_res["ok"])
        self.assertEqual(main.follow_controller.person_tracker.selected_person_id, 4)

        # 8. Person status endpoint
        p_stat = main.get_follow_person_status()
        self.assertTrue(p_stat["ok"])
        self.assertIn("people", p_stat["person_tracker"])

        # 9. Tag status endpoint
        t_stat = main.get_follow_tag_status()
        self.assertTrue(t_stat["ok"])
        self.assertIn("active_source", t_stat["tag_tracker"])

        # 10. Emergency stop halts follow controller
        stop_res = main.stop()
        self.assertTrue(stop_res["ok"])
        final_follow = main.get_follow_status()
        self.assertFalse(final_follow["follow"]["enabled"])
        self.assertEqual(final_follow["follow"]["decision"], FollowDecision.STOP)


class TestPhase4BPersonFollowingIntegration(unittest.TestCase):
    """
    Focused integration tests for Phase 4B: Person Following & Hardware Integration Preparation.
    Validates end-to-end vision tracking to follow controller decisions, target locking,
    stale detection handling, emergency-stop priority, obstacle priority, and dispatch.
    """

    def setUp(self):
        self.person_tracker = PersonTracker(max_disappeared_frames=5, stale_timeout_sec=0.8)
        self.controller = FollowController(
            person_tracker=self.person_tracker,
            target_distance_m=1.5,
            stale_timeout_sec=0.8
        )
        self.controller.set_mode(FollowMode.PERSON)
        self.controller.set_enabled(True)

    def test_real_detection_to_person_tracker_feed(self):
        """Verifies feeding YOLO-format detections into person tracker establishes ID and bearing."""
        obs = PersonObservation(
            person_id=0,
            bbox=(60, 100, 100, 200),  # cx = 110 (< 213 in 640w frame -> LEFT)
            confidence=0.91,
            features=[0.05] * 128
        )
        tracked = self.person_tracker.update([obs], frame_width=640, is_detection_frame=True)
        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0].id, 1)
        self.assertEqual(tracked[0].direction, "LEFT")

    def test_left_center_right_steering_decisions(self):
        """Verifies steering decisions match the target person's bounding-box horizontal bearing."""
        # LEFT: cx = 100
        p_left = PersonObservation(person_id=1, bbox=(50, 100, 100, 200), confidence=0.90, direction="LEFT")
        self.person_tracker.update([p_left], frame_width=640)
        self.controller.select_person_target(1)
        cmd = self.controller.evaluate()
        self.assertEqual(self.controller.decision, FollowDecision.TURN_LEFT)
        self.assertEqual(cmd["direction"], "left")

        # RIGHT: cx = 550
        p_right = PersonObservation(person_id=1, bbox=(500, 100, 100, 200), confidence=0.90, direction="RIGHT")
        self.person_tracker.update([p_right], frame_width=640)
        cmd = self.controller.evaluate()
        self.assertEqual(self.controller.decision, FollowDecision.TURN_RIGHT)
        self.assertEqual(cmd["direction"], "right")

        # CENTER: cx = 320 -> HOLD (no forward movement without metric distance sensor)
        p_center = PersonObservation(person_id=1, bbox=(270, 100, 100, 200), confidence=0.90, direction="CENTER")
        self.person_tracker.update([p_center], frame_width=640)
        cmd = self.controller.evaluate()
        self.assertEqual(self.controller.decision, FollowDecision.HOLD)
        self.assertEqual(cmd["direction"], "stop")
        self.assertEqual(cmd["speed"], 0)

    def test_forward_movement_blocked_without_calibrated_distance(self):
        """Verifies that bounding-box size never produces forward movement in person mode."""
        p_huge = PersonObservation(person_id=1, bbox=(120, 20, 400, 440), confidence=0.99, direction="CENTER")
        self.person_tracker.update([p_huge], frame_width=640)
        self.controller.select_person_target(1)
        cmd = self.controller.evaluate()
        self.assertNotEqual(self.controller.decision, FollowDecision.MOVE_FORWARD)
        self.assertEqual(self.controller.decision, FollowDecision.HOLD)
        self.assertEqual(cmd["direction"], "stop")

    def test_stale_detection_handling(self):
        """Verifies that stale observations are rejected and immediately halt the robot with STOP."""
        stale_time = time.time() - 3.0
        p_stale = PersonObservation(person_id=1, bbox=(270, 100, 100, 200), confidence=0.90, timestamp=stale_time)
        self.person_tracker.update([p_stale], frame_width=640)
        self.controller.select_person_target(1)
        cmd = self.controller.evaluate()
        self.assertEqual(self.controller.decision, FollowDecision.STOP)
        self.assertEqual(cmd["direction"], "stop")
        self.assertEqual(cmd["speed"], 0)
        self.assertIn(self.controller.state, (FollowState.COMMUNICATION_ERROR, FollowState.TARGET_LOST))

    def test_target_locking_and_loss(self):
        """Verifies target locking onto active person and transition to STOP on target loss."""
        p1 = PersonObservation(person_id=1, bbox=(270, 100, 100, 200), confidence=0.92, direction="CENTER")
        self.person_tracker.update([p1], frame_width=640)
        self.controller.evaluate()
        self.assertEqual(self.controller.selected_person_id, 1, "Target ID must be locked")

        # Disappear target for 7 frames (> 5 max_disappeared)
        for _ in range(7):
            self.person_tracker.update([])

        cmd = self.controller.evaluate()
        self.assertEqual(self.controller.decision, FollowDecision.STOP)
        self.assertEqual(self.controller.state, FollowState.TARGET_LOST)

    def test_prevent_automatic_switching_on_target_loss(self):
        """
        Critical safety check: When the locked target person is lost, the controller must STOP
        and must NOT automatically switch to a different person in the frame.
        """
        # Frame 1: Person 1 (large) and Person 2 (small)
        p1 = PersonObservation(person_id=1, bbox=(200, 50, 150, 300), confidence=0.95, direction="CENTER")
        p2 = PersonObservation(person_id=2, bbox=(50, 200, 50, 100), confidence=0.88, direction="LEFT")
        self.person_tracker.update([p1, p2], frame_width=640)

        # Evaluate: locks on Person 1
        self.controller.evaluate()
        self.assertEqual(self.controller.selected_person_id, 1)

        # Frame 2: Person 1 disappears, but Person 2 is still present
        self.person_tracker.update([p2], frame_width=640)

        # Controller evaluates: must NOT switch to Person 2! Must command STOP awaiting Person 1
        cmd = self.controller.evaluate()
        self.assertEqual(self.controller.selected_person_id, 1, "Must remain locked on Person 1")
        self.assertEqual(self.controller.decision, FollowDecision.STOP)
        self.assertIn(self.controller.state, (FollowState.ACQUIRING, FollowState.TARGET_LOST))
        self.assertEqual(cmd["direction"], "stop")

    def test_emergency_stop_priority_over_person_following(self):
        """Verifies that emergency stop overrides person follow, latches, and rejects follow re-enable."""
        p1 = PersonObservation(person_id=1, bbox=(50, 100, 100, 200), confidence=0.90, direction="LEFT")
        self.person_tracker.update([p1], frame_width=640)
        self.controller.select_person_target(1)

        # Stop via controller emergency_stop()
        self.controller.emergency_stop()
        self.assertTrue(self.controller.emergency_stop_latched)
        self.assertFalse(self.controller.enabled)

        # Attempt to evaluate or re-enable while latched
        self.assertFalse(self.controller.set_enabled(True))
        cmd = self.controller.evaluate()
        self.assertEqual(self.controller.decision, FollowDecision.STOP)
        self.assertEqual(self.controller.state, FollowState.MANUAL_OVERRIDE)

        # Explicit reset restores IDLE
        self.controller.reset_emergency_stop()
        self.assertFalse(self.controller.emergency_stop_latched)
        self.assertEqual(self.controller.state, FollowState.IDLE)

    def test_obstacle_stop_priority_over_person_following(self):
        """Verifies that obstacle detection immediately overrides person following."""
        p_left = PersonObservation(person_id=1, bbox=(50, 100, 100, 200), confidence=0.90, direction="LEFT")
        self.person_tracker.update([p_left], frame_width=640)
        self.controller.select_person_target(1)

        # Trigger obstacle stop
        self.controller.trigger_obstacle(True)
        cmd = self.controller.evaluate()
        self.assertEqual(self.controller.decision, FollowDecision.STOP)
        self.assertEqual(self.controller.state, FollowState.OBSTACLE_STOP)
        self.assertEqual(cmd["direction"], "stop")

        # Clear obstacle allows steering to resume
        self.controller.trigger_obstacle(False)
        cmd2 = self.controller.evaluate()
        self.assertEqual(self.controller.decision, FollowDecision.TURN_LEFT)

    def test_command_dispatcher_integration(self):
        """Verifies that follow controller decisions dispatch to the communication layer."""
        dispatched_commands = []
        self.controller.command_dispatcher = lambda direction, speed: dispatched_commands.append((direction, speed))

        p_right = PersonObservation(person_id=1, bbox=(500, 100, 100, 200), confidence=0.90, direction="RIGHT")
        self.person_tracker.update([p_right], frame_width=640)
        self.controller.select_person_target(1)
        self.controller.evaluate()

        self.assertEqual(len(dispatched_commands), 1)
        self.assertEqual(dispatched_commands[0][0], "right")


if __name__ == "__main__":
    unittest.main()

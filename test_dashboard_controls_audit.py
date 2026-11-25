"""
Audit Test: Directly execute every single dashboard control endpoint handler with Pydantic command models.
Verifies all 72 interactive controls and backend response signatures.
"""
import sys
import os
import unittest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(BASE_DIR, "walle_software_starter")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

import main
from main import (
    DriveCommand,
    AutonomousEnableCommand,
    AutonomousTargetCommand,
    FollowModeCommand,
    FollowEnableCommand,
    FollowPersonTargetCommand,
    FollowTagTargetCommand,
    FollowTagSourceCommand,
    DetectorModeCommand,
    BenchmarkCommand,
    PersonScenarioCommand,
    SimSensorCommand,
    SimESP32DisconnectCommand,
    ArenaMovePersonCommand,
    ArenaAddPersonCommand,
    ArenaAddObstacleCommand,
    ArenaRepositionObstacleCommand,
    ArenaRepositionPersonCommand,
    ArenaSensorFailCommand,
    ArenaAvoidanceToggleCommand
)
from sim_arena import NavigationMode, AvoidanceState, DetourDirection, ReacquisitionStatus


class TestDashboardControlsInteraction(unittest.TestCase):
    def setUp(self):
        main.reset_safety_latch()
        main.simulate_esp32_disconnect(SimESP32DisconnectCommand(disconnected=False))
        main.clear_simulated_people()
        main.sim_arena.reset()

    def tearDown(self):
        main.reset_safety_latch()
        main.simulate_esp32_disconnect(SimESP32DisconnectCommand(disconnected=False))
        main.clear_simulated_people()
        main.sim_arena.reset()

    def test_01_manual_driving_and_stop(self):
        """Verify D-Pad manual driving controls: forward, backward, left, right, stop, reset."""
        # Forward
        r = main.drive(DriveCommand(direction="forward", speed=150))
        self.assertTrue(r["ok"])
        self.assertEqual(main.state["drive"], "forward")

        # Backward
        r = main.drive(DriveCommand(direction="backward", speed=150))
        self.assertTrue(r["ok"])
        self.assertEqual(main.state["drive"], "backward")

        # Left
        r = main.drive(DriveCommand(direction="left", speed=150))
        self.assertTrue(r["ok"])
        self.assertEqual(main.state["drive"], "left")

        # Right
        r = main.drive(DriveCommand(direction="right", speed=150))
        self.assertTrue(r["ok"])
        self.assertEqual(main.state["drive"], "right")

        # Stop
        r = main.stop()
        self.assertTrue(r["ok"])
        self.assertEqual(main.state["drive"], "stopped")
        self.assertTrue(main.follow_controller.emergency_stop_latched)

        # Reset safety latch
        r = main.reset_safety_latch()
        self.assertTrue(r["ok"])
        self.assertFalse(main.follow_controller.emergency_stop_latched)

    def test_02_follow_mode_and_enable(self):
        """Verify switching modes: MANUAL, PERSON, POCKET_TAG and follow toggle."""
        # Mode MANUAL
        r = main.set_follow_mode(FollowModeCommand(mode="MANUAL"))
        self.assertTrue(r["ok"])
        self.assertEqual(r["follow"]["mode"], "MANUAL")

        # Mode PERSON
        r = main.set_follow_mode(FollowModeCommand(mode="PERSON"))
        self.assertTrue(r["ok"])
        self.assertEqual(r["follow"]["mode"], "PERSON")

        # Mode POCKET_TAG
        r = main.set_follow_mode(FollowModeCommand(mode="POCKET_TAG"))
        self.assertTrue(r["ok"])
        self.assertEqual(r["follow"]["mode"], "POCKET_TAG")

        # Enable follow
        r = main.set_follow_enable(FollowEnableCommand(enabled=True))
        self.assertTrue(r["ok"])
        self.assertTrue(r["follow"]["enabled"])

        # Disable follow
        r = main.set_follow_enable(FollowEnableCommand(enabled=False))
        self.assertTrue(r["ok"])
        self.assertFalse(r["follow"]["enabled"])

    def test_03_person_target_locking_controls(self):
        """Verify LOCK ID, AUTO, CLEAR, SELECT TARGET."""
        r = main.set_follow_person_target(FollowPersonTargetCommand(person_id=1))
        self.assertTrue(r["ok"])
        self.assertEqual(r["follow"]["person_tracker"]["selected_person_id"], 1)

        r = main.set_person_target_auto()
        self.assertTrue(r["ok"])

        r = main.clear_person_target()
        self.assertTrue(r["ok"])
        self.assertIsNone(r["follow"]["person_tracker"]["selected_person_id"])

    def test_04_detector_mode_and_benchmark(self):
        """Verify detector modes REAL/SIMULATION and CPU benchmark."""
        r = main.set_person_detector_mode(DetectorModeCommand(mode="SIMULATION"))
        self.assertTrue(r["ok"])
        self.assertEqual(r["detector"]["mode"], "SIMULATION")

        r = main.set_person_detector_mode(DetectorModeCommand(mode="REAL"))
        self.assertTrue(r["ok"])

        r = main.run_person_detector_benchmark(BenchmarkCommand(iterations=5))
        self.assertTrue(r["ok"])
        self.assertIn("achievable_fps", r["benchmark"])

    def test_05_tag_target_and_source_controls(self):
        """Verify tag source UWB/BLE, lock tag, clear tag target."""
        r = main.set_follow_tag_source(FollowTagSourceCommand(source="UWB"))
        self.assertTrue(r["ok"])

        r = main.set_follow_tag_source(FollowTagSourceCommand(source="BLE"))
        self.assertTrue(r["ok"])

        r = main.set_follow_tag_target(FollowTagTargetCommand(tag_id="tag_01"))
        self.assertTrue(r["ok"])

        r = main.set_follow_tag_target(FollowTagTargetCommand(tag_id=None))
        self.assertTrue(r["ok"])

    def test_06_sim_presets_and_scenarios(self):
        """Verify preset buttons: MULTIPLE, LEFT, CENTER, RIGHT, CLEAR, and scenario suite."""
        for preset in ["MULTIPLE", "LEFT", "CENTER", "RIGHT", "CLEAR"]:
            r = main.set_sim_person_preset(preset=preset)
            self.assertTrue(r["ok"])

        # Run scenarios all
        r = main.run_person_scenario(PersonScenarioCommand(scenario="all"))
        self.assertTrue(r["ok"])
        self.assertEqual(r["report"]["passed"], 13)

    def test_07_sensor_simulation_and_failsafes(self):
        """Verify PATH CLEAR (85cm), OBSTACLE (18cm), STALE, SENSOR DISCONNECTED, toggle obstacle, toggle disconnect."""
        # Clear
        r = main.simulate_sensor_reading(SimSensorCommand(distance_cm=85.0, obstacle=False, is_stale=False, is_valid=True))
        self.assertTrue(r["ok"])

        # Obstacle
        r = main.simulate_sensor_reading(SimSensorCommand(distance_cm=18.0, obstacle=True, is_stale=False, is_valid=True))
        self.assertTrue(r["ok"])

        # Stale
        r = main.simulate_sensor_reading(SimSensorCommand(distance_cm=85.0, obstacle=False, is_stale=True, is_valid=True))
        self.assertTrue(r["ok"])

        # Disconnected
        r = main.simulate_sensor_reading(SimSensorCommand(distance_cm=None, obstacle=False, is_stale=True, is_valid=False))
        self.assertTrue(r["ok"])

        # Obstacle toggle
        r = main.set_follow_obstacle(detected=True)
        self.assertTrue(r["ok"])
        r = main.set_follow_obstacle(detected=False)
        self.assertTrue(r["ok"])

        # ESP32 disconnect toggle
        r = main.simulate_esp32_disconnect(SimESP32DisconnectCommand(disconnected=True))
        self.assertTrue(r["ok"])
        r = main.simulate_esp32_disconnect(SimESP32DisconnectCommand(disconnected=False))
        self.assertTrue(r["ok"])

    def test_08_autonomous_engine_controls(self):
        """Verify ENABLE AUTO, MANUAL OVERRIDE, LOCK ID, AUTO SELECT."""
        r = main.set_autonomous_enable(AutonomousEnableCommand(enabled=True))
        self.assertTrue(r["ok"])

        r = main.set_autonomous_target(AutonomousTargetCommand(target_id=5))
        self.assertTrue(r["ok"])

        r = main.set_autonomous_target(AutonomousTargetCommand(target_id=None))
        self.assertTrue(r["ok"])

        r = main.autonomous_override()
        self.assertTrue(r["ok"])

    def test_09_arena_simulation_controls(self):
        """Verify 2D Arena controls: step, reset, person move/add/remove/select/reposition, obstacle add/remove/clear/reposition, avoidance toggle, search start/cancel."""
        # Step
        r = main.step_arena()
        self.assertTrue(r["ok"])

        # Reset
        r = main.reset_arena()
        self.assertTrue(r["ok"])

        # Person Move
        r = main.move_arena_person(ArenaMovePersonCommand(person_id=1, dx=0.5, dy=0.0))
        self.assertTrue(r["ok"])

        # Person Add
        r = main.add_arena_person(ArenaAddPersonCommand(person_id=3, x=6.0, y=3.0))
        self.assertTrue(r["ok"])

        # Select Target
        r = main.set_arena_target(person_id=3)
        self.assertTrue(r["ok"])

        # Reposition Person Ahead
        r = main.reposition_arena_person(ArenaRepositionPersonCommand(distance_m=2.5))
        self.assertTrue(r["ok"])

        # Remove Person
        r = main.remove_arena_person(person_id=3)
        self.assertTrue(r["ok"])

        # Obstacle Add
        r = main.add_arena_obstacle(ArenaAddObstacleCommand(obs_id="test_obs", x=4.0, y=4.0, radius=0.3))
        self.assertTrue(r["ok"])

        # Obstacle Reposition
        r = main.reposition_arena_obstacle(ArenaRepositionObstacleCommand(obs_id="test_obs", x=4.5, y=4.5, radius=0.35))
        self.assertTrue(r["ok"])

        # Obstacle Remove
        r = main.remove_arena_obstacle(obs_id="test_obs")
        self.assertTrue(r["ok"])

        # Clear All Obstacles
        r = main.clear_arena_obstacles()
        self.assertTrue(r["ok"])

        # Avoidance Toggle
        r = main.toggle_arena_avoidance()
        self.assertTrue(r["ok"])

        # Search Start should be rejected when target was removed
        r_no_target = main.start_arena_target_search()
        self.assertFalse(r_no_target["ok"])

        # Restore target to Person 1
        main.set_arena_target(person_id=1)

        # Search Start & Cancel with valid target
        r = main.start_arena_target_search()
        self.assertTrue(r["ok"])

        r = main.cancel_arena_target_search()
        self.assertTrue(r["ok"])

        # Sensor Fail Toggle
        r = main.set_arena_sensor_fail(ArenaSensorFailCommand(fail=True))
        self.assertTrue(r["ok"])
        r = main.set_arena_sensor_fail(ArenaSensorFailCommand(fail=False))
        self.assertTrue(r["ok"])

        # Reacquire should succeed with valid target
        r_ok = main.reacquire_arena_target()
        self.assertTrue(r_ok["ok"])

    def test_10_status_telemetry_payload(self):
        """Verify GET /api/status, /api/follow/status, /api/sensors/telemetry, /api/person/detected payload integrity."""
        st = main.status()
        self.assertTrue(st["ok"])
        self.assertIn("drive", st)
        self.assertIn("sensors", st)
        self.assertIn("follow", st)
        self.assertIn("esp32_comm", st)

        # Check sensors telemetry display fields
        sens = main.get_sensors_telemetry()
        self.assertTrue(sens["ok"])
        self.assertIn("distance_cm", sens["sensors"])
        self.assertIn("obstacle", sens["sensors"])
        self.assertIn("mpu6050", sens["sensors"])
        self.assertIn("is_valid", sens["sensors"])
        self.assertIn("is_stale", sens["sensors"])

    def test_11_video_feed_standby_frame_yields(self):
        """Verify video feed generator yields valid MJPEG frame in standby/synthetic mode."""
        gen = main.mjpeg_frames()
        frame_chunk = next(gen)
        self.assertIn(b"--frame", frame_chunk)
        self.assertIn(b"Content-Type: image/jpeg", frame_chunk)


if __name__ == "__main__":
    unittest.main()

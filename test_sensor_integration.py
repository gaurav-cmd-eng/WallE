import os
import sys
import time
import json
import unittest

# Ensure walle_software_starter is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(BASE_DIR, "walle_software_starter")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from esp32_comm import (
    ESP32CommunicationManager, MockSerialTransport, SensorTelemetry, ESP32Command
)
from follow_controller import (
    FollowController, FollowMode, FollowState, FollowDecision
)
from tag_tracker import TagTrackerManager
from person_tracker import PersonTracker, PersonObservation
import main


class TestSensorIntegrationStage2(unittest.TestCase):
    """
    Integration tests for Stage 2 of the WALL-E Rover Sensor Integration:
    - Telemetry pipeline connecting HC-SR04 & MPU6050 to FollowController safety logic
    - Fail-safe behavior (threshold stop, stale/invalid rejection, recovery non-lurch)
    - Separation of simulated obstacle from hardware sensor obstacle
    - FastAPI endpoints (/api/status, /api/sensors/telemetry, /api/sensors/config)
    """

    def setUp(self):
        self.mock_transport = MockSerialTransport()
        self.esp32_comm = ESP32CommunicationManager(
            transport=self.mock_transport,
            simulation_mode=True,
            telemetry_timeout_sec=1.0
        )
        self.person_tracker = PersonTracker(max_disappeared_frames=5, stale_timeout_sec=0.8)
        self.tag_tracker = TagTrackerManager()
        self.controller = FollowController(
            person_tracker=self.person_tracker,
            tag_tracker=self.tag_tracker,
            target_distance_m=1.5,
            approach_threshold_m=1.8,
            hold_min_m=1.2,
            hold_max_m=1.8,
            min_safe_distance_m=0.8,
            stale_timeout_sec=0.8
        )

    # 1. Valid clear distance (> 30 cm) allows forward movement in tag mode
    def test_valid_clear_distance_allows_forward_movement(self):
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.controller.set_enabled(True)
        self.controller.tag_tracker.set_source("UWB")
        self.controller.tag_tracker.uwb_tracker.record_measurement(
            tag_id="TAG_01", distance_m=2.5, direction="CENTER", connected=True
        )
        self.controller.tag_tracker.set_external_direction("CENTER", tag_id="TAG_01", ttl_sec=1.0)

        # Feed clear ultrasonic telemetry (85 cm > 30 cm threshold)
        self.controller.update_sensor_telemetry(
            distance_cm=85.0, obstacle=False, is_valid=True, is_stale=False
        )

        clear, reason = self.controller.is_forward_path_clear()
        self.assertTrue(clear)
        self.assertIn("clear", reason.lower())

        cmd = self.controller.evaluate()
        self.assertEqual(cmd["decision"], FollowDecision.MOVE_FORWARD)
        self.assertEqual(self.controller.state, FollowState.FOLLOWING)

    # 2. Valid obstacle distance (<= 30 cm) forces OBSTACLE_STOP and STOP decision
    def test_valid_obstacle_distance_forces_stop(self):
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.controller.set_enabled(True)
        self.controller.tag_tracker.set_source("UWB")
        self.controller.tag_tracker.uwb_tracker.record_measurement(
            tag_id="TAG_01", distance_m=2.5, direction="CENTER", connected=True
        )
        self.controller.tag_tracker.set_external_direction("CENTER", tag_id="TAG_01", ttl_sec=1.0)

        # Feed obstacle distance (18.5 cm <= 30 cm threshold)
        self.controller.update_sensor_telemetry(
            distance_cm=18.5, obstacle=True, is_valid=True, is_stale=False
        )

        self.assertTrue(self.controller.sensor_obstacle)
        self.assertTrue(self.controller.obstacle_detected)

        clear, reason = self.controller.is_forward_path_clear()
        self.assertFalse(clear)

        cmd = self.controller.evaluate()
        self.assertEqual(cmd["decision"], FollowDecision.STOP)
        self.assertEqual(self.controller.state, FollowState.OBSTACLE_STOP)
        self.assertTrue(self.controller.was_in_obstacle_stop)

    # 3. Missing distance prevents autonomous forward movement when required
    def test_missing_distance_prevents_forward_movement(self):
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.controller.set_enabled(True)
        self.controller.require_ultrasonic_for_forward = True
        self.controller.tag_tracker.set_source("UWB")
        self.controller.tag_tracker.uwb_tracker.record_measurement(
            tag_id="TAG_01", distance_m=2.5, direction="CENTER", connected=True
        )
        self.controller.tag_tracker.set_external_direction("CENTER", tag_id="TAG_01", ttl_sec=1.0)

        # No ultrasonic distance fed (distance_cm is None)
        clear, reason = self.controller.is_forward_path_clear()
        self.assertFalse(clear)
        self.assertIn("missing", reason.lower())

        cmd = self.controller.evaluate()
        self.assertEqual(cmd["decision"], FollowDecision.HOLD)
        self.assertEqual(self.controller.state, FollowState.HOLDING_DISTANCE)

    # 4. Invalid distance reading is rejected and never interpreted as clear path
    def test_invalid_distance_reading_rejected(self):
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.controller.set_enabled(True)
        self.controller.tag_tracker.set_source("UWB")
        self.controller.tag_tracker.uwb_tracker.record_measurement(
            tag_id="TAG_01", distance_m=2.5, direction="CENTER", connected=True
        )
        self.controller.tag_tracker.set_external_direction("CENTER", tag_id="TAG_01", ttl_sec=1.0)

        # Feed corrupted / invalid reading (e.g. out of range or sensor timeout)
        self.controller.update_sensor_telemetry(
            distance_cm=None, obstacle=False, is_valid=False, is_stale=False
        )

        clear, reason = self.controller.is_forward_path_clear()
        self.assertFalse(clear)
        self.assertIn("invalid", reason.lower())

        cmd = self.controller.evaluate()
        self.assertEqual(cmd["decision"], FollowDecision.HOLD)

    # 5. Stale telemetry prevents autonomous forward movement
    def test_stale_telemetry_prevents_forward_movement(self):
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.controller.set_enabled(True)
        self.controller.tag_tracker.set_source("UWB")
        self.controller.tag_tracker.uwb_tracker.record_measurement(
            tag_id="TAG_01", distance_m=2.5, direction="CENTER", connected=True
        )
        self.controller.tag_tracker.set_external_direction("CENTER", tag_id="TAG_01", ttl_sec=1.0)

        # First receive fresh valid reading
        self.controller.update_sensor_telemetry(
            distance_cm=90.0, obstacle=False, is_valid=True, is_stale=False
        )
        self.assertTrue(self.controller.is_forward_path_clear()[0])

        # Then telemetry becomes stale (> 2.0s without fresh packet)
        self.controller.update_sensor_telemetry(
            distance_cm=90.0, obstacle=False, is_valid=True, is_stale=True
        )
        clear, reason = self.controller.is_forward_path_clear()
        self.assertFalse(clear)
        self.assertIn("stale", reason.lower())

        cmd = self.controller.evaluate()
        self.assertEqual(cmd["decision"], FollowDecision.HOLD)

    # 6. Sensor recovery after obstacle clearing does NOT automatically resume movement
    def test_sensor_recovery_does_not_automatically_lurch_forward(self):
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.controller.set_enabled(True)
        self.controller.tag_tracker.set_source("UWB")
        self.controller.tag_tracker.uwb_tracker.record_measurement(
            tag_id="TAG_01", distance_m=2.5, direction="CENTER", connected=True
        )
        self.controller.tag_tracker.set_external_direction("CENTER", tag_id="TAG_01", ttl_sec=1.0)

        # Trigger obstacle stop
        self.controller.update_sensor_telemetry(
            distance_cm=15.0, obstacle=True, is_valid=True, is_stale=False
        )
        cmd_stop = self.controller.evaluate()
        self.assertEqual(cmd_stop["decision"], FollowDecision.STOP)
        self.assertEqual(self.controller.state, FollowState.OBSTACLE_STOP)
        self.assertTrue(self.controller.was_in_obstacle_stop)

        # Obstacle clears with fresh valid distance
        self.controller.update_sensor_telemetry(
            distance_cm=120.0, obstacle=False, is_valid=True, is_stale=False
        )
        self.assertFalse(self.controller.obstacle_detected)

        # Failsafe check: First evaluation cycle MUST NOT lurch forward into MOVE_FORWARD!
        cmd_recovery = self.controller.evaluate()
        self.assertEqual(cmd_recovery["decision"], FollowDecision.HOLD)
        self.assertEqual(self.controller.state, FollowState.HOLDING_DISTANCE)
        self.assertIn("holding position", cmd_recovery["reason"].lower())

        # Once held and stable, subsequent evaluation with clear path can approach target
        cmd_next = self.controller.evaluate()
        self.assertEqual(cmd_next["decision"], FollowDecision.MOVE_FORWARD)

    # 7. Obstacle STOP overrides person tracking immediately
    def test_obstacle_stop_overrides_person_tracking(self):
        self.controller.set_mode(FollowMode.PERSON)
        self.controller.set_enabled(True)
        obs = PersonObservation(person_id=1, bbox=(50, 100, 100, 200), confidence=0.92, direction="LEFT")
        self.person_tracker.update([obs], frame_width=640)
        self.controller.select_person_target(1)

        # Trigger obstacle stop via sensor
        self.controller.update_sensor_telemetry(
            distance_cm=22.0, obstacle=True, is_valid=True, is_stale=False
        )
        cmd = self.controller.evaluate()
        self.assertEqual(cmd["decision"], FollowDecision.STOP)
        self.assertEqual(self.controller.state, FollowState.OBSTACLE_STOP)

    # 8. Emergency STOP latch overrides all sensor clearing and cannot resume until reset
    def test_emergency_stop_overrides_all_sensor_states(self):
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.controller.set_enabled(True)

        # Engage emergency stop
        self.controller.emergency_stop()
        self.assertTrue(self.controller.emergency_stop_latched)

        # Even with clear sensor telemetry, emergency stop holds STOP
        self.controller.update_sensor_telemetry(
            distance_cm=150.0, obstacle=False, is_valid=True, is_stale=False
        )
        cmd = self.controller.evaluate()
        self.assertEqual(cmd["decision"], FollowDecision.STOP)
        self.assertEqual(self.controller.state, FollowState.MANUAL_OVERRIDE)

        # Resetting safety latch restores IDLE, does NOT resume movement
        self.controller.reset_emergency_stop()
        self.assertFalse(self.controller.emergency_stop_latched)
        self.assertEqual(self.controller.state, FollowState.IDLE)
        self.assertFalse(self.controller.enabled)

    # 9. Simulated obstacle clearing does not clear real/mock sensor obstacle
    def test_simulated_obstacle_clear_does_not_clear_sensor_obstacle(self):
        # Hardware/telemetry detects an obstacle
        self.controller.update_sensor_telemetry(
            distance_cm=20.0, obstacle=True, is_valid=True, is_stale=False
        )
        self.assertTrue(self.controller.sensor_obstacle)
        self.assertTrue(self.controller.obstacle_detected)

        # Clearing simulated obstacle via dashboard or API
        self.controller.trigger_obstacle(False)
        self.assertFalse(self.controller.simulated_obstacle)
        # Fail-safe check: sensor_obstacle remains True, overall obstacle_detected remains True!
        self.assertTrue(self.controller.sensor_obstacle)
        self.assertTrue(self.controller.obstacle_detected)

    # 10. Interleaved telemetry and motor ACK handling via MockSerialTransport
    def test_interleaved_telemetry_and_ack_handling(self):
        # Stage incoming serial buffer with interleaved telemetry and command responses
        telemetry_dict = {
            "type": "telemetry",
            "ts": 123456,
            "distance_cm": 45.2,
            "obstacle": False,
            "imu": {"ax": 0.05, "ay": -0.12, "az": 9.81, "gx": 0.01, "gy": -0.02, "gz": 0.0},
            "status": {"us": "ok", "imu": "ok"}
        }
        self.mock_transport.queue_response(telemetry_dict)

        # Send a motor command; transport should demux telemetry and ACK
        resp = self.esp32_comm.send_command(ESP32Command.FORWARD, speed=160)
        self.assertTrue(resp["ok"])
        self.assertEqual(resp["response"]["status"], "ok")

        # Telemetry should be captured
        t = self.esp32_comm.get_telemetry()
        self.assertIsNotNone(t)
        self.assertEqual(t["distance_cm"], 45.2)
        self.assertFalse(t["obstacle"])
        self.assertEqual(t["imu"]["az"], 9.81)

    # 11. Backend endpoints expose sensor telemetry and follow safety
    def test_backend_sensor_endpoints(self):
        # Inject mock telemetry packet
        telemetry_dict = {
            "type": "telemetry",
            "ts": 999999,
            "distance_cm": 68.4,
            "obstacle": False,
            "imu": {"ax": 0.1, "ay": -0.2, "az": 9.79, "gx": 0.0, "gy": 0.01, "gz": -0.01},
            "status": {"us": "ok", "imu": "ok"}
        }
        main.esp32_comm.transport.queue_response(telemetry_dict)

        # Query /api/sensors/telemetry
        res = main.get_sensors_telemetry()
        self.assertTrue(res["ok"])
        sensors = res["sensors"]
        self.assertEqual(sensors["distance_cm"], 68.4)
        self.assertFalse(sensors["obstacle"])
        self.assertAlmostEqual(sensors["imu"]["az"], 9.79, places=2)
        self.assertTrue(sensors["is_valid"])
        self.assertFalse(sensors["is_stale"])

        # Query /api/status includes sensors
        status_res = main.status()
        self.assertIn("sensors", status_res)
        self.assertEqual(status_res["sensors"]["distance_cm"], 68.4)

    # 12. Backend /api/sensors/config endpoint configuration
    def test_backend_sensor_config_endpoint(self):
        # Configure threshold and requirement
        cmd = main.SensorConfigCommand(
            obstacle_threshold_cm=35.0,
            require_ultrasonic_for_forward=True
        )
        res = main.set_sensor_config(cmd)
        self.assertTrue(res["ok"])
        self.assertEqual(res["obstacle_threshold_cm"], 35.0)
        self.assertTrue(res["require_ultrasonic_for_forward"])
        self.assertEqual(main.follow_controller.ultrasonic_obstacle_threshold_cm, 35.0)
        self.assertTrue(main.follow_controller.require_ultrasonic_for_forward)

        # Reset back to defaults
        main.set_sensor_config(main.SensorConfigCommand(
            obstacle_threshold_cm=30.0,
            require_ultrasonic_for_forward=False
        ))


if __name__ == "__main__":
    unittest.main()

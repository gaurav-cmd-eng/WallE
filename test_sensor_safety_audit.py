import os
import sys
import time
import json
import unittest
from fastapi import HTTPException

# Ensure walle_software_starter is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(BASE_DIR, "walle_software_starter")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from esp32_comm import (
    ESP32CommunicationManager, MockSerialTransport, SensorTelemetry, ESP32Command, parse_telemetry_packet
)
from follow_controller import (
    FollowController, FollowMode, FollowState, FollowDecision
)
from tag_tracker import TagTrackerManager
from person_tracker import PersonTracker
import main


class TestSensorSafetyAudit(unittest.TestCase):
    """
    Focused safety verification tests for physical sensor integration:
    1. Manual forward blocked by obstacle guard
    2. Active motion halted if telemetry becomes stale
    3. Invalid / missing readings never clear path
    4. Dashboard simulation controls cannot override actual sensor safety
    5. Sensor recovery requires fresh valid readings & prevents auto-lurch
    6. Emergency STOP remains latched regardless of sensor updates
    7. ESP32 watchdog independently halts motion
    8. Dashboard telemetry elements verification
    """

    def setUp(self):
        # Reset main state for clean tests
        main.follow_controller.reset_emergency_stop()
        main.follow_controller.trigger_obstacle(False)
        main.follow_controller.sensor_obstacle = False
        main.state["drive"] = "stopped"
        main.state["emergency_stop_latched"] = False
        main.esp32_comm.transport = MockSerialTransport()
        main.esp32_comm.latest_telemetry = None
        main.esp32_comm.current_state = "STOPPED"

    # 1. Can manual FORWARD commands bypass the HC-SR04 safety guard? -> NO, BLOCKED
    def test_manual_forward_blocked_by_obstacle(self):
        # Trigger sensor obstacle
        main.follow_controller.update_sensor_telemetry(
            distance_cm=18.0, obstacle=True, is_valid=True, is_stale=False
        )
        self.assertTrue(main.follow_controller.obstacle_detected)

        # Attempting manual forward drive must be blocked by HTTPException
        with self.assertRaises(HTTPException) as ctx:
            main.drive(main.DriveCommand(direction="forward", speed=150))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("obstacle", ctx.exception.detail.lower())

        # Non-forward commands (e.g. left, right, stop) are permitted for steering away
        res_left = main.drive(main.DriveCommand(direction="left", speed=150))
        self.assertTrue(res_left["ok"])
        self.assertEqual(main.state["drive"], "left")

        res_stop = main.drive(main.DriveCommand(direction="stop", speed=0))
        self.assertTrue(res_stop["ok"])
        self.assertEqual(main.state["drive"], "stopped")

    # 1b. Manual drive blocked by emergency stop latch
    def test_manual_drive_blocked_by_emergency_stop_latch(self):
        main.stop()
        self.assertTrue(main.follow_controller.emergency_stop_latched)

        with self.assertRaises(HTTPException) as ctx:
            main.drive(main.DriveCommand(direction="forward", speed=150))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("emergency stop latch", ctx.exception.detail.lower())

        with self.assertRaises(HTTPException) as ctx2:
            main.drive(main.DriveCommand(direction="left", speed=150))
        self.assertEqual(ctx2.exception.status_code, 400)

        # Resetting latch allows driving again
        main.reset_safety_latch()
        self.assertFalse(main.follow_controller.emergency_stop_latched)

    # 1c. ESP32 communication manager blocks FORWARD command directly if obstacle in telemetry
    def test_esp32_comm_blocks_forward_with_obstacle(self):
        transport = MockSerialTransport()
        comm = ESP32CommunicationManager(transport=transport, simulation_mode=True)
        # Inject obstacle telemetry
        comm.latest_telemetry = SensorTelemetry(
            distance_cm=22.0,
            obstacle=True,
            imu=None,
            timestamp_ms=1000,
            received_at=time.time(),
            is_valid=True
        )

        res = comm.send_command(ESP32Command.FORWARD, speed=160)
        self.assertFalse(res["ok"])
        self.assertIn("blocked", res["error"].lower())
        self.assertEqual(len(transport.sent_lines), 0)  # Packet was NOT dispatched

    # 2. If ultrasonic telemetry becomes stale while rover is moving, actively issue STOP
    def test_stale_telemetry_halts_active_motion(self):
        # Rover was commanded forward
        main.state["drive"] = "forward"
        # Provide older telemetry that has now exceeded timeout
        old_time = time.time() - 3.5
        main.esp32_comm.latest_telemetry = SensorTelemetry(
            distance_cm=80.0,
            obstacle=False,
            imu=None,
            timestamp_ms=1000,
            received_at=old_time,
            is_valid=True
        )

        # sync_esp32_telemetry detects motion with stale telemetry and actively halts
        main.sync_esp32_telemetry()
        self.assertEqual(main.state["drive"], "stopped")
        self.assertIn("safety halt", main.state["last_command"].lower())

    # 2b. Watchdog in esp32_comm actively halts forward motion on stale telemetry
    def test_esp32_comm_watchdog_halts_on_stale_telemetry(self):
        transport = MockSerialTransport()
        comm = ESP32CommunicationManager(transport=transport, simulation_mode=True, telemetry_timeout_sec=1.0)
        comm.send_command(ESP32Command.FORWARD, speed=150)
        self.assertEqual(comm.current_state, "RUNNING")

        # Telemetry is stale (> 1.0s old)
        comm.latest_telemetry = SensorTelemetry(
            distance_cm=75.0,
            obstacle=False,
            imu=None,
            timestamp_ms=1000,
            received_at=time.time() - 2.5,
            is_valid=True
        )

        # check_watchdog detects forward motion with stale telemetry and actively sends STOP
        ok = comm.check_watchdog(now=time.time())
        self.assertFalse(ok)
        self.assertEqual(comm.current_state, "STOPPED")
        self.assertIn("stale", comm.error_message.lower())

    # 3. Can invalid or missing distance readings ever be interpreted as clear path? -> NO
    def test_invalid_or_missing_distance_never_clear_path(self):
        ctrl = FollowController()

        # Missing distance
        ctrl.update_sensor_telemetry(distance_cm=None, obstacle=False, is_valid=False, is_stale=False)
        clear, reason = ctrl.is_forward_path_clear()
        self.assertFalse(clear)

        # Stale distance
        ctrl.update_sensor_telemetry(distance_cm=100.0, obstacle=False, is_valid=True, is_stale=True)
        clear, reason = ctrl.is_forward_path_clear()
        self.assertFalse(clear)

        # Out of bounds / corrupted values rejected by parse_telemetry_packet
        t_neg = parse_telemetry_packet({"type": "telemetry", "distance_cm": -5.0})
        self.assertIsNone(t_neg.distance_cm)

        t_too_far = parse_telemetry_packet({"type": "telemetry", "distance_cm": 999.0})
        self.assertIsNone(t_too_far.distance_cm)

    # 4. Can dashboard simulation controls override actual sensor safety? -> NO
    def test_dashboard_simulation_cannot_override_sensor_safety(self):
        ctrl = FollowController()
        ctrl.set_mode(FollowMode.POCKET_TAG)
        ctrl.set_enabled(True)

        # Actual sensor detects obstacle
        ctrl.update_sensor_telemetry(distance_cm=15.0, obstacle=True, is_valid=True, is_stale=False)
        self.assertTrue(ctrl.sensor_obstacle)
        self.assertTrue(ctrl.obstacle_detected)

        # Dashboard / simulation clears simulated obstacle
        ctrl.trigger_obstacle(False)
        self.assertFalse(ctrl.simulated_obstacle)
        # Fail-safe: sensor_obstacle persists and overall obstacle_detected remains True
        self.assertTrue(ctrl.sensor_obstacle)
        self.assertTrue(ctrl.obstacle_detected)

        # Controller evaluation remains halted in OBSTACLE_STOP
        cmd = ctrl.evaluate()
        self.assertEqual(cmd["decision"], FollowDecision.STOP)
        self.assertEqual(ctrl.state, FollowState.OBSTACLE_STOP)

    # 5. Does sensor recovery require fresh valid readings and avoid auto-lurch? -> YES
    def test_sensor_recovery_requires_fresh_valid_and_avoids_lurch(self):
        ctrl = FollowController()
        ctrl.set_mode(FollowMode.POCKET_TAG)
        ctrl.set_enabled(True)
        ctrl.tag_tracker.set_source("UWB")
        ctrl.tag_tracker.uwb_tracker.record_measurement("TAG_01", distance_m=2.5, direction="CENTER", connected=True)
        ctrl.tag_tracker.set_external_direction("CENTER", tag_id="TAG_01", ttl_sec=1.0)

        # Trigger obstacle stop
        ctrl.update_sensor_telemetry(distance_cm=20.0, obstacle=True, is_valid=True, is_stale=False)
        cmd_stop = ctrl.evaluate()
        self.assertEqual(cmd_stop["decision"], FollowDecision.STOP)
        self.assertEqual(ctrl.state, FollowState.OBSTACLE_STOP)

        # Stale reading CANNOT clear obstacle
        ctrl.update_sensor_telemetry(distance_cm=150.0, obstacle=False, is_valid=True, is_stale=True)
        self.assertTrue(ctrl.obstacle_detected)

        # Fresh valid reading clears obstacle
        ctrl.update_sensor_telemetry(distance_cm=150.0, obstacle=False, is_valid=True, is_stale=False)
        self.assertFalse(ctrl.obstacle_detected)

        # First evaluation after recovery MUST NOT lurch forward into MOVE_FORWARD!
        cmd_recovery = ctrl.evaluate()
        self.assertEqual(cmd_recovery["decision"], FollowDecision.HOLD)
        self.assertEqual(ctrl.state, FollowState.HOLDING_DISTANCE)

        # Subsequent stable cycle allows forward approach
        cmd_next = ctrl.evaluate()
        self.assertEqual(cmd_next["decision"], FollowDecision.MOVE_FORWARD)

    # 6. Does emergency STOP remain latched regardless of sensor updates? -> YES
    def test_emergency_stop_remains_latched_regardless_of_sensors(self):
        ctrl = FollowController()
        ctrl.set_mode(FollowMode.POCKET_TAG)
        ctrl.set_enabled(True)

        # Latch e-stop
        ctrl.emergency_stop()
        self.assertTrue(ctrl.emergency_stop_latched)

        # Update sensor with clear telemetry
        ctrl.update_sensor_telemetry(distance_cm=200.0, obstacle=False, is_valid=True, is_stale=False)
        ctrl.trigger_obstacle(False)

        # Even with clear telemetry, evaluate returns STOP
        cmd = ctrl.evaluate()
        self.assertEqual(cmd["decision"], FollowDecision.STOP)
        self.assertEqual(ctrl.state, FollowState.MANUAL_OVERRIDE)

        # Re-enabling follow is strictly rejected
        ctrl.set_enabled(True)
        self.assertFalse(ctrl.enabled)
        self.assertEqual(ctrl.decision, FollowDecision.STOP)

    # 7. Dashboard HTML contains all required sensor telemetry elements
    def test_dashboard_html_contains_sensor_elements(self):
        html_path = os.path.join(PKG_DIR, "static", "index.html")
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Verify required display IDs
        self.assertIn('id="sensor-distance-cm"', content)
        self.assertIn('id="sensor-obstacle-state"', content)
        self.assertIn('id="sensor-accel"', content)
        self.assertIn('id="sensor-gyro"', content)
        self.assertIn('id="sensor-freshness"', content)
        self.assertIn('id="sensor-error"', content)
        self.assertIn('id="sensor-freshness-badge"', content)
        self.assertIn('id="sensor-obstacle-badge"', content)

        # Verify JS renders s.sensors
        self.assertIn('s.sensors', content)
        self.assertIn('sens.distance_cm', content)
        self.assertIn('sens.imu', content)


if __name__ == "__main__":
    unittest.main()

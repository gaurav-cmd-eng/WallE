import os
import sys
import time
import json
import unittest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(BASE_DIR, "walle_software_starter")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from esp32_comm import (
    ESP32CommunicationManager, ESP32Command, ConnectionStatus, MockSerialTransport
)


class TestESP32CommunicationLayer(unittest.TestCase):
    """
    Focused unit tests for Phase 4B ESP32 Communication Layer.
    Validates JSON command protocol, serialization, response parsing,
    STOP priority, watchdog timeout, disconnect handling, and mock transport.
    """

    def setUp(self):
        self.mock_transport = MockSerialTransport(auto_ack=True)
        self.comm = ESP32CommunicationManager(
            transport=self.mock_transport,
            watchdog_timeout_sec=0.5,
            default_speed=150,
            simulation_mode=True
        )

    def test_mock_transport_operations(self):
        """Verifies mock transport connection, line writing, and auto-ack."""
        self.assertTrue(self.mock_transport.is_connected())
        write_ok = self.mock_transport.write_line('{"cmd":"FORWARD","speed":150,"seq":1}\n')
        self.assertTrue(write_ok)
        self.assertEqual(len(self.mock_transport.sent_lines), 1)

        resp_line = self.mock_transport.read_line()
        self.assertIsNotNone(resp_line)
        resp = json.loads(resp_line.strip())
        self.assertEqual(resp["status"], "ok")
        self.assertEqual(resp["cmd"], "FORWARD")
        self.assertEqual(resp["seq"], 1)

    def test_json_command_serialization(self):
        """Verifies that movement commands serialize to valid newline-delimited JSON."""
        # FORWARD
        res_fwd = self.comm.send_command("FORWARD", speed=180)
        self.assertTrue(res_fwd["ok"])
        last_sent = json.loads(self.mock_transport.sent_lines[-1].strip())
        self.assertEqual(last_sent["cmd"], "FORWARD")
        self.assertEqual(last_sent["speed"], 180)
        self.assertEqual(last_sent["seq"], 1)
        self.assertIn("ts", last_sent)

        # REVERSE (and BACKWARD alias)
        res_rev = self.comm.send_command("BACKWARD", speed=120)
        self.assertTrue(res_rev["ok"])
        last_sent = json.loads(self.mock_transport.sent_lines[-1].strip())
        self.assertEqual(last_sent["cmd"], "REVERSE")
        self.assertEqual(last_sent["speed"], 120)

        # LEFT
        res_left = self.comm.send_command("LEFT", speed=100)
        self.assertTrue(res_left["ok"])
        last_sent = json.loads(self.mock_transport.sent_lines[-1].strip())
        self.assertEqual(last_sent["cmd"], "LEFT")
        self.assertEqual(last_sent["speed"], 100)

        # RIGHT
        res_right = self.comm.send_command("RIGHT", speed=100)
        self.assertTrue(res_right["ok"])
        last_sent = json.loads(self.mock_transport.sent_lines[-1].strip())
        self.assertEqual(last_sent["cmd"], "RIGHT")

        # SET_SPEED
        res_spd = self.comm.send_command("SET_SPEED", speed=200)
        self.assertTrue(res_spd["ok"])
        last_sent = json.loads(self.mock_transport.sent_lines[-1].strip())
        self.assertEqual(last_sent["cmd"], "SET_SPEED")
        self.assertEqual(last_sent["speed"], 200)

    def test_stop_command_priority_forces_speed_zero(self):
        """Verifies that STOP command always forces speed to 0 regardless of arguments."""
        # Attempt to pass non-zero speed to STOP
        res = self.comm.send_command("STOP", speed=255)
        self.assertTrue(res["ok"])
        last_sent = json.loads(self.mock_transport.sent_lines[-1].strip())
        self.assertEqual(last_sent["cmd"], "STOP")
        self.assertEqual(last_sent["speed"], 0, "STOP command must strictly enforce speed=0")
        self.assertEqual(self.comm.current_state, "STOPPED")

    def test_invalid_command_rejected_safely(self):
        """Verifies that invalid command strings return safe error without crashing."""
        res = self.comm.send_command("INVALID_MOVE", speed=100)
        self.assertFalse(res["ok"])
        self.assertFalse(res["sent"])
        self.assertIn("Invalid ESP32 command", res["error"])
        self.assertEqual(res["command"]["cmd"], "STOP")

    def test_communication_timeout_transitions_to_stop(self):
        """Verifies that response timeouts transition controller state to STOP."""
        self.mock_transport.simulate_timeout = True

        res = self.comm.send_command("FORWARD", speed=150)
        self.assertFalse(res["ok"])
        self.assertTrue(res.get("timeout", False))
        self.assertEqual(self.comm.connection_status, ConnectionStatus.ERROR)
        self.assertEqual(self.comm.current_state, "STOPPED")
        self.assertEqual(self.comm.last_command["cmd"], "STOP")
        self.assertEqual(self.comm.last_command["speed"], 0)

    def test_disconnect_handling(self):
        """Verifies that disconnected transport rejects motion commands safely."""
        self.comm.disconnect()
        self.assertEqual(self.comm.connection_status, ConnectionStatus.DISCONNECTED)
        self.assertFalse(self.comm.is_connected())

        # Attempt movement command while disconnected
        res = self.comm.send_command("FORWARD", speed=150)
        self.assertFalse(res["ok"])
        self.assertFalse(res["sent"])
        self.assertIn("disconnected", res["error"].lower())
        self.assertEqual(self.comm.current_state, "STOPPED")

    def test_watchdog_timeout_forces_stop_during_motion(self):
        """Verifies that watchdog triggers STOP if no communication occurs during motion."""
        # Set to running state
        self.comm.send_command("FORWARD", speed=150)
        self.assertEqual(self.comm.current_state, "RUNNING")

        # Simulate 2 seconds passing without communication (watchdog is 0.5s)
        now_future = time.time() + 2.0
        healthy = self.comm.check_watchdog(now=now_future)
        self.assertFalse(healthy)
        self.assertEqual(self.comm.current_state, "STOPPED")
        self.assertEqual(self.comm.connection_status, ConnectionStatus.ERROR)
        self.assertIn("Watchdog timeout", self.comm.error_message)

    def test_custom_response_parsing(self):
        """Verifies that custom ESP32 responses (e.g. error packets) are parsed accurately."""
        # Inject custom error packet
        self.mock_transport.auto_ack = False
        self.mock_transport.queue_response({"status": "error", "code": 500, "msg": "Motor stall"})

        res = self.comm.send_command("FORWARD", speed=150)
        self.assertFalse(res["ok"])
        self.assertEqual(res["response"]["status"], "error")
        self.assertEqual(res["response"]["msg"], "Motor stall")

    def test_status_telemetry(self):
        """Verifies get_status reporting."""
        status = self.comm.get_status()
        self.assertIn("status", status)
        self.assertIn("connected", status)
        self.assertIn("simulation_mode", status)
        self.assertIn("last_command", status)
        self.assertIn("watchdog_timeout_sec", status)
        self.assertTrue(status["simulation_mode"])
        self.assertIn("telemetry", status)
        self.assertIn("telemetry_stale", status)
        self.assertTrue(status["telemetry_stale"])  # Initially stale before telemetry arrives

    def test_valid_sensor_telemetry_parsing(self):
        """Verifies complete parsing of valid HC-SR04 and MPU6050 telemetry."""
        packet = {
            "type": "telemetry",
            "distance_cm": 45.2,
            "obstacle": False,
            "imu": {
                "ax": 0.015,
                "ay": -0.032,
                "az": 0.981,
                "gx": 0.12,
                "gy": -0.25,
                "gz": 0.04
            },
            "timestamp_ms": 12345
        }
        self.mock_transport.queue_response(packet)
        telem = self.comm.read_telemetry()

        self.assertIsNotNone(telem)
        self.assertTrue(telem.is_valid)
        self.assertEqual(telem.distance_cm, 45.2)
        self.assertFalse(telem.obstacle)
        self.assertTrue(telem.has_imu)
        self.assertAlmostEqual(telem.imu["az"], 0.981, places=3)
        self.assertAlmostEqual(telem.imu["gx"], 0.12, places=2)
        self.assertEqual(telem.timestamp_ms, 12345)
        self.assertFalse(self.comm.is_telemetry_stale())

    def test_missing_and_null_sensor_fields(self):
        """Verifies graceful handling when sensors fail to initialize or are out of range."""
        # HC-SR04 out of range and MPU6050 init failure
        packet = {
            "type": "telemetry",
            "distance_cm": None,
            "obstacle": False,
            "imu": None,
            "timestamp_ms": 20000
        }
        self.mock_transport.queue_response(packet)
        telem = self.comm.read_telemetry()

        self.assertIsNotNone(telem)
        self.assertTrue(telem.is_valid)
        self.assertIsNone(telem.distance_cm)
        self.assertFalse(telem.obstacle)
        self.assertFalse(telem.has_imu)
        self.assertEqual(telem.imu, {})
        self.assertEqual(telem.timestamp_ms, 20000)

    def test_obstacle_threshold_calculation(self):
        """Verifies that distance <= 30.0 cm automatically triggers obstacle flag."""
        # 15.5 cm should be obstacle
        packet = {
            "type": "telemetry",
            "distance_cm": 15.5,
            "obstacle": False,  # Even if raw flag was false, calculated obstacle overrides
            "imu": None,
            "timestamp_ms": 30000
        }
        self.mock_transport.queue_response(packet)
        telem = self.comm.read_telemetry()
        self.assertTrue(telem.obstacle)
        self.assertTrue(self.comm.get_status()["obstacle_detected"])

    def test_invalid_and_out_of_bounds_sensor_data(self):
        """Verifies that non-physical or corrupted values are filtered safely."""
        # Negative distance and invalid string IMU
        packet = {
            "type": "telemetry",
            "distance_cm": -50.0,
            "obstacle": False,
            "imu": {"ax": "corrupted_string", "ay": 0.0},
            "timestamp_ms": "not_an_int"
        }
        self.mock_transport.queue_response(packet)
        telem = self.comm.read_telemetry()

        self.assertIsNotNone(telem)
        self.assertTrue(telem.is_valid)
        self.assertIsNone(telem.distance_cm)  # Filtered out because negative
        self.assertIn("ay", telem.imu)
        self.assertNotIn("ax", telem.imu)    # Corrupted string dropped safely
        self.assertIsNone(telem.timestamp_ms)

    def test_malformed_json_packet_handling(self):
        """Verifies that malformed JSON or junk lines on serial do not crash the parser."""
        self.mock_transport.queue_response("MALFORMED JSON {{{[[[")
        telem = self.comm.read_telemetry()
        self.assertIn("Malformed JSON", self.comm.error_message)

    def test_stale_telemetry_expiration(self):
        """Verifies that telemetry is flagged as stale after configured timeout."""
        packet = {
            "type": "telemetry",
            "distance_cm": 25.0,
            "obstacle": True,
            "timestamp_ms": 1000
        }
        self.mock_transport.queue_response(packet)
        self.comm.read_telemetry()
        self.assertFalse(self.comm.is_telemetry_stale())
        self.assertTrue(self.comm.get_status()["obstacle_detected"])

        # Simulate time jump past 2.0s stale timeout
        future_time = time.time() + 3.0
        self.assertTrue(self.comm.is_telemetry_stale(now=future_time))
        # When stale, obstacle flag is inhibited
        self.assertFalse(self.comm.latest_telemetry.is_stale(now=time.time()))
        self.assertTrue(self.comm.latest_telemetry.is_stale(now=future_time))

    def test_multiplexed_telemetry_and_command_ack(self):
        """
        Critical test: Verifies that telemetry packets arriving while waiting for
        a command acknowledgement are ingested without blocking or breaking command execution.
        """
        # Turn off auto-ack to manually interleave responses
        self.mock_transport.auto_ack = False

        # Queue telemetry first, then command acknowledgement
        self.mock_transport.queue_telemetry(distance_cm=88.0, obstacle=False)
        self.mock_transport.queue_response({
            "status": "ok",
            "cmd": "FORWARD",
            "speed": 150,
            "seq": 1,
            "state": "RUNNING"
        })

        res = self.comm.send_command("FORWARD", speed=150)
        self.assertTrue(res["ok"])
        self.assertEqual(res["command"]["cmd"], "FORWARD")

        # Confirm that the interleaved telemetry was captured and parsed
        self.assertIsNotNone(self.comm.latest_telemetry)
        self.assertEqual(self.comm.latest_telemetry.distance_cm, 88.0)
        self.assertEqual(self.comm.get_status()["distance_cm"], 88.0)

    def test_serial_transport_configuration(self):
        """Verifies physical SerialTransport configuration and safe disconnect behavior."""
        from esp32_comm import SerialTransport
        transport = SerialTransport(port="/dev/ttyUSB0", baudrate=115200, timeout=0.5)
        self.assertEqual(transport.port, "/dev/ttyUSB0")
        self.assertEqual(transport.baudrate, 115200)
        self.assertFalse(transport.is_connected())

        # Attempting connect to non-existent port returns False safely without exception
        ok = transport.connect()
        self.assertFalse(ok)
        self.assertFalse(transport.is_connected())

        # Disconnecting cleanly handles unopened state
        transport.disconnect()
        self.assertFalse(transport.is_connected())


if __name__ == "__main__":
    unittest.main()

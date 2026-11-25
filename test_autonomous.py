import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "walle_software_starter")
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import main
from autonomous import (
    AutonomousDecisionEngine,
    AutonomousCommandInterface,
    Decision,
    AutonomousState
)


class TestAutonomousDecisionEngine(unittest.TestCase):
    def setUp(self):
        # Fresh engine instance with explicit test parameters
        self.engine = AutonomousDecisionEngine(
            frame_width=640,
            center_deadzone_ratio=0.20,  # Center is 256..384
            hysteresis_px=15.0,
            proximity_area=12000,
            stale_timeout_sec=0.8,
            search_policy="STOP",
            drive_speed=150,
            turn_speed=120
        )
        self.engine.set_enabled(True)

    def test_target_on_left_generates_turn_left(self):
        """1. Target on the left generates TURN_LEFT."""
        tracked = [{"id": 1, "color": "RED", "x": 120, "y": 240, "area": 3000, "direction": "LEFT"}]
        status = self.engine.evaluate(tracked, frame_width=640)
        self.assertEqual(status["decision"], Decision.TURN_LEFT)
        self.assertEqual(status["command"]["direction"], "left")
        self.assertEqual(status["command"]["speed"], 120)

    def test_target_on_right_generates_turn_right(self):
        """2. Target on the right generates TURN_RIGHT."""
        tracked = [{"id": 1, "color": "RED", "x": 520, "y": 240, "area": 3000, "direction": "RIGHT"}]
        status = self.engine.evaluate(tracked, frame_width=640)
        self.assertEqual(status["decision"], Decision.TURN_RIGHT)
        self.assertEqual(status["command"]["direction"], "right")
        self.assertEqual(status["command"]["speed"], 120)

    def test_target_centred_and_far_away_generates_move_forward(self):
        """3. Target centred and far away generates MOVE_FORWARD."""
        tracked = [{"id": 1, "color": "RED", "x": 320, "y": 240, "area": 3000, "direction": "CENTER"}]
        status = self.engine.evaluate(tracked, frame_width=640)
        self.assertEqual(status["decision"], Decision.MOVE_FORWARD)
        self.assertEqual(status["command"]["direction"], "forward")
        self.assertEqual(status["command"]["speed"], 150)

    def test_target_centred_and_sufficiently_close_generates_stop(self):
        """4. Target centred and sufficiently close (area >= proximity_area) generates STOP."""
        # proximity_area is 12000; area 14000 indicates target reached
        tracked = [{"id": 1, "color": "RED", "x": 320, "y": 240, "area": 14000, "direction": "CENTER"}]
        status = self.engine.evaluate(tracked, frame_width=640)
        self.assertEqual(status["decision"], Decision.STOP)
        self.assertEqual(status["state"], AutonomousState.TARGET_REACHED)
        self.assertEqual(status["command"]["direction"], "stop")
        self.assertEqual(status["command"]["speed"], 0)

    def test_missing_target_generates_search_target_or_stop_by_policy(self):
        """5. Missing target generates SEARCH_TARGET or STOP according to configured safety policy."""
        # Case A: Safety policy "STOP"
        engine_stop = AutonomousDecisionEngine(search_policy="STOP", stale_timeout_sec=0.5)
        engine_stop.set_enabled(True)
        res_stop = engine_stop.evaluate([], current_time=100.0)
        self.assertEqual(res_stop["decision"], Decision.STOP)

        # Case B: Safety policy "SEARCH_TARGET"
        engine_search = AutonomousDecisionEngine(search_policy="SEARCH_TARGET", stale_timeout_sec=0.5)
        engine_search.set_enabled(True)
        res_search = engine_search.evaluate([], current_time=100.0)
        self.assertEqual(res_search["decision"], Decision.SEARCH_TARGET)
        self.assertEqual(res_search["command"]["direction"], "left")

    def test_invalid_or_stale_tracking_data_cannot_generate_movement(self):
        """6. Invalid or stale tracking data cannot generate movement."""
        # Target was seen at t=10.0
        self.engine.evaluate([{"id": 1, "color": "RED", "x": 320, "y": 240, "area": 3000}], current_time=10.0)

        # At t=11.5, elapsed time is 1.5s (> stale_timeout 0.8s) with no target
        stale_status = self.engine.evaluate([], current_time=11.5)
        self.assertEqual(stale_status["decision"], Decision.STOP)
        self.assertEqual(stale_status["state"], AutonomousState.STALE_DATA)
        self.assertEqual(stale_status["command"]["direction"], "stop")
        self.assertEqual(stale_status["command"]["speed"], 0)

    def test_manual_override_always_generates_stop(self):
        """7. Manual override immediately disables autonomous mode and commands STOP."""
        # Active tracking
        self.engine.evaluate([{"id": 1, "color": "RED", "x": 320, "y": 240, "area": 3000}])
        self.assertTrue(self.engine.enabled)

        # Trigger manual override
        self.engine.manual_override()
        self.assertFalse(self.engine.enabled)
        self.assertEqual(self.engine.current_decision, Decision.STOP)
        self.assertEqual(self.engine.current_state, AutonomousState.DISABLED)
        self.assertEqual(self.engine.command_interface.last_command["direction"], "stop")

    def test_autonomous_mode_disabled_never_generates_movement(self):
        """8. Autonomous mode disabled never generates movement commands."""
        self.engine.set_enabled(False)
        tracked = [{"id": 1, "color": "RED", "x": 320, "y": 240, "area": 3000}]
        status = self.engine.evaluate(tracked)
        self.assertEqual(status["decision"], Decision.STOP)
        self.assertEqual(status["command"]["direction"], "stop")
        self.assertEqual(status["command"]["speed"], 0)

    def test_target_selection_by_persistent_id(self):
        """9. Target selection by persistent ID works."""
        # Two tracked objects: Object 1 (RED at x=100) and Object 2 (BLUE at x=550)
        tracked = [
            {"id": 1, "color": "RED", "x": 100, "y": 240, "area": 2000},
            {"id": 2, "color": "BLUE", "x": 550, "y": 240, "area": 8000}
        ]

        # Automatic mode selects largest object (ID 2 at x=550 -> TURN_RIGHT)
        self.engine.set_target_id(None)
        res_auto = self.engine.evaluate(tracked)
        self.assertEqual(res_auto["target_id"], None)
        self.assertEqual(res_auto["target"]["id"], 2)
        self.assertEqual(res_auto["decision"], Decision.TURN_RIGHT)

        # Manual selection locks ID 1 (x=100 -> TURN_LEFT)
        self.engine.set_target_id(1)
        res_locked = self.engine.evaluate(tracked)
        self.assertEqual(res_locked["target_id"], 1)
        self.assertEqual(res_locked["target"]["id"], 1)
        self.assertEqual(res_locked["decision"], Decision.TURN_LEFT)

    def test_rapid_position_changes_do_not_cause_command_oscillation(self):
        """10. Rapid position changes near boundaries do not cause command oscillation (hysteresis)."""
        # Deadzone: 256 to 384. Hysteresis: 15px.
        # Step 1: Start on the left at x=240 (< 256) -> TURN_LEFT
        s1 = self.engine.evaluate([{"id": 1, "x": 240, "y": 240, "area": 3000}])
        self.assertEqual(s1["decision"], Decision.TURN_LEFT)

        # Step 2: Target jitters to x=260 (nominally inside 256..384, but < 256 + 15 = 271)
        # Without hysteresis, this would jitter to MOVE_FORWARD.
        # With hysteresis, it must stay in TURN_LEFT!
        s2 = self.engine.evaluate([{"id": 1, "x": 260, "y": 240, "area": 3000}])
        self.assertEqual(s2["decision"], Decision.TURN_LEFT, "Must remain TURN_LEFT inside hysteresis band")

        # Step 3: Target moves deeper into center at x=280 (> 271) -> cleanly transitions to MOVE_FORWARD
        s3 = self.engine.evaluate([{"id": 1, "x": 280, "y": 240, "area": 3000}])
        self.assertEqual(s3["decision"], Decision.MOVE_FORWARD)

    def test_simulation_command_interface_works_without_esp32(self):
        """11. Simulation command interface works without an ESP32."""
        ci = AutonomousCommandInterface(drive_speed=150, turn_speed=120)
        res = ci.dispatch(Decision.MOVE_FORWARD, esp32_url="")

        self.assertTrue(res["simulated"])
        self.assertEqual(res["mode"], "simulation")
        self.assertEqual(res["command"]["direction"], "forward")
        self.assertEqual(res["command"]["speed"], 150)
        self.assertGreater(len(ci.command_history), 0)
        self.assertEqual(ci.command_history[-1]["direction"], "forward")

    def test_api_endpoints_integration(self):
        """12. Test FastAPI endpoints for autonomous control."""
        # Enable autonomous mode via API
        enable_res = main.set_autonomous_enable(main.AutonomousEnableCommand(enabled=True))
        self.assertTrue(enable_res["ok"])
        self.assertTrue(enable_res["autonomous"]["enabled"])

        # Set target ID via API
        target_res = main.set_autonomous_target(main.AutonomousTargetCommand(target_id=7))
        self.assertTrue(target_res["ok"])
        self.assertEqual(target_res["autonomous"]["target_id"], 7)

        # Get decision via API
        dec_res = main.get_autonomous_decision()
        self.assertTrue(dec_res["ok"])
        self.assertIn("decision", dec_res["autonomous"])

        # Manual override via API
        override_res = main.autonomous_override()
        self.assertTrue(override_res["ok"])
        self.assertFalse(override_res["autonomous"]["enabled"])
        self.assertEqual(override_res["autonomous"]["decision"], Decision.STOP)

        # Get command history via API
        cmd_res = main.get_autonomous_commands()
        self.assertTrue(cmd_res["ok"])
        self.assertTrue(cmd_res["simulation_mode"])
        self.assertIsInstance(cmd_res["history"], list)

        # Global status includes autonomous dictionary
        status_res = main.status()
        self.assertIn("autonomous", status_res)
        self.assertFalse(status_res["autonomous"]["enabled"])


if __name__ == "__main__":
    unittest.main()

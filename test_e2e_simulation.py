"""WALL-E Stage 5: End-to-End Person Following & Hardware-Simulated Demonstration Test Suite.

Verifies the complete integration across:
- Vision detections (multiple people)
- PersonTracker (multi-target management, target locking, anti-switching)
- FollowController (steering decisions: LEFT -> TURN_LEFT, RIGHT -> TURN_RIGHT, CENTER -> HOLD)
- Sensor Safety Guards (obstacle stops, stale telemetry forwards guard, recovery non-lurch)
- Emergency STOP (latching, reset to IDLE, no auto-resume)
- ESP32 Mock Serial Transport (command dispatch, acknowledges, disconnect failsafe)
- Dashboard Simulation Endpoints & Route Handlers (/api/follow/sim/*, /api/sensors/sim, /api/esp32/simulate_disconnect)
"""

import os
import sys
import unittest
import time
from typing import Dict, Any

# Ensure walle_software_starter is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(BASE_DIR, "walle_software_starter")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from person_tracker import PersonTracker, PersonObservation, ReacquisitionStatus
from follow_controller import FollowController, FollowMode, FollowState, FollowDecision
from esp32_comm import ESP32CommunicationManager, MockSerialTransport, SensorTelemetry
from person_simulation import DeterministicPersonSimulator
import main


class TestEndToEndSimulation(unittest.TestCase):
    """End-to-End Software Demonstration & Safety Invariant Validation."""

    def setUp(self):
        self.sim = DeterministicPersonSimulator()

    def tearDown(self):
        # Reset global system states cleanly
        main.reset_safety_latch()
        main.simulate_esp32_disconnect(main.SimESP32DisconnectCommand(disconnected=False))
        main.clear_simulated_people()
        # Restore clear sensor telemetry for subsequent test suites
        main.follow_controller.update_sensor_telemetry(distance_cm=85.0, obstacle=False, is_valid=True, is_stale=False)
        main.esp32_comm.latest_telemetry = None

    def test_e2e_all_13_deterministic_scenarios(self):
        """Validates all 13 core end-to-end person simulation scenarios."""
        report = self.sim.run_all_scenarios()
        self.assertEqual(report["total_scenarios"], 13)
        self.assertEqual(report["passed"], 13)
        self.assertEqual(report["failed"], 0)
        self.assertTrue(report["all_passed"])

    def test_scenario_1_multi_person_detection(self):
        res = self.sim.scenario_1_multiple_people_detected()
        self.assertEqual(res["status"], "PASS")
        self.assertEqual(res["people_count"], 3)

    def test_scenario_2_target_selection_and_lock(self):
        res = self.sim.scenario_2_target_selection_and_lock()
        self.assertEqual(res["status"], "PASS")
        self.assertEqual(res["selected_person_id"], 2)

    def test_scenario_3_and_4_direction_steering(self):
        left_res = self.sim.scenario_3_steering_tracking_left()
        self.assertEqual(left_res["status"], "PASS")
        self.assertEqual(left_res["decision"], FollowDecision.TURN_LEFT)
        self.assertEqual(left_res["dispatched_direction"], "left")

        right_res = self.sim.scenario_4_steering_tracking_right()
        self.assertEqual(right_res["status"], "PASS")
        self.assertEqual(right_res["decision"], FollowDecision.TURN_RIGHT)
        self.assertEqual(right_res["dispatched_direction"], "right")

        center_res = self.sim.scenario_5_centering_holds_position()
        self.assertEqual(center_res["status"], "PASS")
        self.assertEqual(center_res["decision"], FollowDecision.HOLD)

    def test_scenario_5_and_6_target_loss_and_anti_switching(self):
        loss_res = self.sim.scenario_6_missing_or_stale_target_stops()
        self.assertEqual(loss_res["status"], "PASS")
        self.assertEqual(loss_res["decision"], FollowDecision.STOP)
        self.assertEqual(loss_res["state"], FollowState.TARGET_LOST)

        anti_switch_res = self.sim.scenario_7_lock_prevents_switching()
        self.assertEqual(anti_switch_res["status"], "PASS")
        self.assertEqual(anti_switch_res["decision"], FollowDecision.STOP)
        self.assertEqual(anti_switch_res["locked_target_id"], 2)

    def test_scenario_8_obstacle_stop_priority(self):
        res = self.sim.scenario_8_obstacle_detection_overrides_following()
        self.assertEqual(res["status"], "PASS")
        self.assertEqual(res["decision"], FollowDecision.STOP)
        self.assertEqual(res["state"], FollowState.OBSTACLE_STOP)

    def test_scenario_9_and_10_sensor_freshness_and_recovery(self):
        stale_res = self.sim.scenario_9_stale_or_missing_ultrasonic_prevents_forward()
        self.assertEqual(stale_res["status"], "PASS")
        self.assertEqual(stale_res["decision"], FollowDecision.HOLD)

        recovery_res = self.sim.scenario_10_sensor_recovery_avoids_lurch()
        self.assertEqual(recovery_res["status"], "PASS")
        self.assertIn(recovery_res["recovery_decision"], (FollowDecision.HOLD, FollowDecision.TURN_LEFT))

    def test_scenario_11_and_12_emergency_stop_lifecycle(self):
        estop_res = self.sim.scenario_11_emergency_stop_latches()
        self.assertEqual(estop_res["status"], "PASS")
        self.assertTrue(estop_res["latched"])
        self.assertEqual(estop_res["state"], FollowState.MANUAL_OVERRIDE)

        reset_res = self.sim.scenario_12_reset_emergency_stop_requires_reacquisition()
        self.assertEqual(reset_res["status"], "PASS")
        self.assertFalse(reset_res["latched"])
        self.assertFalse(reset_res["enabled"])
        self.assertEqual(reset_res["state"], FollowState.IDLE)

    def test_scenario_13_esp32_disconnect_failsafe(self):
        res = self.sim.scenario_13_esp32_disconnect_failsafe()
        self.assertEqual(res["status"], "PASS")
        self.assertFalse(res["connected"])

    def test_backend_simulation_endpoints(self):
        """Verifies all Stage 5 FastAPI simulation handlers directly."""
        # 1. Preset injection MULTIPLE
        res = main.set_sim_person_preset(preset="MULTIPLE")
        self.assertTrue(res["ok"])
        self.assertEqual(len(res["tracked"]), 3)

        # Check main.status() contains the 3 simulated persons
        st = main.status()
        self.assertEqual(len(st["follow"]["person_tracker"]["people"]), 3)

        # 2. Preset LEFT
        res = main.set_sim_person_preset(preset="LEFT")
        self.assertTrue(res["ok"])
        self.assertEqual(len(res["tracked"]), 1)

        # 3. Clear people
        res = main.clear_simulated_people()
        self.assertTrue(res["ok"])
        self.assertTrue(res["cleared"])

        # 4. Sensor simulation handler
        res = main.simulate_sensor_reading(main.SimSensorCommand(
            distance_cm=22.5,
            obstacle=True,
            is_stale=False,
            is_valid=True
        ))
        self.assertTrue(res["ok"])
        st = main.status()
        self.assertEqual(st["sensors"]["distance_cm"], 22.5)
        self.assertTrue(st["sensors"]["obstacle"])

        # 5. ESP32 simulate disconnect handler
        res = main.simulate_esp32_disconnect(main.SimESP32DisconnectCommand(disconnected=True))
        self.assertTrue(res["ok"])
        self.assertTrue(res["disconnected"])
        st = main.status()
        self.assertEqual(st["esp32_comm"]["status"], "DISCONNECTED")

        # Restore ESP32 connection
        res = main.simulate_esp32_disconnect(main.SimESP32DisconnectCommand(disconnected=False))
        self.assertTrue(res["ok"])
        self.assertFalse(res["disconnected"])

        # 6. Run scenario runner handler
        res = main.run_person_scenario(main.PersonScenarioCommand(scenario="all"))
        self.assertTrue(res["ok"])
        self.assertEqual(res["report"]["total_scenarios"], 13)
        self.assertEqual(res["report"]["passed"], 13)


if __name__ == "__main__":
    unittest.main()

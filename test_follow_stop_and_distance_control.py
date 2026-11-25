"""
WALL-E Following Reliability, Stop-Reason Diagnostics & Adjustable Target Distance Verification Suite
Module: test_follow_stop_and_distance_control.py

Automated regression suite verifying:
1. Stop Reason & Diagnostics:
   - A stopped robot (command == 'stop' or linear_speed == 0) never reports stop_reason == 'NONE'
   - Comprehensive structured stop reasons and uppercase snake_case stop codes
   - Diagnostics dictionary in arena state telemetry
2. Following Reliability & Detour Hysteresis:
   - Directional hysteresis prevents left/right oscillation during detours
   - Target lock ID is preserved throughout detour navigation and reacquisition
   - Path clearance and resume movement diagnostics
3. Adjustable Target Following Distance:
   - Default distance setpoint (2.5m) and deadband (2.2m to 2.8m, +-0.3m tolerance)
   - Dynamic setpoint adjustment within [1.0m, 5.0m]
   - Rejection of out-of-bounds values (<1.0m or >5.0m)
   - Following behavior adapts to distance changes without losing target or safety
4. REST API Endpoints:
   - GET /api/follow/distance and GET /api/arena/distance
   - POST /api/follow/distance and POST /api/arena/distance
   - Input validation, boundary error handling, and state synchronization
"""

import os
import sys
import unittest
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(BASE_DIR, "walle_software_starter")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from pydantic import ValidationError
from fastapi import HTTPException

from sim_arena import (
    SimulationArena,
    ArenaRobot,
    ArenaPerson,
    ArenaObstacle,
    NavigationMode,
    AvoidanceState,
    DetourDirection,
    ReacquisitionStatus
)
from follow_controller import FollowController, FollowMode, FollowState, FollowDecision
from esp32_comm import ESP32CommunicationManager, MockSerialTransport
import main
from main import FollowDistanceCommand


class TestStopReasonAndDiagnostics(unittest.TestCase):
    """Verifies that stopped robot states are completely diagnostic and never report 'NONE'."""

    def setUp(self):
        self.mock_transport = MockSerialTransport()
        self.esp32_comm = ESP32CommunicationManager(transport=self.mock_transport, simulation_mode=True)
        self.mock_transport.connect()
        self.controller = FollowController()
        self.controller.enabled = True
        self.arena = SimulationArena(
            follow_controller=self.controller,
            esp32_comm=self.esp32_comm,
            target_following_distance_m=2.5
        )
        self.arena.people = {1: ArenaPerson(1, 5.0, 4.0)}
        self.arena.selected_target_id = 1
        self.arena.obstacles.clear()

    def test_stopped_robot_never_reports_none_reason(self):
        """Invariant: Whenever the robot is stopped, stop_reason and stop_code must never be 'NONE'."""
        # 1. Robot holding distance
        self.arena.robot.x = 2.5
        self.arena.robot.y = 4.0
        self.arena.robot.heading_deg = 0.0
        self.arena.people[1].x = 5.0
        self.arena.people[1].y = 4.0  # exactly 2.5m ahead
        state = self.arena.step()
        self.assertIn(state["robot"]["command"], ("stop", "hold"))
        self.assertNotEqual(state["stop_reason"], "NONE")
        self.assertNotEqual(state["stop_code"], "NONE")

        # 2. Controller disabled
        self.controller.enabled = False
        state = self.arena.step()
        self.assertEqual(state["robot"]["command"], "stop")
        self.assertEqual(state["stop_reason"], "FOLLOW DISABLED")
        self.assertEqual(state["stop_code"], "FOLLOWING_DISABLED")

        # 3. Emergency stop latched
        self.controller.enabled = True
        self.controller.emergency_stop()
        state = self.arena.step()
        self.assertEqual(state["robot"]["command"], "stop")
        self.assertEqual(state["stop_reason"], "E-STOP")
        self.assertEqual(state["stop_code"], "EMERGENCY_STOP")

    def test_stop_reason_comm_failure(self):
        """Simulated communication disconnection triggers COMMUNICATION FAILURE stop reason."""
        self.mock_transport.disconnect()
        state = self.arena.step()
        self.assertEqual(state["robot"]["command"], "stop")
        self.assertEqual(state["stop_reason"], "COMMUNICATION FAILURE")
        self.assertEqual(state["stop_code"], "COMMUNICATION_FAILURE")

    def test_stop_reason_sensor_stale_or_fail(self):
        """Simulated sensor failure triggers SENSOR STALE stop reason."""
        self.arena.simulate_sensor_fail = True
        state = self.arena.step()
        self.assertEqual(state["robot"]["command"], "stop")
        self.assertEqual(state["stop_reason"], "SENSOR STALE")
        self.assertEqual(state["stop_code"], "SENSOR_DATA_STALE")

    def test_stop_reason_obstacle_blocking_path(self):
        """Obstacle directly in front triggers OBSTACLE stop reason and OBSTACLE_BLOCKING_PATH code."""
        self.arena.auto_avoidance_enabled = False
        self.arena.obstacles["obs1"] = ArenaObstacle("obs1", 2.8, 4.0, radius=0.15)
        state = self.arena.step()
        self.assertEqual(state["robot"]["command"], "stop")
        self.assertEqual(state["stop_reason"], "OBSTACLE")
        self.assertEqual(state["stop_code"], "OBSTACLE_BLOCKING_PATH")

    def test_stop_reason_both_routes_blocked(self):
        """When auto avoidance is enabled but all routes are blocked, reports BOTH_ROUTES_BLOCKED."""
        self.arena.auto_avoidance_enabled = True
        # Block ahead, left, and right
        self.arena.obstacles["obs_front"] = ArenaObstacle("obs_front", 3.0, 4.0, radius=0.35)
        self.arena.obstacles["obs_left"] = ArenaObstacle("obs_left", 3.0, 2.5, radius=1.6)
        self.arena.obstacles["obs_right"] = ArenaObstacle("obs_right", 3.0, 5.5, radius=1.6)
        state = self.arena.step()
        self.assertEqual(state["robot"]["command"], "stop")
        self.assertEqual(state["avoidance_state"], AvoidanceState.BLOCKED)
        self.assertEqual(state["stop_code"], "BOTH_ROUTES_BLOCKED")

    def test_stop_reason_target_lost_and_search_progress(self):
        """Target removal reports TARGET_LOST, and search states report TARGET_SEARCH and TARGET_NOT_FOUND."""
        # 1. Target removed from arena -> TARGET_LOST
        self.arena.people.clear()
        state = self.arena.step()
        self.assertEqual(state["robot"]["command"], "stop")
        self.assertEqual(state["stop_reason"], "TARGET LOST")
        self.assertEqual(state["stop_code"], "TARGET_LOST")

        # 2. Target exists behind WALL-E (out of FOV) -> start search -> TARGET_SEARCH
        self.arena.people[1] = ArenaPerson(1, 0.0, 4.0)  # behind robot at x=2.5 heading=0
        self.arena.selected_target_id = 1
        res = self.arena.start_360_search()
        self.assertTrue(res["ok"])
        self.assertEqual(self.arena.get_stop_code(), "TARGET_SEARCH")

        # 3. Full 360 scan completes without reacquiring -> TARGET_NOT_FOUND
        self.arena.search_angle_accumulated = 360.0
        self.arena.target_reacquisition_status = ReacquisitionStatus.NOT_FOUND
        self.arena.search_active = False
        self.arena.navigation_mode = NavigationMode.DIRECT_FOLLOW
        state = self.arena.step()
        self.assertEqual(state["stop_reason"], "TARGET NOT FOUND")
        self.assertEqual(state["stop_code"], "TARGET_NOT_FOUND")

    def test_diagnostics_dictionary_in_state(self):
        """Telemetry includes comprehensive diagnostics dictionary."""
        state = self.arena.get_state()
        self.assertIn("diagnostics", state)
        diag = state["diagnostics"]
        self.assertIn("can_resume_movement", diag)
        self.assertIn("target_in_camera_view", diag)
        self.assertIn("target_in_range", diag)
        self.assertIn("path_clear", diag)
        self.assertIn("obstacle_active", diag)
        self.assertIn("sensor_fail", diag)
        self.assertIn("controller_enabled", diag)
        self.assertIn("controller_stopped", diag)
        self.assertIn("active_stop_reason", diag)
        self.assertIn("active_stop_code", diag)


class TestFollowReliabilityAndDetourHysteresis(unittest.TestCase):
    """Verifies detour path stability, directional hysteresis, and target lock continuity."""

    def setUp(self):
        self.mock_transport = MockSerialTransport()
        self.esp32_comm = ESP32CommunicationManager(transport=self.mock_transport, simulation_mode=True)
        self.mock_transport.connect()
        self.controller = FollowController()
        self.controller.enabled = True
        self.arena = SimulationArena(
            follow_controller=self.controller,
            esp32_comm=self.esp32_comm,
            target_following_distance_m=2.5
        )
        self.arena.robot.x = 2.0
        self.arena.robot.y = 4.0
        self.arena.robot.heading_deg = 0.0
        self.arena.people = {1: ArenaPerson(1, 6.0, 4.0)}
        self.arena.selected_target_id = 1
        self.arena.auto_avoidance_enabled = True

    def test_directional_hysteresis_prevents_oscillation(self):
        """Directional hysteresis bias prevents robot from flip-flopping left and right detour choices."""
        # Place symmetric obstacle in front
        self.arena.obstacles["obs1"] = ArenaObstacle("obs1", 3.0, 4.0, radius=0.3)
        self.arena.plan_detour()
        first_chosen_dir = self.arena.detour_direction
        self.assertIn(first_chosen_dir, (DetourDirection.LEFT, DetourDirection.RIGHT))

        # Perturb person slightly to other side by a tiny delta (0.05m)
        delta_y = -0.05 if first_chosen_dir == DetourDirection.LEFT else 0.05
        self.arena.people[1].y += delta_y

        # Re-evaluating detour must stick to current direction due to hysteresis bias
        self.arena.plan_detour()
        self.assertEqual(
            self.arena.detour_direction,
            first_chosen_dir,
            "Directional hysteresis must maintain detour direction against slight perturbations"
        )

    def test_target_id_preserved_throughout_detour(self):
        """Selected target lock ID remains intact throughout obstacle avoidance."""
        self.arena.obstacles["obs1"] = ArenaObstacle("obs1", 3.0, 4.0, radius=0.3)
        self.arena.people[2] = ArenaPerson(2, 6.0, 2.0)  # secondary bystander

        self.assertEqual(self.arena.selected_target_id, 1)
        for _ in range(5):
            state = self.arena.step(dt=0.05)
            self.assertEqual(self.arena.selected_target_id, 1)
            self.assertEqual(state["selected_target_id"], 1)

    def test_can_resume_movement_diagnostic(self):
        """can_resume_movement correctly reflects whether following can safely advance."""
        self.arena.obstacles.clear()
        self.assertTrue(self.arena.can_resume_movement())

        # Block with obstacle
        self.arena.obstacles["obs1"] = ArenaObstacle("obs1", 2.3, 4.0, radius=0.15)
        self.arena.step()
        self.assertFalse(self.arena.can_resume_movement())

        # Remove obstacle
        self.arena.obstacles.clear()
        self.arena.step()
        self.assertTrue(self.arena.can_resume_movement())


class TestAdjustableFollowingDistance(unittest.TestCase):
    """Verifies target distance configurability, deadband tolerances, and boundary validation."""

    def setUp(self):
        self.mock_transport = MockSerialTransport()
        self.esp32_comm = ESP32CommunicationManager(transport=self.mock_transport, simulation_mode=True)
        self.mock_transport.connect()
        self.controller = FollowController()
        self.controller.enabled = True
        self.arena = SimulationArena(
            follow_controller=self.controller,
            esp32_comm=self.esp32_comm,
            target_following_distance_m=2.5
        )
        self.arena.robot.x = 2.0
        self.arena.robot.y = 4.0
        self.arena.robot.heading_deg = 0.0
        self.arena.people = {1: ArenaPerson(1, 4.5, 4.0)}  # 2.5m ahead
        self.arena.selected_target_id = 1
        self.arena.obstacles.clear()

    def test_default_distance_and_safe_zone(self):
        """Default target distance is 2.5m with safe zone deadband [2.2m, 2.8m]."""
        self.assertAlmostEqual(self.arena.target_following_distance_m, 2.5, places=2)
        self.assertAlmostEqual(self.arena.safe_zone_min_m, 2.2, places=2)
        self.assertAlmostEqual(self.arena.safe_zone_max_m, 2.8, places=2)
        self.assertAlmostEqual(self.arena.distance_tolerance_m, 0.3, places=2)

    def test_set_valid_target_distances(self):
        """Setting distances within [1.0m, 5.0m] updates controller and arena setpoints and safe zones."""
        test_distances = [1.0, 1.5, 2.5, 3.2, 4.0, 5.0]
        for d in test_distances:
            ok_ctrl = self.controller.set_target_distance(d)
            ok_arena = self.arena.set_target_following_distance(d)
            self.assertTrue(ok_ctrl)
            self.assertTrue(ok_arena)
            self.assertAlmostEqual(self.arena.target_following_distance_m, d, places=2)
            self.assertAlmostEqual(self.arena.safe_zone_min_m, max(0.7, d - 0.3), places=2)
            self.assertAlmostEqual(self.arena.safe_zone_max_m, d + 0.3, places=2)
            self.assertAlmostEqual(self.controller.target_distance_m, d, places=2)

    def test_boundary_validation_rejects_out_of_bounds(self):
        """Rejects distances below 1.0m and above 5.0m, preserving previous valid setting."""
        self.arena.set_target_following_distance(2.5)

        # Under 1.0m
        self.assertFalse(self.arena.set_target_following_distance(0.9))
        self.assertFalse(self.arena.set_target_following_distance(0.0))
        self.assertFalse(self.arena.set_target_following_distance(-1.5))
        self.assertFalse(self.controller.set_target_distance(0.8))
        self.assertAlmostEqual(self.arena.target_following_distance_m, 2.5, places=2)

        # Over 5.0m
        self.assertFalse(self.arena.set_target_following_distance(5.1))
        self.assertFalse(self.arena.set_target_following_distance(10.0))
        self.assertFalse(self.controller.set_target_distance(5.5))
        self.assertAlmostEqual(self.arena.target_following_distance_m, 2.5, places=2)

    def test_dynamic_setpoint_changes_behavior_seamlessly(self):
        """Changing setpoint dynamically shifts robot between HOLD and APPROACH without dropping target."""
        # Person is 2.5m away (x=4.5, robot x=2.0)
        # At 2.5m setpoint: safe zone [2.2, 2.8] -> HOLD
        self.arena.set_target_following_distance(2.5)
        state1 = self.arena.step()
        self.assertEqual(state1["decision"], "HOLD")
        self.assertEqual(state1["stop_code"], "TARGET_DISTANCE_REACHED")

        # Now increase target distance setpoint to 3.5m (safe zone: [3.2, 3.8])
        # Current distance is 2.5m (< 3.2m safe zone min) -> robot must HOLD
        self.arena.set_target_following_distance(3.5)
        state2 = self.arena.step()
        self.assertEqual(state2["decision"], "HOLD")

        # Now decrease target distance setpoint to 1.5m (safe zone: [1.2, 1.8])
        # Current distance is 2.5m (> 1.8m safe zone max) -> robot must MOVE_FORWARD
        self.arena.set_target_following_distance(1.5)
        state3 = self.arena.step()
        self.assertEqual(state3["decision"], "MOVE_FORWARD")
        self.assertEqual(state3["robot"]["command"], "forward")
        self.assertEqual(self.arena.selected_target_id, 1)


class TestDistanceApiEndpoints(unittest.TestCase):
    """Verifies FastAPI REST endpoint handlers for distance configuration and boundary checking."""

    def setUp(self):
        main.reset_safety_latch()
        main.sim_arena.reset()
        main.set_follow_distance(FollowDistanceCommand(distance_m=2.5))

    def tearDown(self):
        main.reset_safety_latch()
        main.sim_arena.reset()
        main.set_follow_distance(FollowDistanceCommand(distance_m=2.5))

    def test_get_distance_endpoints(self):
        """GET /api/follow/distance and /api/arena/distance return target distance & safe zones."""
        data1 = main.get_follow_distance()
        self.assertTrue(data1["ok"])
        self.assertIn("target_distance_m", data1)
        self.assertIn("safe_zone_min_m", data1)
        self.assertIn("safe_zone_max_m", data1)
        self.assertIn("distance_tolerance_m", data1)
        self.assertEqual(data1["min_distance_m"], 1.0)
        self.assertEqual(data1["max_distance_m"], 5.0)

    def test_post_distance_endpoints_success(self):
        """POST /api/follow/distance with valid distance updates setpoint and returns 200 payload."""
        cmd = FollowDistanceCommand(distance_m=3.0)
        data = main.set_follow_distance(cmd)
        self.assertTrue(data["ok"])
        self.assertAlmostEqual(data["target_distance_m"], 3.0, places=2)
        self.assertAlmostEqual(data["safe_zone_min_m"], 2.7, places=2)
        self.assertAlmostEqual(data["safe_zone_max_m"], 3.3, places=2)
        self.assertAlmostEqual(main.follow_controller.target_distance_m, 3.0, places=2)
        self.assertAlmostEqual(main.sim_arena.target_following_distance_m, 3.0, places=2)

    def test_post_distance_endpoints_boundary_errors(self):
        """POST /api/follow/distance rejects values outside [1.0, 5.0] with HTTP 400."""
        # Under min
        cmd_low = FollowDistanceCommand(distance_m=0.5)
        with self.assertRaises(HTTPException) as ctx_low:
            main.set_follow_distance(cmd_low)
        self.assertEqual(ctx_low.exception.status_code, 400)
        self.assertIn("out of bounds", ctx_low.exception.detail)

        # Over max
        cmd_high = FollowDistanceCommand(distance_m=5.5)
        with self.assertRaises(HTTPException) as ctx_high:
            main.set_follow_distance(cmd_high)
        self.assertEqual(ctx_high.exception.status_code, 400)
        self.assertIn("out of bounds", ctx_high.exception.detail)

    def test_post_distance_endpoints_invalid_payload(self):
        """Pydantic model rejects missing or non-numeric payload."""
        with self.assertRaises(ValidationError):
            FollowDistanceCommand()

        with self.assertRaises(ValidationError):
            FollowDistanceCommand(distance_m="invalid")


if __name__ == "__main__":
    unittest.main()

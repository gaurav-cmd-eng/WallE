"""
WALL-E Stage 6B: Autonomous Obstacle Navigation & 360-Degree Target Search Verification Suite
Module: test_stage6b_avoidance_search.py

Verifies all 12 Stage 6B Scenarios:
1. Obstacle detected and robot stops.
2. Safe left detour selected when right is blocked.
3. Safe right detour selected when left is blocked.
4. Both directions blocked results in STOP (BLOCKED).
5. Robot completes a detour without collision.
6. Robot reacquires original person after detouring and maintains ~1.5m separation.
7. Target loss initiates 360-degree search.
8. Same target reacquisition ends search.
9. Another person cannot steal target lock during search.
10. Full scan without detection results in TARGET_NOT_FOUND and STOP.
11. Emergency STOP overrides both behaviors immediately.
12. Communication and sensor failures prevent unsafe movement.
"""

import os
import sys
import unittest
import math
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(BASE_DIR, "walle_software_starter")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

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


class TestStage6BAvoidanceAndSearch(unittest.TestCase):
    """Automated verification suite for Stage 6B autonomous obstacle navigation & 360 target search."""

    def setUp(self):
        self.mock_transport = MockSerialTransport()
        self.esp32_comm = ESP32CommunicationManager(transport=self.mock_transport, simulation_mode=True)
        self.controller = FollowController()
        self.controller.command_dispatcher = lambda direction, speed: self.esp32_comm.send_command(direction, speed)
        self.controller.set_mode(FollowMode.PERSON)
        self.controller.set_enabled(True)

        self.arena = SimulationArena(
            follow_controller=self.controller,
            esp32_comm=self.esp32_comm,
            arena_width=10.0,
            arena_height=8.0
        )
        self.arena.reset()
        main.reset_safety_latch()
        main.simulate_esp32_disconnect(main.SimESP32DisconnectCommand(disconnected=False))
        main.clear_simulated_people()

    def tearDown(self):
        self.arena.reset()
        main.reset_safety_latch()
        main.simulate_esp32_disconnect(main.SimESP32DisconnectCommand(disconnected=False))
        main.clear_simulated_people()

    # ------------------------------------------------------------------------
    # Scenario 1: Obstacle detected and robot stops
    # ------------------------------------------------------------------------
    def test_01_obstacle_detected_and_robot_stops(self):
        """When an obstacle is detected within 30cm, robot halts motion immediately."""
        self.arena.auto_avoidance_enabled = False
        # Place obstacle directly in front of WALL-E (robot is at (2.5, 4.0), heading 0)
        self.arena.add_obstacle("obs_front", x=2.95, y=4.0, radius=0.20)
        
        state = self.arena.step(dt=0.05)
        
        self.assertTrue(state["sensors"]["obstacle"], "Ultrasonic barrier should trigger obstacle flag")
        self.assertLess(state["sensors"]["distance_cm"], 30.0, "Distance reading should be < 30cm")
        self.assertEqual(state["robot"]["command"], "stop", "Robot command must be 'stop'")
        self.assertEqual(state["robot"]["speed"], 0, "Motor speed must be 0")
        self.assertEqual(state["stop_reason"], "OBSTACLE", "Stop reason must report OBSTACLE")

    # ------------------------------------------------------------------------
    # Scenario 2: Safe left detour selected
    # ------------------------------------------------------------------------
    def test_02_safe_left_detour_selected(self):
        """When an obstacle is blocking and the right side is blocked, left detour is selected."""
        self.arena.auto_avoidance_enabled = True
        # Blocker in front
        self.arena.add_obstacle("obs_blocker", x=3.10, y=4.0, radius=0.25)
        # Block right side: y < 4.0
        self.arena.add_obstacle("obs_block_right", x=3.10, y=3.10, radius=0.50)

        planned = self.arena.plan_detour()
        self.assertTrue(planned, "Detour planning should succeed")
        self.assertEqual(self.arena.detour_direction, DetourDirection.LEFT, "Should select LEFT detour")
        self.assertEqual(len(self.arena.planned_path), 2, "Planned path should contain W1 and W2")
        # First waypoint should have y > 4.0 (left direction in arena coordinates)
        w1 = self.arena.planned_path[0]
        self.assertGreater(w1[1], 4.0, "W1 should be offset to the left side")

    # ------------------------------------------------------------------------
    # Scenario 3: Safe right detour selected
    # ------------------------------------------------------------------------
    def test_03_safe_right_detour_selected(self):
        """When an obstacle is blocking and the left side is blocked, right detour is selected."""
        self.arena.auto_avoidance_enabled = True
        self.arena.add_obstacle("obs_blocker", x=3.10, y=4.0, radius=0.25)
        # Block left side (y > 4.0)
        self.arena.add_obstacle("obs_block_left", x=3.10, y=4.90, radius=0.50)

        planned = self.arena.plan_detour()
        self.assertTrue(planned, "Detour planning should succeed")
        self.assertEqual(self.arena.detour_direction, DetourDirection.RIGHT, "Should select RIGHT detour")
        w1 = self.arena.planned_path[0]
        self.assertLess(w1[1], 4.0, "W1 should be offset to the right side")

    # ------------------------------------------------------------------------
    # Scenario 4: Both directions blocked results in STOP
    # ------------------------------------------------------------------------
    def test_04_both_directions_blocked_results_in_stop(self):
        """When both left and right flanks are blocked, avoidance transitions to BLOCKED and robot halts."""
        self.arena.auto_avoidance_enabled = True
        self.arena.add_obstacle("obs_blocker", x=3.10, y=4.0, radius=0.25)
        # Block both flanks
        self.arena.add_obstacle("obs_block_left", x=3.10, y=4.90, radius=0.55)
        self.arena.add_obstacle("obs_block_right", x=3.10, y=3.10, radius=0.55)

        state = self.arena.step(dt=0.05)
        
        self.assertEqual(state["avoidance_state"], AvoidanceState.BLOCKED, "State should be BLOCKED")
        self.assertEqual(state["detour_direction"], DetourDirection.NONE, "Detour direction should be NONE")
        self.assertEqual(state["robot"]["command"], "stop", "Robot must remain stopped")
        self.assertEqual(state["robot"]["speed"], 0, "Motor speed must be 0")
        self.assertEqual(state["stop_reason"], "OBSTACLE", "Stop reason must report OBSTACLE")

    # ------------------------------------------------------------------------
    # Scenario 5: Robot completes a detour without collision
    # ------------------------------------------------------------------------
    def test_05_robot_completes_detour_without_collision(self):
        """Robot autonomously executes avoidance waypoints around an obstacle without collision."""
        self.arena.auto_avoidance_enabled = True
        obs_x, obs_y, obs_r = 3.20, 4.0, 0.25
        self.arena.add_obstacle("obs_blocker", x=obs_x, y=obs_y, radius=obs_r)
        
        min_clearance = 999.0
        detour_started = False
        detour_completed = False

        for step_idx in range(120):
            state = self.arena.step(dt=0.05)
            rx, ry = state["robot"]["x"], state["robot"]["y"]
            dist_to_obs = math.hypot(rx - obs_x, ry - obs_y) - obs_r - (self.arena.robot.length / 2.0)
            min_clearance = min(min_clearance, dist_to_obs)
            
            # Assert safety invariant at every single timestep
            self.assertGreater(dist_to_obs, 0.02, f"Collision detected at step {step_idx}: clearance={dist_to_obs:.3f}m")
            
            if state["navigation_mode"] == NavigationMode.OBSTACLE_AVOIDANCE:
                detour_started = True
            if detour_started and state["navigation_mode"] == NavigationMode.DIRECT_FOLLOW:
                detour_completed = True
                break

        self.assertTrue(detour_started, "Obstacle avoidance should have engaged")
        self.assertTrue(detour_completed, "Detour should have navigated all waypoints successfully")
        self.assertGreater(min_clearance, 0.02, f"Minimum clearance must be positive (got {min_clearance:.3f}m)")

    # ------------------------------------------------------------------------
    # Scenario 6: Robot reacquires original person after detouring
    # ------------------------------------------------------------------------
    def test_06_robot_reacquires_original_person_after_detouring(self):
        """After completing obstacle detour, robot returns to DIRECT_FOLLOW and resumes following Person #1."""
        self.arena.auto_avoidance_enabled = True
        self.arena.add_obstacle("obs_blocker", x=3.20, y=4.0, radius=0.25)
        
        # Run until detour completes
        for _ in range(120):
            st = self.arena.step(dt=0.05)
            if st["navigation_mode"] == NavigationMode.DIRECT_FOLLOW and st["avoidance_state"] == AvoidanceState.CLEAR:
                break

        # Now robot should be in DIRECT_FOLLOW targeting Person #1
        final_state = self.arena.step(dt=0.05)
        self.assertEqual(final_state["navigation_mode"], NavigationMode.DIRECT_FOLLOW)
        self.assertEqual(final_state["selected_target_id"], 1, "Locked target must still be Person #1")
        self.assertEqual(final_state["target_reacquisition_status"], ReacquisitionStatus.ACQUIRED)
        self.assertGreater(final_state["target_distance_m"], 1.0)
        self.assertLess(final_state["target_distance_m"], 4.0)

    # ------------------------------------------------------------------------
    # Scenario 7: Target loss initiates 360-degree search
    # ------------------------------------------------------------------------
    def test_07_target_loss_initiates_360_degree_search(self):
        """When target disappears from camera view, robot stops forward movement and enters 360 search."""
        # Move person behind WALL-E outside the 75-degree FOV
        self.arena.move_person(person_id=1, dx=-5.0, dy=0.0)
        st = self.arena.step(dt=0.05)
        
        # Verify robot immediately halts forward progress
        self.assertNotEqual(st["robot"]["command"], "forward", "Robot must not drive forward when target is out of view")
        
        # Start 360 search
        res = self.arena.start_360_search()
        self.assertTrue(res["ok"], "360 search initiation should succeed")
        
        st_search = self.arena.step(dt=0.05)
        self.assertEqual(st_search["navigation_mode"], NavigationMode.TARGET_SEARCH)
        self.assertEqual(st_search["target_reacquisition_status"], ReacquisitionStatus.SEARCHING)
        # Robot rotates in place
        self.assertIn(st_search["robot"]["command"], ["left", "right", "turn_left", "turn_right"])

    # ------------------------------------------------------------------------
    # Scenario 8: Same target reacquisition ends search
    # ------------------------------------------------------------------------
    def test_08_same_target_reacquisition_ends_search(self):
        """During 360 search, when the locked Person #1 re-enters camera view, search immediately ends."""
        # Move person to the right (behind/flank of robot) at (2.5, 6.0)
        self.arena.people[1].x = 2.5
        self.arena.people[1].y = 6.0  # 90 degrees to the right
        
        self.arena.start_360_search()
        reacquired = False
        
        for _ in range(80):
            st = self.arena.step(dt=0.05)
            if st["navigation_mode"] == NavigationMode.DIRECT_FOLLOW:
                reacquired = True
                self.assertEqual(st["target_reacquisition_status"], ReacquisitionStatus.REACQUIRED)
                self.assertEqual(st["selected_target_id"], 1)
                break

        self.assertTrue(reacquired, "Robot should have reacquired Person #1 and returned to DIRECT_FOLLOW")

    # ------------------------------------------------------------------------
    # Scenario 9: Another person cannot steal target lock
    # ------------------------------------------------------------------------
    def test_09_another_person_cannot_steal_target_lock(self):
        """An unselected person crossing the camera FOV during search cannot steal the lock from Person #1."""
        # Move Person #1 out of sight
        self.arena.people[1].x = 1.0
        self.arena.people[1].y = 1.0
        
        # Add Person #2 right in front of WALL-E (2.5, 4.0) facing heading 0 -> at (4.5, 4.0)
        self.arena.add_person(person_id=2, x=4.5, y=4.0)
        self.arena.start_360_search()
        
        st = self.arena.step(dt=0.05)
        self.assertEqual(self.arena.selected_target_id, 1, "Target lock must strictly remain Person #1")
        self.assertNotEqual(self.arena.selected_target_id, 2, "Person #2 must not steal target lock")
        self.assertEqual(st["navigation_mode"], NavigationMode.TARGET_SEARCH, "Search must continue for Person #1")

    # ------------------------------------------------------------------------
    # Scenario 10: Full scan without detection results in TARGET_NOT_FOUND
    # ------------------------------------------------------------------------
    def test_10_full_scan_without_detection_results_in_target_not_found(self):
        """When 360-degree rotation completes without finding the target, robot enters TARGET_NOT_FOUND and stops."""
        # Remove Person #1 completely from arena
        self.arena.remove_person(1)
        self.arena.selected_target_id = 1  # Retain lock ID
        
        # Start search
        self.arena.navigation_mode = NavigationMode.TARGET_SEARCH
        self.arena.search_active = True
        self.arena.search_angle_accumulated = 0.0
        self.arena.target_reacquisition_status = ReacquisitionStatus.SEARCHING

        # At 45 deg/s, 360 degrees takes 8.0s (160 steps at dt=0.05)
        for _ in range(180):
            st = self.arena.step(dt=0.05)
            if st["target_reacquisition_status"] == ReacquisitionStatus.NOT_FOUND:
                break

        final_st = self.arena.step(dt=0.05)
        self.assertEqual(final_st["target_reacquisition_status"], ReacquisitionStatus.NOT_FOUND)
        self.assertEqual(final_st["robot"]["command"], "stop", "Robot must stop when target is not found")
        self.assertEqual(final_st["stop_reason"], "TARGET NOT FOUND")

    # ------------------------------------------------------------------------
    # Scenario 11: Emergency STOP overrides both behaviors
    # ------------------------------------------------------------------------
    def test_11_estop_overrides_both_behaviors(self):
        """Emergency STOP immediately aborts obstacle avoidance detour and 360-degree search."""
        # 1. E-STOP during obstacle avoidance
        self.arena.auto_avoidance_enabled = True
        self.arena.add_obstacle("obs_blocker", x=3.10, y=4.0, radius=0.25)
        self.arena.step(dt=0.05)  # Enters planning/navigating
        
        main.stop()
        self.controller.emergency_stop()
        
        st_estop_avoid = self.arena.step(dt=0.05)
        self.assertEqual(st_estop_avoid["robot"]["command"], "stop", "E-STOP must force stop in avoidance")
        self.assertEqual(st_estop_avoid["robot"]["speed"], 0)
        self.assertEqual(st_estop_avoid["stop_reason"], "E-STOP")

        # 2. E-STOP during 360 search
        main.reset_safety_latch()
        self.controller.reset_emergency_stop()
        self.arena.clear_obstacles()
        self.arena.move_person(1, dx=-5.0, dy=0.0)
        self.arena.start_360_search()
        self.arena.step(dt=0.05)
        
        main.stop()
        self.controller.emergency_stop()
        
        st_estop_search = self.arena.step(dt=0.05)
        self.assertEqual(st_estop_search["robot"]["command"], "stop", "E-STOP must force stop in 360 search")
        self.assertEqual(st_estop_search["robot"]["speed"], 0)
        self.assertEqual(st_estop_search["stop_reason"], "E-STOP")

    # ------------------------------------------------------------------------
    # Scenario 12: Communication and sensor failures prevent unsafe movement
    # ------------------------------------------------------------------------
    def test_12_communication_and_sensor_failures_prevent_unsafe_movement(self):
        """ESP32 disconnect and stale/failed ultrasonic sensor data prevent movement in all modes."""
        # Test A: ESP32 Disconnect prevents movement
        self.mock_transport.simulate_disconnect = True
        st_comm = self.arena.step(dt=0.05)
        self.assertEqual(st_comm["robot"]["command"], "stop")
        self.assertEqual(st_comm["robot"]["speed"], 0)
        self.assertEqual(st_comm["stop_reason"], "COMMUNICATION FAILURE")

        # Start search or avoidance should also be rejected
        res_search = self.arena.start_360_search()
        self.assertFalse(res_search["ok"], "Search must be rejected during ESP32 disconnect")

        # Reconnect ESP32
        self.mock_transport.simulate_disconnect = False

        # Test B: Simulated Sensor Failure prevents movement
        self.arena.simulate_sensor_fail = True
        self.controller.update_sensor_telemetry(distance_cm=0.0, obstacle=False, is_valid=False, is_stale=True)
        st_sensor = self.arena.step(dt=0.05)
        self.assertEqual(st_sensor["robot"]["command"], "stop")
        self.assertEqual(st_sensor["robot"]["speed"], 0)
        self.assertEqual(st_sensor["stop_reason"], "SENSOR STALE")

    # ------------------------------------------------------------------------
    # Scenario 13: Stage 6B API Endpoints
    # ------------------------------------------------------------------------
    def test_13_stage6b_api_endpoints(self):
        """Validates /api/arena/avoidance/toggle, /api/arena/search/start, /api/arena/search/cancel endpoints."""
        # 1. Toggle avoidance
        res1 = main.toggle_arena_avoidance()
        self.assertTrue(res1["ok"])
        initial_val = res1["auto_avoidance_enabled"]
        res2 = main.toggle_arena_avoidance()
        self.assertEqual(res2["auto_avoidance_enabled"], not initial_val)

        # 2. Search start & cancel: move target person out of view first
        main.sim_arena.move_person(1, dx=-5.0, dy=0.0)
        res_start = main.start_arena_target_search()
        self.assertTrue(res_start["ok"])
        self.assertEqual(main.sim_arena.navigation_mode, NavigationMode.TARGET_SEARCH)

        res_cancel = main.cancel_arena_target_search()
        self.assertTrue(res_cancel["ok"])
        self.assertEqual(main.sim_arena.navigation_mode, NavigationMode.DIRECT_FOLLOW)


if __name__ == "__main__":
    unittest.main()

"""
WALL-E Autonomous Safety & State Conflict Regression Test Suite
File: test_state_conflict_and_safety.py

Comprehensive tests for:
1. Original dashboard state conflict resolution (no contradictory states).
2. Autonomous safety priority hierarchy (E-STOP > Comm > Sensor > Obstacle > Detour > Search > Follow > HOLD).
3. Target reacquisition & 360-degree search safety (in-place rotation, anti-hijack lock, immediate termination upon reacquisition).
4. Obstacle avoidance & swept-path clearance validation.
5. Reacquisition protection during active detours (no premature cancellation).
6. Non-zero stop reason verification whenever robot is stopped.
"""

import math
import time
import unittest

from walle_software_starter.sim_arena import (
    SimulationArena,
    NavigationMode,
    AvoidanceState,
    DetourDirection,
    ReacquisitionStatus,
    ArenaPerson,
    ArenaObstacle,
)
from walle_software_starter.follow_controller import FollowController, FollowMode, FollowDecision, FollowState
from walle_software_starter.esp32_comm import ESP32CommunicationManager, MockSerialTransport


class TestStateConflictAndSafetyHierarchy(unittest.TestCase):
    def setUp(self):
        self.mock_transport = MockSerialTransport()
        self.esp32 = ESP32CommunicationManager(transport=self.mock_transport)
        self.controller = FollowController()
        self.controller.set_mode(FollowMode.PERSON)
        self.controller.set_enabled(True)
        self.arena = SimulationArena(follow_controller=self.controller, esp32_comm=self.esp32)
        self.arena.reset()

    # =========================================================================
    # TEST SUITE A: RESOLUTION OF THE ORIGINAL DASHBOARD STATE CONFLICT
    # =========================================================================
    def test_original_dashboard_state_conflict_resolved(self):
        """
        Exact reproduction of the observed state conflict:
        - Target Person #2 locked, distance 4.90m, bearing 10.2°
        - Stale/corrupted fields: avoidance_state=REACQUIRING, search_progress=149°, reacquisition=SEARCHING
        When step() runs, state machine must synchronize and resolve all contradictions.
        """
        # Set up Person #2 at distance 4.90m, bearing 10.2° from robot at (2.5, 4.0, heading 0°)
        self.arena.robot.x = 2.5
        self.arena.robot.y = 4.0
        self.arena.robot.heading_deg = 0.0

        p2_x = 2.5 + 4.90 * math.cos(math.radians(10.2))
        p2_y = 4.0 + 4.90 * math.sin(math.radians(10.2))
        self.arena.add_person(person_id=2, x=p2_x, y=p2_y)
        self.arena.selected_target_id = 2
        self.controller.select_person_target(2)

        # Inject conflicting states from the bug report
        self.arena.avoidance_state = AvoidanceState.REACQUIRING
        self.arena.navigation_mode = NavigationMode.DIRECT_FOLLOW
        self.arena.detour_direction = DetourDirection.NONE
        self.arena.search_angle_accumulated = 149.0
        self.arena.target_reacquisition_status = ReacquisitionStatus.SEARCHING
        self.arena.simulated_distance_cm = 596.3
        self.arena.simulated_obstacle_active = False

        # Execute simulation step
        state = self.arena.step(dt=0.05)

        # 1. Navigation mode must be DIRECT_FOLLOW
        self.assertEqual(state["navigation_mode"], NavigationMode.DIRECT_FOLLOW)

        # 2. Reacquisition status must NOT be SEARCHING while in DIRECT_FOLLOW with target visible
        self.assertNotEqual(state["target_reacquisition_status"], ReacquisitionStatus.SEARCHING)
        self.assertIn(state["target_reacquisition_status"], (ReacquisitionStatus.ACQUIRED, ReacquisitionStatus.REACQUIRED))

        # 3. Search progress must be reset to 0° when not searching
        self.assertEqual(state["target_search_progress_deg"], 0.0)

        # 4. Avoidance state must transition from REACQUIRING to CLEAR once aligned with target
        self.assertEqual(state["avoidance_state"], AvoidanceState.CLEAR)

        # 5. Robot must be commanded to forward motion since target is at 4.9m (>1.8m) and path is clear
        self.assertEqual(state["decision"], "MOVE_FORWARD")
        self.assertEqual(self.arena.robot.current_cmd, "forward")
        self.assertGreater(self.arena.robot.linear_speed, 0.0)
        self.assertEqual(state["stop_reason"], "NONE")

    def test_stopped_robot_never_reports_none_stop_reason(self):
        """
        When the robot is stopped (linear speed 0 or command 'stop'),
        the stop reason must accurately describe why and NEVER be 'NONE'.
        """
        # Case 1: Controller disabled
        self.controller.set_enabled(False)
        st = self.arena.step(dt=0.05)
        self.assertEqual(st["robot"]["command"], "stop")
        self.assertEqual(st["stop_reason"], "FOLLOW DISABLED")

        # Case 2: Inside safe holding distance (1.5m)
        self.controller.set_enabled(True)
        self.arena.reposition_person_ahead(distance_m=1.5)
        st = self.arena.step(dt=0.05)
        self.assertEqual(st["robot"]["command"], "stop")
        self.assertEqual(st["decision"], "HOLD")
        self.assertEqual(st["stop_reason"], "HOLDING DISTANCE")

        # Case 3: Target removed (lost)
        self.arena.remove_person(1)
        st = self.arena.step(dt=0.05)
        self.assertEqual(st["robot"]["command"], "stop")
        self.assertEqual(st["stop_reason"], "TARGET LOST")

    # =========================================================================
    # TEST SUITE B: CENTRALIZED SAFETY HIERARCHY
    # =========================================================================
    def test_hierarchy_level_1_estop_overrides_all(self):
        """Emergency STOP must latch and override obstacle avoidance, search, and following."""
        self.arena.start_360_search()
        self.assertEqual(self.arena.navigation_mode, NavigationMode.TARGET_SEARCH)

        # Trigger E-STOP
        self.controller.emergency_stop()
        st = self.arena.step(dt=0.05)

        self.assertEqual(st["decision"], "STOP")
        self.assertEqual(st["stop_reason"], "E-STOP")
        self.assertEqual(st["robot"]["command"], "stop")
        self.assertEqual(st["robot"]["linear_speed"], 0.0)
        self.assertEqual(st["robot"]["angular_speed"], 0.0)
        self.assertFalse(st["search_active"])

    def test_hierarchy_level_2_comm_failure_overrides_movement(self):
        """ESP32 disconnect must immediately stop the robot with COMMUNICATION FAILURE."""
        self.mock_transport.disconnect()
        st = self.arena.step(dt=0.05)

        self.assertEqual(st["decision"], "STOP")
        self.assertEqual(st["stop_reason"], "COMMUNICATION FAILURE")
        self.assertEqual(st["robot"]["command"], "stop")
        self.assertEqual(st["robot"]["linear_speed"], 0.0)

    def test_hierarchy_level_3_stale_sensor_overrides_movement(self):
        """Stale or invalid ultrasonic sensor telemetry must halt robot with SENSOR STALE."""
        self.arena.simulate_sensor_fail = True
        st = self.arena.step(dt=0.05)

        self.assertEqual(st["decision"], "STOP")
        self.assertEqual(st["stop_reason"], "SENSOR STALE")
        self.assertEqual(st["robot"]["command"], "stop")
        self.assertEqual(st["robot"]["linear_speed"], 0.0)

    def test_hierarchy_level_4_obstacle_overrides_target_search(self):
        """An obstacle detected during 360 target search must immediately abort search and halt."""
        self.arena.start_360_search()
        self.assertTrue(self.arena.search_active)
        self.assertEqual(self.arena.navigation_mode, NavigationMode.TARGET_SEARCH)

        # Inject obstacle directly in front (< 30cm)
        self.arena.add_obstacle("blocking_obs", x=self.arena.robot.x + 0.3, y=self.arena.robot.y, radius=0.2)
        st = self.arena.step(dt=0.05)

        # Target search must be aborted
        self.assertFalse(self.arena.search_active)
        self.assertNotEqual(st["navigation_mode"], NavigationMode.TARGET_SEARCH)
        self.assertEqual(st["sensors"]["obstacle"], True)

    # =========================================================================
    # TEST SUITE C: TARGET REACQUISITION & 360-DEGREE SEARCH
    # =========================================================================
    def test_search_strictly_in_place_zero_linear_velocity(self):
        """During 360-degree target search, linear speed must be strictly 0.0 (in-place rotation only)."""
        # Place person behind robot so search begins
        self.arena.people[1].x = self.arena.robot.x - 3.0
        self.arena.people[1].y = self.arena.robot.y

        res = self.arena.start_360_search()
        self.assertTrue(res["ok"])

        # Run several search steps
        for _ in range(5):
            st = self.arena.step(dt=0.05)
            self.assertEqual(st["navigation_mode"], NavigationMode.TARGET_SEARCH)
            self.assertEqual(st["robot"]["linear_speed"], 0.0)
            self.assertNotEqual(st["robot"]["angular_speed"], 0.0)
            self.assertIn(st["robot"]["command"], ("left", "right"))

    def test_search_terminates_immediately_upon_target_reacquisition(self):
        """As soon as the locked target enters FOV cone (|bearing| <= 15°), search terminates cleanly."""
        # Start search with target behind
        self.arena.people[1].x = self.arena.robot.x - 3.0
        self.arena.people[1].y = self.arena.robot.y
        self.arena.start_360_search()

        self.assertTrue(self.arena.search_active)
        self.assertEqual(self.arena.navigation_mode, NavigationMode.TARGET_SEARCH)

        # Move target into camera FOV cone directly ahead
        self.arena.people[1].x = self.arena.robot.x + 2.5
        self.arena.people[1].y = self.arena.robot.y

        st = self.arena.step(dt=0.05)

        # Search must be terminated
        self.assertFalse(self.arena.search_active)
        self.assertEqual(st["navigation_mode"], NavigationMode.DIRECT_FOLLOW)
        self.assertEqual(st["target_reacquisition_status"], ReacquisitionStatus.REACQUIRED)
        self.assertEqual(st["target_search_progress_deg"], 0.0)
        self.assertEqual(st["decision"], "HOLD")

    def test_search_completes_full_360_when_target_missing(self):
        """If target is never found, search stops at 360° with NOT_FOUND."""
        self.arena.remove_person(1)
        # Add target person outside camera range (e.g. 15m away)
        self.arena.add_person(person_id=1, x=20.0, y=20.0)
        self.arena.selected_target_id = 1
        self.arena.start_360_search()

        # Step until 360 degrees rotation completes
        for _ in range(200):
            st = self.arena.step(dt=0.05)
            if not self.arena.search_active:
                break

        self.assertFalse(self.arena.search_active)
        self.assertEqual(st["target_search_progress_deg"], 360.0)
        self.assertEqual(st["target_reacquisition_status"], ReacquisitionStatus.NOT_FOUND)
        self.assertEqual(st["navigation_mode"], NavigationMode.DIRECT_FOLLOW)
        self.assertEqual(st["stop_reason"], "TARGET NOT FOUND")

    # =========================================================================
    # TEST SUITE D: OBSTACLE DETOUR PROTECTION & REACQUISITION
    # =========================================================================
    def test_reacquisition_rejected_during_active_detour(self):
        """Calling reacquire_target or start_360_search during active detour navigation must be rejected."""
        # Put robot in detour navigation
        self.arena.navigation_mode = NavigationMode.OBSTACLE_AVOIDANCE
        self.arena.avoidance_state = AvoidanceState.NAVIGATING
        self.arena.planned_path = [(3.0, 4.0), (4.0, 4.0)]

        res_reacq = self.arena.reacquire_target()
        self.assertFalse(res_reacq["ok"])
        self.assertEqual(res_reacq["reason"], "OBSTACLE DETOUR ACTIVE")

        res_search = self.arena.start_360_search()
        self.assertFalse(res_search["ok"])
        self.assertEqual(res_search["reason"], "OBSTACLE DETOUR ACTIVE")

        # Detour waypoints and state must remain intact
        self.assertEqual(self.arena.navigation_mode, NavigationMode.OBSTACLE_AVOIDANCE)
        self.assertEqual(self.arena.avoidance_state, AvoidanceState.NAVIGATING)
        self.assertEqual(len(self.arena.planned_path), 2)

    def test_first_recovery_cycle_hold_invariant(self):
        """
        When an obstacle clears, the robot must execute at least one cycle of HOLD
        before resuming forward motion, preventing immediate sudden acceleration.
        """
        # Place obstacle in front (< 30cm)
        self.arena.add_obstacle("temp_obs", x=self.arena.robot.x + 0.35, y=self.arena.robot.y, radius=0.15)
        st_obs = self.arena.step(dt=0.05)
        self.assertTrue(self.arena.arena_was_in_obstacle_stop)

        # Clear obstacle
        self.arena.remove_obstacle("temp_obs")
        self.arena.simulated_distance_cm = 400.0
        self.arena.simulated_obstacle_active = False

        # First cycle after clear: MUST honor HOLD
        st_recovery = self.arena.step(dt=0.05)
        self.assertEqual(st_recovery["decision"], "HOLD")
        self.assertEqual(st_recovery["robot"]["command"], "stop")
        self.assertEqual(st_recovery["robot"]["linear_speed"], 0.0)

    def test_footprint_swept_path_clearance_validation(self):
        """Validates that check_segment_clearance accurately accounts for robot radius + margin."""
        # Obstacle at (5.0, 5.0) with radius 0.4
        self.arena.obstacles = {"obs_test": ArenaObstacle("obs_test", x=5.0, y=5.0, radius=0.4)}

        # Segment passing 0.2m from obstacle center: must fail (0.2 < 0.4 + 0.35 + 0.15 = 0.90)
        clear = self.arena.check_segment_clearance(4.8, 2.0, 4.8, 8.0, margin=0.15)
        self.assertFalse(clear)

        # Segment passing 1.5m from obstacle center inside arena boundary: must pass
        clear_wide = self.arena.check_segment_clearance(7.0, 2.0, 7.0, 7.0, margin=0.15)
        self.assertTrue(clear_wide)


if __name__ == "__main__":
    unittest.main()

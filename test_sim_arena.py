"""
WALL-E Stage 6: Interactive 2D Simulation Arena Automated Verification Suite
Module: test_sim_arena.py

Verifies:
1. Robot follows a moving person across the 2D arena.
2. Robot maintains the configured following distance (~1.5m).
3. Left/right person displacement produces steering corrections.
4. Target loss (removing person) halts robot with STOP.
5. Obstacle detection within 30cm overrides movement.
6. Stale/failed ultrasonic telemetry inhibits forward approach.
7. Emergency STOP overrides all commands and halts motion.
8. ESP32 disconnection causes a safe stop and reports DISCONNECTED.
9. Resetting emergency STOP does not automatically resume following.
10. Simulation reset returns the scene to a reproducible initial state.
11. Backend API endpoints for arena state, stepping, movement, and obstacle management.
"""

import os
import sys
import unittest
import math
import time

# Ensure walle_software_starter is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(BASE_DIR, "walle_software_starter")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from sim_arena import SimulationArena, ArenaRobot, ArenaPerson, ArenaObstacle
from follow_controller import FollowController, FollowMode, FollowState, FollowDecision
from esp32_comm import ESP32CommunicationManager, MockSerialTransport
import main


class TestSimulationArenaStage6(unittest.TestCase):
    """Automated integration tests for the interactive 2D simulation arena."""

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

    def tearDown(self):
        self.arena.reset()
        main.reset_safety_latch()
        main.simulate_esp32_disconnect(main.SimESP32DisconnectCommand(disconnected=False))
        main.clear_simulated_people()
        main.follow_controller.update_sensor_telemetry(distance_cm=85.0, obstacle=False, is_valid=True, is_stale=False)
        main.esp32_comm.latest_telemetry = None

    def test_1_robot_follows_moving_person(self):
        """1. Robot visibly moves and approaches when person moves farther away."""
        # Initial: Robot at (2.5, 4.0), Person 1 at (5.0, 4.0), Distance = 2.5m (> 1.8m approach threshold)
        initial_rx = self.arena.robot.x
        st = self.arena.step(dt=0.1)

        self.assertEqual(st["decision"], "MOVE_FORWARD")
        self.assertEqual(self.arena.robot.current_cmd, "forward")
        self.assertGreater(self.arena.robot.x, initial_rx, "Robot X coordinate must advance toward person")

    def test_2_robot_maintains_configured_following_distance(self):
        """2. Robot stops forward movement and enters HOLD when inside following distance zone (1.2m - 1.8m)."""
        # Place person at exactly 1.5m straight ahead
        self.arena.robot.x = 2.0
        self.arena.robot.y = 4.0
        self.arena.robot.heading_deg = 0.0
        self.arena.people[1].x = 3.5  # 1.5m separation
        self.arena.people[1].y = 4.0

        st = self.arena.step(dt=0.1)
        self.assertEqual(st["decision"], "HOLD")
        self.assertEqual(self.arena.robot.current_cmd, "stop")
        self.assertEqual(self.arena.robot.linear_speed, 0.0)

    def test_3_steering_corrections_left_and_right(self):
        """3. Left/right person movement relative to robot heading produces appropriate steering corrections."""
        self.arena.robot.x = 2.0
        self.arena.robot.y = 4.0
        self.arena.robot.heading_deg = 0.0

        # Move person to LEFT (negative Y in screen coords or higher angle)
        self.arena.people[1].x = 4.0
        self.arena.people[1].y = 2.0  # Bearing ~ -45 deg
        st_left = self.arena.step(dt=0.1)
        self.assertEqual(st_left["decision"], "TURN_LEFT")
        self.assertEqual(self.arena.robot.current_cmd, "left")
        self.assertLess(self.arena.robot.angular_speed, 0.0)

        # Move person to RIGHT (positive Y or lower angle)
        self.arena.people[1].x = 4.0
        self.arena.people[1].y = 6.0  # Bearing ~ +45 deg
        st_right = self.arena.step(dt=0.1)
        self.assertEqual(st_right["decision"], "TURN_RIGHT")
        self.assertEqual(self.arena.robot.current_cmd, "right")
        self.assertGreater(self.arena.robot.angular_speed, 0.0)

    def test_4_target_loss_causes_stop(self):
        """4. Robot halts with STOP when target person is removed from arena."""
        self.arena.remove_person(1)
        # Advance multiple steps past disappearance threshold
        for _ in range(6):
            st = self.arena.step(dt=0.05)

        self.assertEqual(st["decision"], "STOP")
        self.assertEqual(self.arena.robot.linear_speed, 0.0)
        self.assertEqual(self.controller.state, FollowState.TARGET_LOST)

    def test_5_obstacle_detection_overrides_movement(self):
        """5. Obstacle detection (<= 30cm) overrides person following and halts motion."""
        # Person is at 2.5m (requesting forward motion)
        # Inject obstacle 20cm ahead of robot
        self.arena.add_obstacle("blocker", x=2.7, y=4.0, radius=0.15)
        st = self.arena.step(dt=0.05)

        self.assertEqual(st["decision"], "STOP")
        self.assertEqual(self.arena.robot.linear_speed, 0.0)
        self.assertEqual(self.controller.state, FollowState.OBSTACLE_STOP)
        self.assertTrue(st["sensors"]["obstacle"])

    def test_6_stale_or_failed_sensor_data_prevents_forward(self):
        """6. Missing or stale sensor data inhibits forward approach."""
        self.arena.simulate_sensor_fail = True
        st = self.arena.step(dt=0.05)

        self.assertEqual(st["decision"], "STOP")
        self.assertEqual(self.arena.robot.linear_speed, 0.0)
        self.assertFalse(self.controller.is_forward_path_clear()[0])

    def test_7_emergency_stop_overrides_all(self):
        """7. Emergency STOP halts all motion immediately and persists across steps."""
        self.controller.emergency_stop()
        self.assertTrue(self.controller.emergency_stop_latched)

        st = self.arena.step(dt=0.05)
        self.assertEqual(st["decision"], "STOP")
        self.assertEqual(self.arena.robot.linear_speed, 0.0)
        self.assertEqual(self.arena.robot.angular_speed, 0.0)

    def test_8_esp32_disconnection_causes_safe_stop(self):
        """8. ESP32 serial communication failure fails safe to STOP and DISCONNECTED."""
        self.mock_transport.simulate_disconnect = True
        res = self.esp32_comm.send_command("forward", speed=150)

        self.assertFalse(res["ok"])
        self.assertEqual(self.esp32_comm.connection_status, "DISCONNECTED")
        self.assertEqual(self.esp32_comm.current_state, "STOPPED")

    def test_9_reset_does_not_automatically_resume_following(self):
        """9. Resetting safety latch enters IDLE and requires explicit follow re-enable."""
        self.controller.emergency_stop()
        self.controller.reset_emergency_stop()

        self.assertFalse(self.controller.emergency_stop_latched)
        self.assertFalse(self.controller.enabled)
        self.assertEqual(self.controller.state, FollowState.IDLE)

        st = self.arena.step(dt=0.05)
        self.assertEqual(st["decision"], "STOP")
        self.assertEqual(self.arena.robot.linear_speed, 0.0)

    def test_10_simulation_reset_returns_scene_to_initial_state(self):
        """10. Scene reset returns robot, person, obstacles, and metrics to clean initial states."""
        # Mutate arena
        self.arena.move_person(1, 3.0, 2.0)
        self.arena.add_person(person_id=2, x=7.0, y=7.0)
        self.arena.robot.x = 8.0
        self.arena.robot.y = 1.0

        # Reset
        self.arena.reset()
        st = self.arena.get_state()

        self.assertEqual(st["robot"]["x"], 2.5)
        self.assertEqual(st["robot"]["y"], 4.0)
        self.assertEqual(st["robot"]["heading_deg"], 0.0)
        self.assertEqual(len(st["people"]), 1)
        self.assertEqual(st["selected_target_id"], 1)
        self.assertEqual(st["target_distance_m"], 2.5)

    def test_11_backend_arena_rest_endpoints(self):
        """11. Direct verification of FastAPI arena REST route handlers in main.py."""
        # Get state
        res = main.get_arena_state()
        self.assertTrue(res["ok"])
        self.assertIn("robot", res["state"])

        # Move person
        move_res = main.move_arena_person(main.ArenaMovePersonCommand(person_id=1, dx=0.5, dy=0.0))
        self.assertTrue(move_res["ok"])

        # Add second person
        add_res = main.add_arena_person(main.ArenaAddPersonCommand(x=6.0, y=3.0))
        self.assertTrue(add_res["ok"])
        pid2 = add_res["person_id"]

        # Switch target
        tgt_res = main.set_arena_target(pid2)
        self.assertTrue(tgt_res["ok"])
        self.assertEqual(tgt_res["target_id"], pid2)

        # Remove person
        del_res = main.remove_arena_person(pid2)
        self.assertTrue(del_res["ok"])

        # Step arena
        step_res = main.step_arena(dt=0.05)
        self.assertTrue(step_res["ok"])

        # Reset arena
        rst_res = main.reset_arena()
        self.assertTrue(rst_res["ok"])
        self.assertEqual(rst_res["state"]["robot"]["x"], 2.5)

    def test_12_obstacle_injection_and_recovery_with_first_cycle_hold(self):
        """
        TASK 4 Items 1-3:
        1. Obstacle injection stops WALL-E.
        2. Removing the obstacle and supplying fresh clear readings permits recovery.
        3. First recovery cycle remains HOLD.
        """
        self.arena.robot.x = 2.5
        self.arena.robot.y = 4.0
        self.arena.robot.heading_deg = 0.0
        self.arena.people[1].x = 6.0
        self.arena.people[1].y = 4.0

        # Verify initial forward movement
        st_initial = self.arena.step(dt=0.05)
        self.assertEqual(st_initial["decision"], "MOVE_FORWARD")
        self.assertEqual(self.arena.robot.current_cmd, "forward")

        # 1. Inject obstacle at 25cm in front (threshold is 30cm)
        self.arena.add_obstacle("blocker_25cm", x=2.75, y=4.0, radius=0.15)
        st_obs = self.arena.step(dt=0.05)
        self.assertEqual(st_obs["decision"], "STOP")
        self.assertEqual(self.arena.robot.current_cmd, "stop")
        self.assertEqual(self.arena.robot.linear_speed, 0.0)
        self.assertEqual(st_obs["stop_reason"], "OBSTACLE")
        self.assertTrue(st_obs["sensors"]["obstacle"])

        # 2. Removing obstacle & supplying fresh clear readings
        self.arena.remove_obstacle("blocker_25cm")

        # 3. First recovery cycle MUST strictly be HOLD
        st_recov_1 = self.arena.step(dt=0.05)
        self.assertEqual(st_recov_1["decision"], "HOLD", "First recovery cycle must strictly remain HOLD")
        self.assertEqual(self.arena.robot.current_cmd, "stop")
        self.assertEqual(self.arena.robot.linear_speed, 0.0)

        # Subsequent step with clear sensor readings permits forward following
        st_recov_2 = self.arena.step(dt=0.05)
        self.assertEqual(st_recov_2["decision"], "MOVE_FORWARD")
        self.assertEqual(self.arena.robot.current_cmd, "forward")
        self.assertGreater(self.arena.robot.linear_speed, 0.0)

    def test_13_person_behind_walle_and_safe_target_reacquisition(self):
        """
        TASK 4 Items 4-5:
        4. Person moving behind WALL-E causes safe behavior (STOP).
        5. WALL-E can reacquire the same locked target via controlled in-place rotation.
        """
        self.arena.robot.x = 4.0
        self.arena.robot.y = 4.0
        self.arena.robot.heading_deg = 0.0

        # 4. Move person behind WALL-E (e.g. at (1.5, 4.0) -> bearing -180 deg)
        self.arena.people[1].x = 1.5
        self.arena.people[1].y = 4.0

        st_behind = self.arena.step(dt=0.05)
        # Must halt safely with STOP; must not blindly drive forward or wander
        self.assertEqual(st_behind["decision"], "STOP")
        self.assertEqual(self.arena.robot.current_cmd, "stop")
        self.assertEqual(self.arena.robot.linear_speed, 0.0)
        self.assertEqual(st_behind["stop_reason"], "TARGET OUT OF VIEW")

        # 5. Trigger controlled safe reacquisition search
        reacq = self.arena.reacquire_target()
        self.assertTrue(reacq["ok"])
        self.assertTrue(self.arena.search_active)

        # Step arena during search: robot rotates in place with ZERO forward speed
        st_search = self.arena.step(dt=0.05)
        self.assertIn(st_search["decision"], ("TURN_LEFT", "TURN_RIGHT"))
        self.assertIn(self.arena.robot.current_cmd, ("left", "right"))
        self.assertEqual(self.arena.robot.linear_speed, 0.0, "Search must be in-place rotation only")

        # Step until person is reacquired and centered in front of WALL-E
        # Reposition person ahead to simulate reacquisition completion
        self.arena.reposition_person_ahead(distance_m=2.5)
        st_reacquired = self.arena.step(dt=0.05)
        self.assertFalse(self.arena.search_active, "Search must stop when target is reacquired")
        self.assertEqual(st_reacquired["decision"], "MOVE_FORWARD")
        self.assertEqual(self.arena.robot.current_cmd, "forward")

    def test_14_target_lock_preservation_against_lock_stealing(self):
        """
        TASK 4 Item 6: Another person cannot steal the existing target lock.
        """
        self.arena.robot.x = 2.5
        self.arena.robot.y = 4.0
        self.arena.robot.heading_deg = 0.0
        self.arena.people[1].x = 5.0
        self.arena.people[1].y = 4.0

        # Step once to establish Person 1 lock
        self.arena.step(dt=0.05)
        self.assertEqual(self.arena.selected_target_id, 1)

        # Add Person #2 closer to the robot
        self.arena.add_person(person_id=2, x=3.5, y=4.0)

        # Step multiple times: lock must remain on Person #1
        for _ in range(5):
            st = self.arena.step(dt=0.05)
            self.assertEqual(st["selected_target_id"], 1, "Person #2 must NOT steal target lock")
            self.assertEqual(self.controller.selected_person_id, 1)

    def test_15_safety_overrides_and_exact_stop_reasons(self):
        """
        TASK 4 Item 7: E-STOP and all other safety priorities remain intact and report exact stop reason.
        """
        self.arena.robot.x = 2.5
        self.arena.robot.y = 4.0
        self.arena.people[1].x = 5.0
        self.arena.people[1].y = 4.0

        # E-STOP
        self.controller.emergency_stop()
        st_estop = self.arena.step(dt=0.05)
        self.assertEqual(st_estop["decision"], "STOP")
        self.assertEqual(st_estop["stop_reason"], "E-STOP")
        self.controller.reset_emergency_stop()
        self.controller.set_mode(FollowMode.PERSON)
        self.controller.set_enabled(True)

        # Obstacle Stop
        self.arena.add_obstacle("obs_test", x=2.7, y=4.0, radius=0.15)
        st_obs = self.arena.step(dt=0.05)
        self.assertEqual(st_obs["stop_reason"], "OBSTACLE")
        self.arena.remove_obstacle("obs_test")

        # Sensor Stale / Fail
        self.arena.simulate_sensor_fail = True
        st_stale = self.arena.step(dt=0.05)
        self.assertEqual(st_stale["stop_reason"], "SENSOR STALE")
        self.arena.simulate_sensor_fail = False

        # Target Lost (remove person)
        self.arena.remove_person(1)
        st_lost = self.arena.step(dt=0.05)
        self.assertEqual(st_lost["stop_reason"], "TARGET LOST")

    def test_16_new_arena_endpoints(self):
        """Verify new REST endpoints for clear obstacles, reposition person, and reacquire."""
        # Add obstacle then clear all
        self.arena.add_obstacle("temp_obs", x=3.0, y=4.0)
        clear_res = main.clear_arena_obstacles()
        self.assertTrue(clear_res["ok"])
        self.assertEqual(len(clear_res["state"]["obstacles"]), 0)

        # Reposition person ahead
        repo_res = main.reposition_arena_person(main.ArenaRepositionPersonCommand(distance_m=2.5))
        self.assertTrue(repo_res["ok"])
        self.assertAlmostEqual(repo_res["state"]["target_distance_m"], 2.5, delta=0.2)

        # Reacquire target
        reacq_res = main.reacquire_arena_target()
        self.assertTrue(reacq_res["ok"])


if __name__ == "__main__":
    unittest.main()

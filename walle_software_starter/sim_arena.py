"""
WALL-E Stage 6: Interactive 2D Simulation Arena Engine
Module: sim_arena.py

Provides a real-time, deterministic 2D top-down simulation environment:
- Virtual WALL-E robot with 2D pose (x, y, heading_deg, linear/angular velocity)
- Simulated person(s) with controllable position (keyboard, scripts, presets)
- Environmental obstacles with bounding boxes
- Feeds camera perspective & bounding box to PersonTracker
- Feeds ultrasonic range & obstacle proximity to FollowController & ESP32CommManager
- Updates robot position and heading from FollowController decisions and ESP32 motor commands
- Full state telemetry for canvas rendering and live dashboard synchronization
"""

import math
import time
from typing import Dict, List, Optional, Tuple, Any

try:
    from person_tracker import PersonObservation
    from follow_controller import FollowDecision, FollowState, FollowMode
    from esp32_comm import ESP32Command, MockSerialTransport
except ImportError:
    from walle_software_starter.person_tracker import PersonObservation
    from walle_software_starter.follow_controller import FollowDecision, FollowState, FollowMode
    from walle_software_starter.esp32_comm import ESP32Command, MockSerialTransport


class NavigationMode:
    DIRECT_FOLLOW = "DIRECT_FOLLOW"
    OBSTACLE_AVOIDANCE = "OBSTACLE_AVOIDANCE"
    TARGET_SEARCH = "TARGET_SEARCH"


class AvoidanceState:
    CLEAR = "CLEAR"
    PLANNING = "PLANNING"
    NAVIGATING = "NAVIGATING"
    REACQUIRING = "REACQUIRING"
    BLOCKED = "BLOCKED"


class DetourDirection:
    NONE = "NONE"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class ReacquisitionStatus:
    ACQUIRED = "ACQUIRED"
    SEARCHING = "SEARCHING"
    REACQUIRED = "REACQUIRED"
    NOT_FOUND = "NOT_FOUND"


def point_to_segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Computes minimum Euclidean distance from point (px, py) to line segment (x1, y1)-(x2, y2)."""
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-6:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


class ArenaObstacle:
    """Rectangular or circular obstacle in the 2D simulation arena."""
    def __init__(self, obs_id: str, x: float, y: float, radius: float = 0.35, label: str = "Obstacle"):
        self.id = obs_id
        self.x = x  # meters
        self.y = y  # meters
        self.radius = radius
        self.label = label

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "radius": round(self.radius, 3),
            "label": self.label
        }


class ArenaPerson:
    """Simulated person in the 2D arena."""
    def __init__(self, person_id: int, x: float, y: float, speed: float = 1.0, is_target: bool = False):
        self.id = person_id
        self.x = x  # meters
        self.y = y  # meters
        self.speed = speed  # m/s
        self.is_target = is_target
        self.trail: List[Tuple[float, float]] = [(x, y)]
        self.max_trail = 30

    def move(self, dx: float, dy: float, arena_w: float = 10.0, arena_h: float = 10.0):
        self.x = max(0.5, min(arena_w - 0.5, self.x + dx))
        self.y = max(0.5, min(arena_h - 0.5, self.y + dy))
        self.trail.append((self.x, self.y))
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "speed": round(self.speed, 2),
            "is_target": self.is_target,
            "trail": [[round(px, 3), round(py, 3)] for px, py in self.trail[-15:]]
        }


class ArenaRobot:
    """
    Virtual WALL-E rover with 2D kinematic differential-drive motion.
    Heading 0 deg = pointing along positive X axis (+X: right, +Y: down).
    """
    def __init__(self, x: float = 2.0, y: float = 5.0, heading_deg: float = 0.0):
        self.x = x  # meters
        self.y = y  # meters
        self.heading_deg = heading_deg  # 0 to 360 deg
        self.width = 0.45   # meters (rover footprint)
        self.length = 0.50  # meters
        self.linear_speed = 0.0   # m/s
        self.angular_speed = 0.0  # deg/s
        self.current_cmd = "stop"
        self.current_speed = 0
        self.trail: List[Tuple[float, float]] = [(x, y)]
        self.max_trail = 40

    def update_physics(self, dt: float, arena_w: float = 10.0, arena_h: float = 10.0):
        # Update orientation
        self.heading_deg = (self.heading_deg + self.angular_speed * dt) % 360.0
        rad = math.radians(self.heading_deg)

        # Update position
        dist = self.linear_speed * dt
        self.x = max(0.4, min(arena_w - 0.4, self.x + dist * math.cos(rad)))
        self.y = max(0.4, min(arena_h - 0.4, self.y + dist * math.sin(rad)))

        self.trail.append((self.x, self.y))
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)

    def apply_command(self, cmd: str, speed: int):
        self.current_cmd = cmd.lower()
        self.current_speed = speed
        max_v = 0.8  # m/s maximum forward speed at speed 255
        v = (speed / 255.0) * max_v if speed > 0 else 0.5

        turn_rate = 65.0  # deg/s rotation rate

        if self.current_cmd in ("forward", "fwd", "move_forward"):
            self.linear_speed = v
            self.angular_speed = 0.0
        elif self.current_cmd in ("backward", "back", "reverse"):
            self.linear_speed = -v * 0.6
            self.angular_speed = 0.0
        elif self.current_cmd in ("left", "turn_left"):
            self.linear_speed = 0.0
            self.angular_speed = -turn_rate
        elif self.current_cmd in ("right", "turn_right"):
            self.linear_speed = 0.0
            self.angular_speed = turn_rate
        else:  # stop, hold
            self.linear_speed = 0.0
            self.angular_speed = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "heading_deg": round(self.heading_deg, 1),
            "linear_speed": round(self.linear_speed, 2),
            "angular_speed": round(self.angular_speed, 1),
            "command": self.current_cmd,
            "speed": self.current_speed,
            "trail": [[round(px, 3), round(py, 3)] for px, py in self.trail[-25:]]
        }


class SimulationArena:
    """
    Complete 2D Simulation Arena orchestrator.
    Bridges visual simulation with existing FollowController, PersonTracker, and ESP32CommManager.
    """
    def __init__(self,
                 follow_controller=None,
                 esp32_comm=None,
                 arena_width: float = 10.0,
                 arena_height: float = 8.0,
                 camera_fov_deg: float = 75.0,
                 camera_range_m: float = 6.0,
                 target_following_distance_m: Optional[float] = None):
        self.arena_width = arena_width
        self.arena_height = arena_height
        self.camera_fov_deg = camera_fov_deg
        self.camera_range_m = camera_range_m

        self.follow_controller = follow_controller
        self.esp32_comm = esp32_comm

        # Configurable Person-Following Distance
        self.min_following_distance_m = 1.0
        self.max_following_distance_m = 5.0
        self.distance_tolerance_m = 0.3
        if target_following_distance_m is not None:
            self.target_following_distance_m = float(target_following_distance_m)
        elif self.follow_controller and hasattr(self.follow_controller, "target_distance_m"):
            self.target_following_distance_m = float(self.follow_controller.target_distance_m)
        else:
            self.target_following_distance_m = 2.5

        self.running = False
        self.last_update_time = time.time()
        self.target_fps = 20.0
        self.fixed_dt = 1.0 / self.target_fps

        # Reacquisition and Search State
        self.search_active = False
        self.arena_was_in_obstacle_stop = False

        # Stage 6B Autonomous Obstacle Navigation State
        self.navigation_mode = NavigationMode.DIRECT_FOLLOW
        self.avoidance_state = AvoidanceState.CLEAR
        self.detour_direction = DetourDirection.NONE
        self.planned_path: List[Tuple[float, float]] = []
        self.current_waypoint_idx = 0
        self.blocking_obstacle_id: Optional[str] = None
        self.auto_avoidance_enabled = True

        # Stage 6B 360-Degree Target Search State
        self.target_search_enabled = True
        self.search_angle_accumulated = 0.0
        self.search_rotation_speed_deg = 45.0
        self.search_direction = "right"
        self.target_reacquisition_status = ReacquisitionStatus.ACQUIRED

        # Initial Scene State
        self.robot = ArenaRobot(x=2.5, y=4.0, heading_deg=0.0)
        self.robot_radius = 0.35  # meters (bounding footprint radius)
        self.people: Dict[int, ArenaPerson] = {
            1: ArenaPerson(person_id=1, x=5.0, y=4.0, speed=1.0, is_target=True)
        }
        self.selected_target_id: Optional[int] = 1
        self.obstacles: Dict[str, ArenaObstacle] = {
            "obs_1": ArenaObstacle(obs_id="obs_1", x=4.0, y=2.0, radius=0.4, label="Crate A"),
            "obs_2": ArenaObstacle(obs_id="obs_2", x=7.0, y=5.5, radius=0.45, label="Barrel B")
        }

        # Simulated sensor settings
        self.simulate_sensor_fail = False
        self.simulated_distance_cm: Optional[float] = 250.0
        self.simulated_obstacle_active = False

        # Metrics for dashboard telemetry
        self.last_decision = "STOP"
        self.last_distance_m = 2.5
        self.last_bearing_deg = 0.0

    @property
    def safe_zone_min_m(self) -> float:
        return max(0.7, self.target_following_distance_m - self.distance_tolerance_m)

    @property
    def safe_zone_max_m(self) -> float:
        return self.target_following_distance_m + self.distance_tolerance_m

    def set_target_following_distance(self, distance_m: float) -> bool:
        """
        Updates the target following distance setpoint [min_following_distance_m, max_following_distance_m].
        Propagates setpoint to attached FollowController.
        Preserves locked target ID, does not disrupt active detours, does not reset simulation.
        """
        try:
            dist = float(distance_m)
        except (TypeError, ValueError):
            return False
        if not (self.min_following_distance_m <= dist <= self.max_following_distance_m):
            return False
        self.target_following_distance_m = dist
        if self.follow_controller and hasattr(self.follow_controller, "set_target_distance"):
            self.follow_controller.set_target_distance(dist)
        return True

    def can_resume_movement(self) -> bool:
        """Determines if the robot can safely resume autonomous motion."""
        if self.follow_controller and self.follow_controller.emergency_stop_latched:
            return False
        if not self._is_comm_connected():
            return False
        if self.simulate_sensor_fail or (self.follow_controller and (self.follow_controller.ultrasonic_stale or not self.follow_controller.ultrasonic_valid)):
            return False
        if self.avoidance_state == AvoidanceState.BLOCKED:
            return False
        if self.simulated_obstacle_active or (self.follow_controller and self.follow_controller.obstacle_detected):
            return False
        return True

    def _is_comm_connected(self) -> bool:
        """Helper to safely query ESP32 comm connection status whether it's a property, method or transport."""
        if not self.esp32_comm:
            return True
        conn = getattr(self.esp32_comm, "is_connected", True)
        if callable(conn):
            try:
                return bool(conn())
            except Exception:
                return False
        return bool(conn)

    def reset(self):
        """Resets the simulation arena to a reproducible initial state."""
        self.robot = ArenaRobot(x=2.5, y=4.0, heading_deg=0.0)
        self.people = {
            1: ArenaPerson(person_id=1, x=5.0, y=4.0, speed=1.0, is_target=True)
        }
        self.selected_target_id = 1
        self.obstacles = {
            "obs_1": ArenaObstacle(obs_id="obs_1", x=4.0, y=2.0, radius=0.4, label="Crate A"),
            "obs_2": ArenaObstacle(obs_id="obs_2", x=7.0, y=5.5, radius=0.45, label="Barrel B")
        }
        self.simulate_sensor_fail = False
        self.simulated_distance_cm = 250.0
        self.simulated_obstacle_active = False
        self.search_active = False
        self.arena_was_in_obstacle_stop = False
        self.navigation_mode = NavigationMode.DIRECT_FOLLOW
        self.avoidance_state = AvoidanceState.CLEAR
        self.detour_direction = DetourDirection.NONE
        self.planned_path = []
        self.current_waypoint_idx = 0
        self.blocking_obstacle_id = None
        self.auto_avoidance_enabled = True
        self.target_search_enabled = True
        self.search_angle_accumulated = 0.0
        self.search_rotation_speed_deg = 45.0
        self.search_direction = "right"
        self.target_reacquisition_status = ReacquisitionStatus.ACQUIRED
        self.last_decision = "STOP"
        self.last_distance_m = 2.5
        self.last_bearing_deg = 0.0

        if self.follow_controller:
            self.follow_controller.reset_emergency_stop()
            self.follow_controller.set_mode(FollowMode.PERSON)
            self.follow_controller.select_person_target(1)
            self.follow_controller.set_enabled(True)
            self.follow_controller.update_sensor_telemetry(distance_cm=250.0, obstacle=False, is_valid=True, is_stale=False)

        if self.esp32_comm and isinstance(self.esp32_comm.transport, MockSerialTransport):
            self.esp32_comm.transport.simulate_disconnect = False
            self.esp32_comm.send_stop(reason="Simulation reset")

    def move_person(self, person_id: int, dx: float, dy: float):
        if person_id in self.people:
            self.people[person_id].move(dx, dy, self.arena_width, self.arena_height)

    def add_person(self, person_id: Optional[int] = None, x: float = 5.0, y: float = 4.0) -> int:
        pid = person_id or (max(self.people.keys(), default=0) + 1)
        self.people[pid] = ArenaPerson(person_id=pid, x=x, y=y, speed=1.0, is_target=(pid == self.selected_target_id))
        return pid

    def remove_person(self, person_id: int):
        if person_id in self.people:
            del self.people[person_id]
            if self.selected_target_id == person_id:
                self.selected_target_id = None

    def add_obstacle(self, obs_id: str, x: float, y: float, radius: float = 0.4):
        self.obstacles[obs_id] = ArenaObstacle(obs_id, x, y, radius)

    def remove_obstacle(self, obs_id: str):
        if obs_id in self.obstacles:
            del self.obstacles[obs_id]
        if self.blocking_obstacle_id == obs_id or not self.obstacles:
            self.blocking_obstacle_id = None
            self.avoidance_state = AvoidanceState.CLEAR
            self.detour_direction = DetourDirection.NONE
            self.planned_path = []

    def clear_obstacles(self):
        """Removes all obstacles from the arena and clears obstacle flags."""
        self.obstacles.clear()
        self.simulated_obstacle_active = False
        self.simulated_distance_cm = 400.0
        self.avoidance_state = AvoidanceState.CLEAR
        self.detour_direction = DetourDirection.NONE
        self.planned_path = []
        if self.follow_controller:
            self.follow_controller.simulated_obstacle = False
            self.follow_controller.sensor_obstacle = False
            self.follow_controller.update_sensor_telemetry(
                distance_cm=400.0,
                obstacle=False,
                is_valid=True,
                is_stale=False
            )
        if self.esp32_comm and isinstance(self.esp32_comm.transport, MockSerialTransport):
            self.esp32_comm.transport.queue_telemetry(
                distance_cm=400.0,
                obstacle=False
            )

    def reposition_obstacle(self, obs_id: str, x: float, y: float, radius: Optional[float] = None):
        """Repositions an existing obstacle or creates one at the specified coordinates."""
        if obs_id in self.obstacles:
            self.obstacles[obs_id].x = x
            self.obstacles[obs_id].y = y
            if radius is not None:
                self.obstacles[obs_id].radius = radius
        else:
            self.obstacles[obs_id] = ArenaObstacle(obs_id, x, y, radius or 0.4)

    def reposition_person_ahead(self, distance_m: float = 2.5) -> Tuple[float, float]:
        """Repositions the locked target person directly in front of WALL-E along its current heading."""
        if self.selected_target_id is None or self.selected_target_id not in self.people:
            self.selected_target_id = 1
            if 1 not in self.people:
                self.add_person(person_id=1, x=self.robot.x + 2.0, y=self.robot.y)

        rad = math.radians(self.robot.heading_deg)
        target_x = self.robot.x + distance_m * math.cos(rad)
        target_y = self.robot.y + distance_m * math.sin(rad)

        target_x = max(0.5, min(self.arena_width - 0.5, target_x))
        target_y = max(0.5, min(self.arena_height - 0.5, target_y))

        p = self.people[self.selected_target_id]
        p.x = target_x
        p.y = target_y
        p.is_target = True

        self.search_active = False
        self.navigation_mode = NavigationMode.DIRECT_FOLLOW
        self.target_reacquisition_status = ReacquisitionStatus.REACQUIRED
        return target_x, target_y

    def reacquire_target(self) -> Dict[str, Any]:
        """
        Activates controlled safe search rotation in place toward the locked target.
        Safety constraints:
        - Search is strictly for the existing locked target ID.
        - Zero forward movement (rotation in place only).
        - Stops immediately when the target enters camera FOV (|bearing| <= 15 deg).
        - Overridden immediately by obstacle, stale sensor, E-STOP, or ESP32 disconnect.
        - Must not cancel active obstacle avoidance detours.
        - Must not bypass obstacle recovery HOLD.
        """
        if self.selected_target_id is None or self.selected_target_id not in self.people:
            self.search_active = False
            self.target_reacquisition_status = ReacquisitionStatus.NOT_FOUND
            return {"ok": False, "reason": "TARGET LOST"}

        # Active obstacle detour protection
        if self.navigation_mode == NavigationMode.OBSTACLE_AVOIDANCE and self.avoidance_state in (AvoidanceState.PLANNING, AvoidanceState.NAVIGATING):
            return {"ok": False, "reason": "OBSTACLE DETOUR ACTIVE"}

        # Recovery HOLD invariant
        if self.arena_was_in_obstacle_stop or (self.follow_controller and getattr(self.follow_controller, "was_in_obstacle_stop", False)):
            return {"ok": False, "reason": "OBSTACLE RECOVERY HOLD"}

        # Safety overrides
        if self.follow_controller:
            if self.follow_controller.emergency_stop_latched:
                self.search_active = False
                return {"ok": False, "reason": "E-STOP"}
            if self.follow_controller.obstacle_detected or self.simulated_obstacle_active:
                self.search_active = False
                return {"ok": False, "reason": "OBSTACLE"}
            if self.simulate_sensor_fail or self.follow_controller.ultrasonic_stale:
                self.search_active = False
                return {"ok": False, "reason": "SENSOR STALE"}

        if not self._is_comm_connected():
            self.search_active = False
            return {"ok": False, "reason": "COMMUNICATION FAILURE"}

        # If target already visible and aligned in camera cone, no search is required
        target_p = self.people.get(self.selected_target_id)
        if target_p:
            dist_t, bear_t = self.compute_relative_polar(target_p.x, target_p.y)
            fov_half = self.camera_fov_deg / 2.0
            if dist_t <= self.camera_range_m and abs(bear_t) <= 15.0:
                self.search_active = False
                self.navigation_mode = NavigationMode.DIRECT_FOLLOW
                self.target_reacquisition_status = ReacquisitionStatus.ACQUIRED
                self.search_angle_accumulated = 0.0
                return {"ok": True, "target_id": self.selected_target_id, "search_active": False, "status": "ALREADY_ACQUIRED"}

        self.search_active = True
        self.navigation_mode = NavigationMode.TARGET_SEARCH
        self.target_reacquisition_status = ReacquisitionStatus.SEARCHING
        self.search_angle_accumulated = 0.0
        if target_p:
            _, bear_t = self.compute_relative_polar(target_p.x, target_p.y)
            self.search_direction = "left" if bear_t < 0.0 else "right"
        if self.follow_controller:
            self.follow_controller.reacquire_target()
        return {"ok": True, "target_id": self.selected_target_id, "search_active": True}

    def start_360_search(self) -> Dict[str, Any]:
        """
        Initiates 360-degree in-place rotation scan for the locked target.
        """
        if self.selected_target_id is None or self.selected_target_id not in self.people:
            self.target_reacquisition_status = ReacquisitionStatus.NOT_FOUND
            return {"ok": False, "reason": "No target locked to search for"}

        # Active obstacle detour protection
        if self.navigation_mode == NavigationMode.OBSTACLE_AVOIDANCE and self.avoidance_state in (AvoidanceState.PLANNING, AvoidanceState.NAVIGATING):
            return {"ok": False, "reason": "OBSTACLE DETOUR ACTIVE"}

        # Recovery HOLD invariant
        if self.arena_was_in_obstacle_stop or (self.follow_controller and getattr(self.follow_controller, "was_in_obstacle_stop", False)):
            return {"ok": False, "reason": "OBSTACLE RECOVERY HOLD"}

        # Safety overrides
        if self.follow_controller and self.follow_controller.emergency_stop_latched:
            return {"ok": False, "reason": "Emergency STOP latched"}
        if not self._is_comm_connected():
            return {"ok": False, "reason": "ESP32 communication failure"}
        if self.simulate_sensor_fail or (self.follow_controller and self.follow_controller.ultrasonic_stale):
            return {"ok": False, "reason": "Sensor telemetry stale/invalid"}
        if self.simulated_obstacle_active or (self.follow_controller and self.follow_controller.obstacle_detected):
            return {"ok": False, "reason": "Obstacle in perimeter"}

        self.navigation_mode = NavigationMode.TARGET_SEARCH
        self.target_reacquisition_status = ReacquisitionStatus.SEARCHING
        self.search_angle_accumulated = 0.0
        self.search_direction = "right"
        self.search_active = True
        return {"ok": True, "target_id": self.selected_target_id, "mode": self.navigation_mode}

    def cancel_target_search(self) -> Dict[str, Any]:
        """Cancels an active 360-degree target search, returning the robot to stopped status."""
        self.search_active = False
        self.navigation_mode = NavigationMode.DIRECT_FOLLOW
        self.target_reacquisition_status = ReacquisitionStatus.NOT_FOUND
        self.robot.apply_command("stop", 0)
        return {"ok": True, "reacquisition_status": self.target_reacquisition_status}

    def find_blocking_obstacle(self) -> Optional[ArenaObstacle]:
        """Finds the closest obstacle directly blocking the robot along its heading within the ultrasonic cone."""
        closest_obs = None
        min_dist = 999.0
        for obs in self.obstacles.values():
            dist, bearing = self.compute_relative_polar(obs.x, obs.y)
            if abs(bearing) <= 35.0:
                dist_surface = dist - obs.radius - 0.25
                if dist_surface <= 0.65 and dist < min_dist:
                    min_dist = dist
                    closest_obs = obs
        return closest_obs

    def check_segment_clearance(self, x1: float, y1: float, x2: float, y2: float, ignore_obs_id: Optional[str] = None, margin: float = 0.15) -> bool:
        """
        Validates swept-path clearance for the robot's rectangular footprint along segment (x1, y1)-(x2, y2).
        Checks distance to all arena obstacles against (obs.radius + robot_radius + margin).
        Also validates that the segment stays within arena boundary margins.
        """
        required_buffer = getattr(self, "robot_radius", 0.35) + margin
        if not (required_buffer <= x1 <= self.arena_width - required_buffer and
                required_buffer <= y1 <= self.arena_height - required_buffer and
                required_buffer <= x2 <= self.arena_width - required_buffer and
                required_buffer <= y2 <= self.arena_height - required_buffer):
            return False

        for obs in self.obstacles.values():
            if ignore_obs_id and obs.id == ignore_obs_id:
                continue
            min_dist = point_to_segment_distance(obs.x, obs.y, x1, y1, x2, y2)
            if min_dist < (obs.radius + required_buffer):
                return False

        return True

    def plan_detour(self) -> bool:
        """
        Uses known 2D obstacle geometry to evaluate left and right clearance and plan a collision-free detour path.
        Validates robot footprint swept-path clearance against all obstacles.
        Returns True if a safe detour was planned, or False if both directions are blocked.
        """
        blocking_obs = self.find_blocking_obstacle()
        if not blocking_obs:
            self.avoidance_state = AvoidanceState.CLEAR
            self.detour_direction = DetourDirection.NONE
            self.planned_path = []
            return True

        self.blocking_obstacle_id = blocking_obs.id
        self.avoidance_state = AvoidanceState.PLANNING

        xr, yr = self.robot.x, self.robot.y
        rad_h = math.radians(self.robot.heading_deg)
        ux = math.cos(rad_h)
        uy = math.sin(rad_h)

        # Normal unit vectors (left and right)
        n_left = (-uy, ux)
        n_right = (uy, -ux)

        target_p = self.people.get(self.selected_target_id)
        tx = target_p.x if target_p else xr + 2.5 * ux
        ty = target_p.y if target_p else yr + 2.5 * uy

        r_obs = blocking_obs.radius
        robot_rad = getattr(self, "robot_radius", 0.35)
        clearance_offset = r_obs + robot_rad + 0.40  # Rover clearance + buffer

        candidate_sides = [("LEFT", n_left), ("RIGHT", n_right)]
        feasible_routes = []

        for side_name, (nx, ny) in candidate_sides:
            # Waypoint 1: lateral flank clearance point (angled outward/rearward to clear the shoulder)
            w1_x = blocking_obs.x - 0.25 * ux + clearance_offset * nx
            w1_y = blocking_obs.y - 0.25 * uy + clearance_offset * ny

            # Waypoint 2: past the obstacle along forward heading
            w2_x = blocking_obs.x + (r_obs + 0.85) * ux + clearance_offset * nx
            w2_y = blocking_obs.y + (r_obs + 0.85) * uy + clearance_offset * ny

            # 1. Check segment from robot (xr, yr) to W1 (ignoring blocking_obs which we check separately)
            if not self.check_segment_clearance(xr, yr, w1_x, w1_y, ignore_obs_id=blocking_obs.id, margin=0.10):
                continue

            # 2. Check segment W1 to W2 (ignoring blocking_obs which we check separately)
            if not self.check_segment_clearance(w1_x, w1_y, w2_x, w2_y, ignore_obs_id=blocking_obs.id, margin=0.10):
                continue

            # 3. Clearance from blocking_obs along segments
            # Segment robot -> W1 must maintain clearance from blocking obstacle
            if point_to_segment_distance(blocking_obs.x, blocking_obs.y, xr, yr, w1_x, w1_y) < (blocking_obs.radius + 0.20):
                continue
            # Segment W1 -> W2 must clear blocking obstacle with footprint margin
            if point_to_segment_distance(blocking_obs.x, blocking_obs.y, w1_x, w1_y, w2_x, w2_y) < (blocking_obs.radius + robot_rad + 0.05):
                continue

            # Compute total path distance to target
            dist_to_w1 = math.hypot(w1_x - xr, w1_y - yr)
            dist_w1_w2 = math.hypot(w2_x - w1_x, w2_y - w1_y)
            dist_w2_target = math.hypot(tx - w2_x, ty - w2_y)
            total_dist = dist_to_w1 + dist_w1_w2 + dist_w2_target
            feasible_routes.append((side_name, [(w1_x, w1_y), (w2_x, w2_y)], total_dist))

        if not feasible_routes:
            # Both directions blocked!
            self.avoidance_state = AvoidanceState.BLOCKED
            self.detour_direction = DetourDirection.NONE
            self.planned_path = []
            return False

        # Apply directional hysteresis to prevent left/right oscillation
        def route_eval_cost(route):
            side, path, dist = route
            bias = -0.5 if (self.detour_direction != DetourDirection.NONE and side == self.detour_direction) else 0.0
            return dist + bias

        feasible_routes.sort(key=route_eval_cost)
        best_side, best_path, _ = feasible_routes[0]

        self.detour_direction = best_side
        self.planned_path = best_path
        self.current_waypoint_idx = 0
        self.avoidance_state = AvoidanceState.NAVIGATING
        self.navigation_mode = NavigationMode.OBSTACLE_AVOIDANCE
        return True

    def get_stop_reason(self) -> str:
        """
        Determines the current reason for robot stop or non-forward state:
        E-STOP, COMMUNICATION FAILURE, SENSOR STALE, OBSTACLE, OBSTACLE RECOVERY HOLD,
        REACQUIRING TARGET, TARGET OUT OF VIEW, TARGET LOST, TARGET NOT FOUND,
        HOLDING DISTANCE, FOLLOW DISABLED, or NONE.
        Invariant: If the robot is stopped (command == "stop" or linear_speed == 0),
        the stop reason must NEVER be "NONE".
        """
        # Level 1: E-STOP
        if self.follow_controller and self.follow_controller.emergency_stop_latched:
            return "E-STOP"

        # Level 2: Communication Failure
        if not self._is_comm_connected():
            return "COMMUNICATION FAILURE"

        # Level 3: Sensor Stale / Invalid
        if self.simulate_sensor_fail:
            return "SENSOR STALE"
        if self.follow_controller and (self.follow_controller.ultrasonic_stale or not self.follow_controller.ultrasonic_valid):
            return "SENSOR STALE"

        # Level 4: Obstacle detection or blocked detour
        if self.avoidance_state == AvoidanceState.BLOCKED:
            return "OBSTACLE"
        if (self.simulated_obstacle_active or (self.follow_controller and self.follow_controller.obstacle_detected)) and self.navigation_mode != NavigationMode.OBSTACLE_AVOIDANCE:
            return "OBSTACLE"

        # Active Detour Navigation
        if self.navigation_mode == NavigationMode.OBSTACLE_AVOIDANCE and self.avoidance_state == AvoidanceState.NAVIGATING:
            if getattr(self.robot, "linear_speed", 0.0) == 0.0 and getattr(self.robot, "current_cmd", "stop") == "stop":
                return "DETOUR IN PROGRESS"
            return "NONE"

        # Level 6: Target Search
        if self.navigation_mode == NavigationMode.TARGET_SEARCH:
            if self.target_reacquisition_status == ReacquisitionStatus.NOT_FOUND:
                return "TARGET NOT FOUND"
            return "TARGET SEARCH"

        if self.target_reacquisition_status == ReacquisitionStatus.NOT_FOUND:
            return "TARGET NOT FOUND"

        # Level 5 Detour Post-Completion / Reacquiring
        if self.avoidance_state == AvoidanceState.REACQUIRING:
            return "REACQUIRING TARGET"

        # Level 7/8: Target Missing or Lost
        if self.selected_target_id is None or self.selected_target_id not in self.people:
            return "TARGET LOST"

        target_person = self.people.get(self.selected_target_id)
        if target_person:
            dist, bearing = self.compute_relative_polar(target_person.x, target_person.y)
            fov_half = self.camera_fov_deg / 2.0
            if (abs(bearing) > fov_half or dist > self.camera_range_m):
                return "TARGET OUT OF VIEW"

        # Follow Controller Disabled
        if self.follow_controller and not self.follow_controller.enabled:
            return "FOLLOW DISABLED"

        # First-Recovery Cycle HOLD
        if self.arena_was_in_obstacle_stop or (self.follow_controller and getattr(self.follow_controller, "was_in_obstacle_stop", False)):
            return "OBSTACLE RECOVERY HOLD"

        # Safe following zone HOLD or stopped states
        if getattr(self.robot, "current_cmd", "stop") == "stop" or self.last_decision in ("HOLD", "STOP") or getattr(self.robot, "linear_speed", 0.0) == 0.0:
            if target_person:
                dist, _ = self.compute_relative_polar(target_person.x, target_person.y)
                if dist <= self.safe_zone_max_m:
                    return "HOLDING DISTANCE"
            if self.last_decision == "HOLD":
                return "HOLDING DISTANCE"
            return "STOP"

        return "NONE"

    def get_stop_code(self) -> str:
        """Returns structured uppercase snake_case stop code for diagnostics and API."""
        reason = self.get_stop_reason()
        mapping = {
            "E-STOP": "EMERGENCY_STOP",
            "COMMUNICATION FAILURE": "COMMUNICATION_FAILURE",
            "SENSOR STALE": "SENSOR_DATA_STALE",
            "OBSTACLE": "BOTH_ROUTES_BLOCKED" if self.avoidance_state == AvoidanceState.BLOCKED else "OBSTACLE_BLOCKING_PATH",
            "DETOUR IN PROGRESS": "DETOUR_IN_PROGRESS",
            "TARGET SEARCH": "TARGET_SEARCH",
            "TARGET NOT FOUND": "TARGET_NOT_FOUND",
            "REACQUIRING TARGET": "REACQUIRING_TARGET",
            "TARGET LOST": "TARGET_LOST",
            "TARGET OUT OF VIEW": "TARGET_OUT_OF_VIEW",
            "FOLLOW DISABLED": "FOLLOWING_DISABLED",
            "OBSTACLE RECOVERY HOLD": "OBSTACLE_RECOVERY_HOLD",
            "HOLDING DISTANCE": "TARGET_DISTANCE_REACHED",
            "STOP": "STOP",
            "NONE": "NONE"
        }
        return mapping.get(reason, reason.replace(" ", "_").upper())

    def compute_relative_polar(self, target_x: float, target_y: float) -> Tuple[float, float]:
        """Computes distance in meters and bearing in degrees relative to robot heading."""
        dx = target_x - self.robot.x
        dy = target_y - self.robot.y
        dist = math.hypot(dx, dy)
        world_angle_deg = math.degrees(math.atan2(dy, dx))
        # Relative bearing: 0 = straight ahead, negative = left, positive = right
        bearing_deg = (world_angle_deg - self.robot.heading_deg + 180.0) % 360.0 - 180.0
        return dist, bearing_deg

    def step(self, dt: Optional[float] = None) -> Dict[str, Any]:
        """
        Executes one fixed simulation step:
        1. Checks ultrasonic obstacle proximity along robot heading
        2. Generates vision PersonObservations for people in robot FOV
        3. Feeds observations into FollowController
        4. Evaluates controller decision & passes through safety guards
        5. Sends command to ESP32 mock transport
        6. Updates robot kinematics in 2D arena
        7. Returns complete state telemetry
        """
        dt_val = dt if dt is not None else self.fixed_dt

        # --- 1. SENSOR SIMULATION (HC-SR04 & MPU6050) ---
        # Check obstacle collision & line-of-sight ultrasonic raycast
        min_obstacle_dist = 999.0
        for obs in self.obstacles.values():
            dist, bearing = self.compute_relative_polar(obs.x, obs.y)
            # In front of ultrasonic cone (+- 25 degrees)
            if abs(bearing) <= 25.0:
                dist_surface = max(0.01, dist - obs.radius - 0.25)
                if dist_surface < min_obstacle_dist:
                    min_obstacle_dist = dist_surface

        # Check proximity to walls
        rad = math.radians(self.robot.heading_deg)
        cos_h, sin_h = math.cos(rad), math.sin(rad)
        if cos_h > 0.1:
            min_obstacle_dist = min(min_obstacle_dist, (self.arena_width - self.robot.x) / cos_h)
        elif cos_h < -0.1:
            min_obstacle_dist = min(min_obstacle_dist, (0.0 - self.robot.x) / cos_h)
        if sin_h > 0.1:
            min_obstacle_dist = min(min_obstacle_dist, (self.arena_height - self.robot.y) / sin_h)
        elif sin_h < -0.1:
            min_obstacle_dist = min(min_obstacle_dist, (0.0 - self.robot.y) / sin_h)

        distance_cm = round(min_obstacle_dist * 100.0, 1) if min_obstacle_dist < 6.0 else 400.0
        obstacle_flag = (distance_cm <= 30.0)

        # Track obstacle stop state for first-recovery-cycle HOLD invariant
        had_obstacle_stop = self.arena_was_in_obstacle_stop
        if obstacle_flag:
            self.arena_was_in_obstacle_stop = True

        if self.simulate_sensor_fail:
            is_valid = False
            is_stale = True
            distance_cm = None
            obstacle_flag = False
        else:
            is_valid = True
            is_stale = False

        self.simulated_distance_cm = distance_cm
        self.simulated_obstacle_active = obstacle_flag

        # Feed sensor reading to FollowController and Mock ESP32 telemetry queue
        if self.follow_controller:
            self.follow_controller.update_sensor_telemetry(
                distance_cm=distance_cm,
                obstacle=obstacle_flag,
                is_valid=is_valid,
                is_stale=is_stale
            )

        if self.esp32_comm and isinstance(self.esp32_comm.transport, MockSerialTransport):
            self.esp32_comm.transport.queue_telemetry(
                distance_cm=distance_cm,
                obstacle=obstacle_flag,
                imu={
                    "ax": round(self.robot.linear_speed * 0.1, 2),
                    "ay": 0.0,
                    "az": 9.81,
                    "gx": 0.0,
                    "gy": 0.0,
                    "gz": round(self.robot.angular_speed, 2)
                }
            )
            try:
                self.esp32_comm.read_telemetry(timeout_sec=0.01)
            except Exception:
                pass

        # --- 2. VISION OBSERVATION GENERATION ---
        # Generate simulated camera observations for people in FOV
        observations: List[PersonObservation] = []
        frame_w = 640
        fov_half = self.camera_fov_deg / 2.0

        target_dist = 0.0
        target_bearing = 0.0

        for pid, p in self.people.items():
            dist, bearing = self.compute_relative_polar(p.x, p.y)

            if pid == self.selected_target_id:
                target_dist = dist
                target_bearing = bearing

            # Visible in camera cone
            if abs(bearing) <= fov_half and dist <= self.camera_range_m:
                # Map bearing [-fov_half, +fov_half] to pixel X [0, frame_w]
                norm_x = (bearing + fov_half) / self.camera_fov_deg
                center_x = int(norm_x * frame_w)

                if center_x < frame_w * 0.35:
                    dir_str = "LEFT"
                elif center_x > frame_w * 0.65:
                    dir_str = "RIGHT"
                else:
                    dir_str = "CENTER"

                # Simulated bounding box size inversely proportional to distance
                box_h = int(max(60, min(420, 280 / max(0.5, dist))))
                box_w = int(box_h * 0.45)
                top_y = max(10, 240 - box_h // 2)
                left_x = max(10, center_x - box_w // 2)

                obs = PersonObservation(
                    person_id=pid,
                    bbox=(left_x, top_y, box_w, box_h),
                    confidence=0.92,
                    direction=dir_str,
                    center=(center_x, top_y + box_h // 2)
                )
                observations.append(obs)

        self.last_distance_m = target_dist
        self.last_bearing_deg = target_bearing

        # --- 3. FEED CONTROLLER & DISPATCH DECISION ---
        decision = "STOP"
        dispatched_cmd = "stop"
        dispatched_speed = 0

        # Check safety overrides FIRST: E-STOP, Comm Failure, Stale/Invalid Sensor
        is_estop = bool(self.follow_controller and self.follow_controller.emergency_stop_latched)
        is_comm_fail = not self._is_comm_connected()
        is_sensor_fail = bool(self.simulate_sensor_fail or (self.follow_controller and (self.follow_controller.ultrasonic_stale or not self.follow_controller.ultrasonic_valid)))

        if is_estop or is_comm_fail or is_sensor_fail:
            self.search_active = False
            if self.navigation_mode == NavigationMode.TARGET_SEARCH:
                self.navigation_mode = NavigationMode.DIRECT_FOLLOW
            if self.navigation_mode == NavigationMode.OBSTACLE_AVOIDANCE:
                self.avoidance_state = AvoidanceState.BLOCKED
            dispatched_cmd = "stop"
            dispatched_speed = 0
            decision = "STOP"
            self.last_decision = decision
            if self.esp32_comm:
                self.esp32_comm.send_command("stop", 0)
            self.robot.apply_command("stop", 0)
            self.robot.update_physics(dt_val, self.arena_width, self.arena_height)
            return self.get_state()

        if self.follow_controller:
            # Sync target lock
            if self.selected_target_id is not None:
                self.follow_controller.select_person_target(self.selected_target_id)

            # Update tracker and evaluate
            self.follow_controller.person_tracker.update(observations, frame_width=frame_w)
            eval_res = self.follow_controller.evaluate()
            decision = str(eval_res.get("decision", "STOP"))

            # Obtain dispatched command
            dispatched_cmd = eval_res.get("direction", "stop")
            dispatched_speed = eval_res.get("speed", 0)

            # Check safety overrides for search
            if self.search_active:
                if (self.follow_controller and (self.follow_controller.emergency_stop_latched or
                                                self.follow_controller.obstacle_detected)) or \
                   self.simulated_obstacle_active or self.simulate_sensor_fail or \
                   not self._is_comm_connected():
                    self.search_active = False

            # First-recovery cycle invariant: if controller or arena recovering from obstacle stop, honor HOLD
            is_obstacle_recovery_hold = (
                (had_obstacle_stop and not obstacle_flag) or
                (eval_res.get("decision") == "HOLD" and "Obstacle cleared" in str(eval_res.get("reason", "")))
            )
            if not obstacle_flag and not self.simulated_obstacle_active:
                self.arena_was_in_obstacle_stop = False

            # --- FEATURE 1: AUTONOMOUS OBSTACLE AVOIDANCE BRANCH ---
            if obstacle_flag:
                if self.navigation_mode != NavigationMode.OBSTACLE_AVOIDANCE:
                    # 1. Detect and stop before the obstacle
                    dispatched_cmd = "stop"
                    dispatched_speed = 0
                    decision = "STOP"
                    # 2 & 3. Plan collision-free detour using arena obstacle geometry
                    if self.auto_avoidance_enabled:
                        planned = self.plan_detour()
                        if not planned:
                            self.avoidance_state = AvoidanceState.BLOCKED
                            dispatched_cmd = "stop"
                            dispatched_speed = 0
                            decision = "STOP"
                else:
                    # Already navigating detour: check if path is blocked very close (< 0.18m)
                    if min_obstacle_dist < 0.18:
                        if not self.plan_detour():
                            self.avoidance_state = AvoidanceState.BLOCKED
                            dispatched_cmd = "stop"
                            dispatched_speed = 0
                            decision = "STOP"
            else:
                if self.avoidance_state == AvoidanceState.BLOCKED:
                    self.avoidance_state = AvoidanceState.CLEAR
                    self.detour_direction = DetourDirection.NONE
                    self.planned_path = []

            if self.navigation_mode == NavigationMode.OBSTACLE_AVOIDANCE and self.planned_path:
                # 4. Rotate and navigate around the obstacle via waypoints
                wp_idx = self.current_waypoint_idx
                if wp_idx < len(self.planned_path):
                    wp_x, wp_y = self.planned_path[wp_idx]
                    dx = wp_x - self.robot.x
                    dy = wp_y - self.robot.y
                    dist_wp = math.hypot(dx, dy)
                    wp_angle_deg = math.degrees(math.atan2(dy, dx))
                    wp_bearing = (wp_angle_deg - self.robot.heading_deg + 180.0) % 360.0 - 180.0

                    if dist_wp < 0.35:
                        # Waypoint reached
                        self.current_waypoint_idx += 1
                        if self.current_waypoint_idx >= len(self.planned_path):
                            # Detour finished! Transition to REACQUIRING target
                            self.avoidance_state = AvoidanceState.REACQUIRING
                            self.navigation_mode = NavigationMode.DIRECT_FOLLOW
                            self.planned_path = []
                            self.detour_direction = DetourDirection.NONE
                            self.search_active = False
                            self.search_angle_accumulated = 0.0
                            dispatched_cmd = "stop"
                            dispatched_speed = 0
                            decision = "HOLD"
                        else:
                            # Advance to next waypoint
                            wp_x, wp_y = self.planned_path[self.current_waypoint_idx]
                            dx = wp_x - self.robot.x
                            dy = wp_y - self.robot.y
                            dist_wp = math.hypot(dx, dy)
                            wp_angle_deg = math.degrees(math.atan2(dy, dx))
                            wp_bearing = (wp_angle_deg - self.robot.heading_deg + 180.0) % 360.0 - 180.0

                    if self.navigation_mode == NavigationMode.OBSTACLE_AVOIDANCE:
                        # Re-verify swept clearance to wp_x, wp_y
                        if not self.check_segment_clearance(self.robot.x, self.robot.y, wp_x, wp_y, ignore_obs_id=self.blocking_obstacle_id, margin=0.05):
                            # Path to waypoint is blocked, attempt replan
                            if not self.plan_detour():
                                self.avoidance_state = AvoidanceState.BLOCKED
                                dispatched_cmd = "stop"
                                dispatched_speed = 0
                                decision = "STOP"
                        if self.avoidance_state != AvoidanceState.BLOCKED:
                            if abs(wp_bearing) > 15.0:
                                if wp_bearing < 0.0:
                                    dispatched_cmd = "left"
                                    dispatched_speed = 140
                                    decision = "TURN_LEFT"
                                else:
                                    dispatched_cmd = "right"
                                    dispatched_speed = 140
                                    decision = "TURN_RIGHT"
                            else:
                                dispatched_cmd = "forward"
                                dispatched_speed = 160
                                decision = "MOVE_FORWARD"

            # --- FEATURE 2: 360-DEGREE TARGET SEARCH BRANCH ---
            elif self.navigation_mode == NavigationMode.TARGET_SEARCH:
                target_p = self.people.get(self.selected_target_id)
                if not target_p:
                    self.target_reacquisition_status = ReacquisitionStatus.NOT_FOUND
                    self.navigation_mode = NavigationMode.DIRECT_FOLLOW
                    self.search_active = False
                    dispatched_cmd = "stop"
                    dispatched_speed = 0
                    decision = "STOP"
                else:
                    dist_t, bear_t = self.compute_relative_polar(target_p.x, target_p.y)
                    # Check if SAME locked target is visible in camera cone
                    if dist_t <= self.camera_range_m and abs(bear_t) <= fov_half:
                        if abs(bear_t) <= 15.0:
                            # Centered & reacquired!
                            self.target_reacquisition_status = ReacquisitionStatus.REACQUIRED
                            self.navigation_mode = NavigationMode.DIRECT_FOLLOW
                            self.search_active = False
                            self.search_angle_accumulated = 0.0
                            dispatched_cmd = "stop"
                            dispatched_speed = 0
                            decision = "HOLD"
                        else:
                            # Steer toward target in place to center it (linear speed strictly 0.0)
                            if bear_t < 0.0:
                                dispatched_cmd = "left"
                                dispatched_speed = 120
                                decision = "TURN_LEFT"
                            else:
                                dispatched_cmd = "right"
                                dispatched_speed = 120
                                decision = "TURN_RIGHT"
                            self.search_angle_accumulated += self.search_rotation_speed_deg * dt_val
                    else:
                        # Continue rotating in place in search direction (linear speed strictly 0.0)
                        dispatched_cmd = self.search_direction
                        dispatched_speed = 120
                        decision = "TURN_RIGHT" if self.search_direction == "right" else "TURN_LEFT"
                        self.search_angle_accumulated += self.search_rotation_speed_deg * dt_val

                    # Check if full 360-degree rotation completed
                    if self.search_angle_accumulated >= 360.0:
                        self.search_angle_accumulated = 360.0
                        self.target_reacquisition_status = ReacquisitionStatus.NOT_FOUND
                        self.navigation_mode = NavigationMode.DIRECT_FOLLOW
                        self.search_active = False
                        dispatched_cmd = "stop"
                        dispatched_speed = 0
                        decision = "STOP"

            elif is_obstacle_recovery_hold:
                dispatched_cmd = "stop"
                dispatched_speed = 0
                decision = "HOLD"

            elif not self.search_active and self.selected_target_id in self.people and abs(target_bearing) > 90.0:
                # Target is behind WALL-E. Without active search command, halt safely with STOP.
                dispatched_cmd = "stop"
                dispatched_speed = 0
                decision = "STOP"

            else:
                # In Person Following mode (DIRECT_FOLLOW)
                # Ensure state synchronization
                self.search_active = False
                self.search_angle_accumulated = 0.0
                if self.target_reacquisition_status == ReacquisitionStatus.SEARCHING:
                    self.target_reacquisition_status = ReacquisitionStatus.ACQUIRED

                target_in_range = (self.selected_target_id in self.people and
                                   target_dist <= self.camera_range_m and
                                   abs(target_bearing) <= 90.0)

                # Transition REACQUIRING -> CLEAR when target is in view and aligned
                if self.avoidance_state == AvoidanceState.REACQUIRING:
                    if target_in_range and abs(target_bearing) <= 15.0:
                        self.avoidance_state = AvoidanceState.CLEAR

                if (self.follow_controller and
                    self.follow_controller.enabled and
                    not self.follow_controller.emergency_stop_latched and
                    not self.follow_controller.obstacle_detected and
                    not self.simulate_sensor_fail and
                    target_in_range):

                    clear_path, _ = self.follow_controller.is_forward_path_clear()

                    # Align & approach logic for virtual demonstration
                    if target_dist > self.safe_zone_max_m:
                        if target_bearing < -12.0:
                            dispatched_cmd = "left"
                            dispatched_speed = 160
                            decision = "TURN_LEFT"
                        elif target_bearing > 12.0:
                            dispatched_cmd = "right"
                            dispatched_speed = 160
                            decision = "TURN_RIGHT"
                        elif clear_path:
                            dispatched_cmd = "forward"
                            dispatched_speed = 180
                            decision = "MOVE_FORWARD"
                        else:
                            dispatched_cmd = "stop"
                            dispatched_speed = 0
                            decision = "STOP"
                    elif target_dist < self.safe_zone_min_m:
                        dispatched_cmd = "stop"
                        dispatched_speed = 0
                        decision = "HOLD"
                    else:
                        # Inside safe following zone [safe_zone_min_m - safe_zone_max_m]
                        if target_bearing < -15.0:
                            dispatched_cmd = "left"
                            dispatched_speed = 140
                            decision = "TURN_LEFT"
                        elif target_bearing > 15.0:
                            dispatched_cmd = "right"
                            dispatched_speed = 140
                            decision = "TURN_RIGHT"
                        else:
                            dispatched_cmd = "stop"
                            dispatched_speed = 0
                            decision = "HOLD"
                elif self.follow_controller and not self.follow_controller.enabled:
                    dispatched_cmd = "stop"
                    dispatched_speed = 0
                    decision = "STOP"
                elif not target_in_range:
                    dispatched_cmd = "stop"
                    dispatched_speed = 0
                    decision = "STOP"
                elif self.simulate_sensor_fail or not self.follow_controller.is_forward_path_clear()[0]:
                    dispatched_cmd = "stop"
                    dispatched_speed = 0
                    decision = "STOP"

        self.last_decision = decision

        # Dispatch to ESP32 mock transport
        if self.esp32_comm:
            self.esp32_comm.send_command(dispatched_cmd, speed=dispatched_speed)

        # --- 4. ROBOT PHYSICS UPDATE ---
        self.robot.apply_command(dispatched_cmd, dispatched_speed)
        self.robot.update_physics(dt_val, self.arena_width, self.arena_height)

        return self.get_state()

    def get_state(self) -> Dict[str, Any]:
        """Returns comprehensive arena state for UI canvas rendering and live telemetry."""
        target_person = self.people.get(self.selected_target_id) if self.selected_target_id else None
        stop_reason = self.get_stop_reason()
        stop_code = self.get_stop_code()
        
        # Invariant check: if robot is stopped, stop_reason and stop_code must never be NONE
        is_stopped = (getattr(self.robot, "linear_speed", 0.0) == 0.0 and getattr(self.robot, "angular_speed", 0.0) == 0.0) or getattr(self.robot, "current_cmd", "stop") == "stop"
        if is_stopped and stop_reason == "NONE":
            stop_reason = "HOLDING DISTANCE" if self.last_decision == "HOLD" else "STOP"
            stop_code = "TARGET_DISTANCE_REACHED" if stop_reason == "HOLDING DISTANCE" else "STOP"

        following_status = (
            "HOLDING" if self.last_decision == "HOLD"
            else "MOVING" if self.last_decision in ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")
            else "SEARCHING" if self.search_active
            else "DETOURING" if self.avoidance_state in (AvoidanceState.NAVIGATING, AvoidanceState.REACQUIRING)
            else "STOPPED"
        )

        return {
            "arena": {
                "width": self.arena_width,
                "height": self.arena_height,
                "camera_fov_deg": self.camera_fov_deg,
                "camera_range_m": self.camera_range_m,
                "target_following_distance_m": round(self.target_following_distance_m, 2),
                "safe_zone_min_m": round(self.safe_zone_min_m, 2),
                "safe_zone_max_m": round(self.safe_zone_max_m, 2),
                "distance_tolerance_m": round(self.distance_tolerance_m, 2),
                "obstacle_threshold_m": 0.30
            },
            "robot": self.robot.to_dict(),
            "people": [p.to_dict() for p in self.people.values()],
            "selected_target_id": self.selected_target_id,
            "target_locked": (self.selected_target_id is not None),
            "target_distance_m": round(self.last_distance_m, 2),
            "target_bearing_deg": round(self.last_bearing_deg, 1),
            "obstacles": [o.to_dict() for o in self.obstacles.values()],
            "sensors": {
                "distance_cm": self.simulated_distance_cm,
                "obstacle": self.simulated_obstacle_active,
                "simulated_sensor_fail": self.simulate_sensor_fail
            },
            "decision": self.last_decision,
            "stop_reason": stop_reason,
            "stop_code": stop_code,
            "following_status": following_status,
            "diagnostics": {
                "can_resume_movement": self.can_resume_movement(),
                "target_in_camera_view": (self.selected_target_id in self.people and
                                          self.last_distance_m <= self.camera_range_m and
                                          abs(self.last_bearing_deg) <= (self.camera_fov_deg / 2.0)),
                "target_in_range": (self.selected_target_id in self.people and
                                    self.last_distance_m <= self.camera_range_m and
                                    abs(self.last_bearing_deg) <= 90.0),
                "path_clear": self.follow_controller.is_forward_path_clear()[0] if self.follow_controller else False,
                "obstacle_active": self.simulated_obstacle_active,
                "sensor_fail": self.simulate_sensor_fail,
                "controller_enabled": self.follow_controller.enabled if self.follow_controller else False,
                "controller_stopped": self.follow_controller.emergency_stop_latched if self.follow_controller else False,
                "active_stop_reason": stop_reason,
                "active_stop_code": stop_code,
            },
            "search_active": self.search_active,
            "navigation_mode": self.navigation_mode,
            "avoidance_state": self.avoidance_state,
            "detour_direction": self.detour_direction,
            "planned_path": [[round(px, 3), round(py, 3)] for px, py in self.planned_path],
            "target_search_progress_deg": round(self.search_angle_accumulated, 1),
            "target_reacquisition_status": self.target_reacquisition_status,
            "auto_avoidance_enabled": self.auto_avoidance_enabled,
            "running": self.running
        }

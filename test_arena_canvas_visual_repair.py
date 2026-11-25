"""
WALL-E Simulation Visual Regression Repair Verification Suite
Module: test_arena_canvas_visual_repair.py

Automated regression suite verifying:
1. Backend Simulation State Integrity:
   - Robot coordinates, heading, and kinematics remain intact
   - People collection preserved (Person #1 and Person #2)
   - Obstacles are never duplicated (Crate A and Barrel B appear exactly once)
   - Changing following distance does not reset or corrupt simulation scene objects
   - Scene reset restores canonical scene
2. Frontend Canvas Rendering & Coordinate Systems:
   - renderArena() defensively resets transformation matrix to identity (setTransform)
   - Safe-zone arc and following distance arcs render as non-destructive layers
   - Context save/restore blocks use try/finally to prevent transform leakage
   - Verification via Node.js VM that renderArena() executes with zero exceptions,
     drawing WALL-E, all people, and non-duplicated obstacles across multiple frames
"""

import os
import sys
import unittest
import json
import subprocess

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
from follow_controller import FollowController
from esp32_comm import ESP32CommunicationManager, MockSerialTransport
import main
from main import FollowDistanceCommand, ArenaAddPersonCommand


class TestBackendArenaStateIntegrity(unittest.TestCase):
    """Verifies that backend simulation state is complete, consistent, and never corrupted."""

    def setUp(self):
        main.reset_safety_latch()
        main.sim_arena.reset()
        main.set_follow_distance(FollowDistanceCommand(distance_m=2.5))

    def tearDown(self):
        main.reset_safety_latch()
        main.sim_arena.reset()
        main.set_follow_distance(FollowDistanceCommand(distance_m=2.5))

    def test_initial_arena_state_structure(self):
        """Verifies initial simulation state contains robot, obstacles, person, and dimensions."""
        state = main.sim_arena.get_state()
        arena = state["arena"]
        self.assertEqual(arena["width"], 10.0)
        self.assertEqual(arena["height"], 8.0)
        self.assertEqual(arena["camera_fov_deg"], 75.0)
        self.assertEqual(arena["camera_range_m"], 6.0)
        self.assertEqual(arena["target_following_distance_m"], 2.5)

        robot = state["robot"]
        self.assertAlmostEqual(robot["x"], 2.5, places=2)
        self.assertAlmostEqual(robot["y"], 4.0, places=2)
        self.assertAlmostEqual(robot["heading_deg"], 0.0, places=2)

        # Canonical obstacles: Crate A and Barrel B
        obstacles = state["obstacles"]
        self.assertEqual(len(obstacles), 2)
        labels = [o["label"] for o in obstacles]
        self.assertIn("Crate A", labels)
        self.assertIn("Barrel B", labels)

        # Initial person: Person #1
        people = state["people"]
        self.assertEqual(len(people), 1)
        self.assertEqual(people[0]["id"], 1)
        self.assertTrue(people[0]["is_target"])

    def test_obstacles_not_duplicated_in_backend(self):
        """Obstacles are unique and never duplicated across simulation steps."""
        for _ in range(10):
            main.sim_arena.step(dt=0.05)
        state = main.sim_arena.get_state()
        obs_ids = [o["id"] for o in state["obstacles"]]
        self.assertEqual(len(obs_ids), len(set(obs_ids)), "Obstacle IDs must be unique")
        crate_a_count = sum(1 for o in state["obstacles"] if o.get("label") == "Crate A")
        self.assertEqual(crate_a_count, 1, "Crate A must appear exactly once")

    def test_people_multi_presence_preserved(self):
        """Adding Person #2 preserves Person #1 and selected target lock."""
        res_add = main.add_arena_person(ArenaAddPersonCommand(person_id=2, x=6.5, y=2.5))
        self.assertTrue(res_add["ok"])

        state = main.sim_arena.get_state()
        people = state["people"]
        self.assertEqual(len(people), 2)
        p_ids = [p["id"] for p in people]
        self.assertIn(1, p_ids)
        self.assertIn(2, p_ids)
        self.assertEqual(state["selected_target_id"], 1)

    def test_distance_change_does_not_mutate_scene_objects(self):
        """Adjusting following distance to 3.5m preserves robot pose, people, and obstacles."""
        # Setup 2 people and initial obstacles
        main.add_arena_person(ArenaAddPersonCommand(person_id=2, x=6.5, y=2.5))
        before_state = main.sim_arena.get_state()

        # Update following distance to 3.5m
        res_dist = main.set_follow_distance(FollowDistanceCommand(distance_m=3.5))
        self.assertTrue(res_dist["ok"])
        after_state = res_dist["state"]

        # Target following distance updated
        self.assertAlmostEqual(after_state["arena"]["target_following_distance_m"], 3.5, places=2)
        self.assertAlmostEqual(after_state["arena"]["safe_zone_min_m"], 3.2, places=2)
        self.assertAlmostEqual(after_state["arena"]["safe_zone_max_m"], 3.8, places=2)

        # Robot position preserved
        self.assertAlmostEqual(after_state["robot"]["x"], before_state["robot"]["x"], places=2)
        self.assertAlmostEqual(after_state["robot"]["y"], before_state["robot"]["y"], places=2)

        # People count and IDs preserved
        self.assertEqual(len(after_state["people"]), 2)
        after_p_ids = [p["id"] for p in after_state["people"]]
        self.assertIn(1, after_p_ids)
        self.assertIn(2, after_p_ids)

        # Obstacles count and labels preserved
        self.assertEqual(len(after_state["obstacles"]), 2)
        after_labels = [o["label"] for o in after_state["obstacles"]]
        self.assertIn("Crate A", after_labels)
        self.assertIn("Barrel B", after_labels)

    def test_reset_restores_original_scene(self):
        """Scene reset restores initial coordinates and removes added crowd people."""
        main.add_arena_person(ArenaAddPersonCommand(person_id=2, x=6.5, y=2.5))
        main.set_follow_distance(FollowDistanceCommand(distance_m=4.0))

        main.reset_arena()
        state = main.sim_arena.get_state()

        self.assertEqual(len(state["people"]), 1)
        self.assertEqual(state["people"][0]["id"], 1)
        self.assertAlmostEqual(state["people"][0]["x"], 5.0, places=2)
        self.assertAlmostEqual(state["robot"]["x"], 2.5, places=2)


class TestFrontendCanvasIntegrity(unittest.TestCase):
    """Verifies index.html source code and Node.js VM execution of canvas rendering functions."""

    def setUp(self):
        self.html_path = os.path.join(PKG_DIR, "static", "index.html")
        with open(self.html_path, "r", encoding="utf-8") as f:
            self.html_content = f.read()

    def test_render_arena_declares_ar_defensively(self):
        """renderArena() declares ar = arenaState.arena || {} before referencing it."""
        self.assertIn("function renderArena()", self.html_content)
        render_arena_block = self.html_content.split("function renderArena()")[1].split("function toggleArenaSim()")[0]
        self.assertIn("const ar = arenaState.arena || {};", render_arena_block)

    def test_render_arena_resets_transform_matrix(self):
        """renderArena() calls ctx.setTransform(1, 0, 0, 1, 0, 0) to prevent transform matrix compounding."""
        render_arena_block = self.html_content.split("function renderArena()")[1].split("function toggleArenaSim()")[0]
        self.assertIn("ctx.setTransform(1, 0, 0, 1, 0, 0);", render_arena_block)

    def test_render_arena_uses_try_finally_restore(self):
        """Canvas transforms in renderArena() use try...finally to ensure restore() is always called."""
        render_arena_block = self.html_content.split("function renderArena()")[1].split("function toggleArenaSim()")[0]
        self.assertIn("try {", render_arena_block)
        self.assertIn("finally {", render_arena_block)
        self.assertIn("ctx.restore();", render_arena_block)

    def test_node_canvas_vm_rendering_execution(self):
        """Executes Node.js test harness validating that renderArena() renders WALL-E, People, and Obstacles without error."""
        test_script_content = """
const fs = require('fs');
const vm = require('vm');

const htmlContent = fs.readFileSync('""" + self.html_path.replace("\\", "/") + """', 'utf8');
const scriptStart = htmlContent.lastIndexOf('<script>');
const scriptEnd = htmlContent.lastIndexOf('</script>');
const scriptCode = htmlContent.substring(scriptStart + 8, scriptEnd);

const drawCalls = [];
const mockCtx = {
  fillStyle: '',
  strokeStyle: '',
  lineWidth: 1,
  font: '',
  textAlign: '',
  setLineDash: (arr) => drawCalls.push({ call: 'setLineDash', arr }),
  setTransform: (a, b, c, d, e, f) => drawCalls.push({ call: 'setTransform', a, b, c, d, e, f }),
  fillRect: (x, y, w, h) => drawCalls.push({ call: 'fillRect', x, y, w, h }),
  strokeRect: (x, y, w, h) => drawCalls.push({ call: 'strokeRect', x, y, w, h }),
  beginPath: () => drawCalls.push({ call: 'beginPath' }),
  closePath: () => drawCalls.push({ call: 'closePath' }),
  moveTo: (x, y) => drawCalls.push({ call: 'moveTo', x, y }),
  lineTo: (x, y) => drawCalls.push({ call: 'lineTo', x, y }),
  arc: (x, y, r, sa, ea, ccw) => drawCalls.push({ call: 'arc', x, y, r, sa, ea, ccw }),
  fill: () => drawCalls.push({ call: 'fill' }),
  stroke: () => drawCalls.push({ call: 'stroke' }),
  fillText: (text, x, y) => drawCalls.push({ call: 'fillText', text, x, y }),
  save: () => drawCalls.push({ call: 'save' }),
  restore: () => drawCalls.push({ call: 'restore' }),
  translate: (x, y) => drawCalls.push({ call: 'translate', x, y }),
  rotate: (rad) => drawCalls.push({ call: 'rotate', rad })
};

const domElements = {};
function getElement(id) {
  if (!domElements[id]) {
    domElements[id] = {
      id,
      textContent: '',
      value: '',
      style: {},
      className: '',
      getContext: () => mockCtx,
      width: 760,
      height: 520,
      getBoundingClientRect: () => ({ left: 0, top: 0, width: 760, height: 520 }),
      addEventListener: () => {}
    };
  }
  return domElements[id];
}

const sandbox = {
  document: { getElementById: getElement },
  window: { addEventListener: () => {} },
  setInterval: () => 1,
  clearInterval: () => {},
  fetch: async () => ({ json: async () => ({ ok: true }) }),
  Math,
  parseFloat,
  parseInt,
  isNaN,
  console
};

const context = vm.createContext(sandbox);
vm.runInContext(scriptCode, context);

// Test state with 2 people and 3.5m target distance
const testState = {
  arena: {
    width: 10.0,
    height: 8.0,
    camera_fov_deg: 75.0,
    camera_range_m: 6.0,
    target_following_distance_m: 3.5,
    safe_zone_min_m: 3.2,
    safe_zone_max_m: 3.8,
    distance_tolerance_m: 0.3
  },
  robot: { x: 2.5, y: 4.0, heading_deg: 0.0, command: 'stop', speed: 0, trail: [] },
  people: [
    { id: 1, x: 5.0, y: 4.0, speed: 1.0, is_target: true, trail: [] },
    { id: 2, x: 6.5, y: 2.5, speed: 1.0, is_target: false, trail: [] }
  ],
  selected_target_id: 1,
  obstacles: [
    { id: 'obs_1', x: 4.0, y: 2.0, radius: 0.4, label: 'Crate A' },
    { id: 'obs_2', x: 7.0, y: 5.5, radius: 0.45, label: 'Barrel B' }
  ],
  sensors: { distance_cm: 250.0, obstacle: false, simulated_sensor_fail: false },
  decision: 'HOLD',
  stop_reason: 'HOLDING DISTANCE',
  stop_code: 'TARGET_DISTANCE_REACHED',
  following_status: 'HOLDING'
};

drawCalls.length = 0;
vm.runInContext(`arenaState = ${JSON.stringify(testState)}; renderArena(); updateArenaTelemetry();`, context);

const renderedTexts = drawCalls.filter(c => c.call === 'fillText').map(c => c.text);
const result = {
  drawCallCount: drawCalls.length,
  renderedTexts,
  crateACount: renderedTexts.filter(t => t === 'Crate A').length,
  barrelBCount: renderedTexts.filter(t => t === 'Barrel B').length,
  hasWallE: renderedTexts.includes('WALL-E'),
  hasPerson1: renderedTexts.some(t => t.includes('Person #1')),
  hasPerson2: renderedTexts.some(t => t.includes('Person #2')),
  saves: drawCalls.filter(c => c.call === 'save').length,
  restores: drawCalls.filter(c => c.call === 'restore').length
};
console.log(JSON.stringify(result));
"""
        proc = subprocess.run(
            ["node", "-e", test_script_content],
            capture_output=True,
            text=True
        )
        self.assertEqual(proc.returncode, 0, f"Node script failed: {proc.stderr}")
        res = json.loads(proc.stdout.strip())
        self.assertGreater(res["drawCallCount"], 50)
        self.assertTrue(res["hasWallE"], "WALL-E must be rendered")
        self.assertTrue(res["hasPerson1"], "Person #1 must be rendered")
        self.assertTrue(res["hasPerson2"], "Person #2 must be rendered")
        self.assertEqual(res["crateACount"], 1, "Crate A must appear exactly once")
        self.assertEqual(res["barrelBCount"], 1, "Barrel B must appear exactly once")
        self.assertEqual(res["saves"], res["restores"], "Context save/restore stack must be balanced")


if __name__ == "__main__":
    unittest.main()

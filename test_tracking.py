import os
import sys
import unittest
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "walle_software_starter")
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import main
from tracker import CentroidTracker, TrackedObject


class TestMultiObjectTracking(unittest.TestCase):
    def setUp(self):
        # Reset global tracker and state before each test
        main.tracker.reset()
        main.state["objects"] = []
        main.state["object_count"] = 0
        main.state["object"] = {
            "detected": False,
            "id": None,
            "color": "NONE",
            "direction": "NO TARGET",
            "movement": "NONE",
            "x": 0,
            "y": 0,
            "area": 0,
        }

    def test_one_moving_object_retains_id_and_tracks_movement(self):
        """1. One moving object should retain its ID across consecutive frames and estimate movement."""
        t = CentroidTracker(max_disappeared=5, max_distance=80.0, movement_threshold=3.0)

        # Frame 1: Object at (100, 100)
        dets1 = [{"color": "RED", "x": 100, "y": 100, "bbox": (80, 80, 40, 40), "area": 1600, "direction": "LEFT"}]
        res1 = t.update(dets1)
        self.assertEqual(len(res1), 1)
        orig_id = res1[0]["id"]
        self.assertEqual(res1[0]["movement"], "STATIONARY")

        # Frame 2: Object moves right to (125, 100) (dx = +25, dy = 0)
        dets2 = [{"color": "RED", "x": 125, "y": 100, "bbox": (105, 80, 40, 40), "area": 1600, "direction": "LEFT"}]
        res2 = t.update(dets2)
        self.assertEqual(len(res2), 1)
        self.assertEqual(res2[0]["id"], orig_id, "Moving object must retain its original ID")
        self.assertEqual(res2[0]["movement"], "MOVING RIGHT")
        self.assertEqual(res2[0]["dx"], 25)
        self.assertEqual(res2[0]["dy"], 0)

        # Frame 3: Object moves down-right to (140, 120) (dx = +15, dy = +20)
        dets3 = [{"color": "RED", "x": 140, "y": 120, "bbox": (120, 100, 40, 40), "area": 1600, "direction": "LEFT"}]
        res3 = t.update(dets3)
        self.assertEqual(len(res3), 1)
        self.assertEqual(res3[0]["id"], orig_id, "Moving object must continue retaining original ID")
        self.assertEqual(res3[0]["movement"], "MOVING DOWN-RIGHT")

        # Frame 4: Object stops moving at (140, 120)
        dets4 = [{"color": "RED", "x": 140, "y": 120, "bbox": (120, 100, 40, 40), "area": 1600, "direction": "LEFT"}]
        res4 = t.update(dets4)
        self.assertEqual(len(res4), 1)
        self.assertEqual(res4[0]["id"], orig_id)
        self.assertEqual(res4[0]["movement"], "STATIONARY")

    def test_two_same_color_objects_receive_different_ids(self):
        """2. Two same-colour objects should receive different IDs and track independently."""
        t = CentroidTracker(max_disappeared=5, max_distance=80.0)

        # Frame 1: Two RED objects at x=100 and x=300
        dets1 = [
            {"color": "RED", "x": 100, "y": 200, "bbox": (80, 180, 40, 40), "area": 1600, "direction": "LEFT"},
            {"color": "RED", "x": 300, "y": 200, "bbox": (280, 180, 40, 40), "area": 1600, "direction": "CENTER"}
        ]
        res1 = t.update(dets1)
        self.assertEqual(len(res1), 2)
        id1, id2 = res1[0]["id"], res1[1]["id"]
        self.assertNotEqual(id1, id2, "Two same-colour objects must receive distinct IDs")

        # Frame 2: Both move slightly
        dets2 = [
            {"color": "RED", "x": 110, "y": 200, "bbox": (90, 180, 40, 40), "area": 1600, "direction": "LEFT"},
            {"color": "RED", "x": 310, "y": 200, "bbox": (290, 180, 40, 40), "area": 1600, "direction": "CENTER"}
        ]
        res2 = t.update(dets2)
        self.assertEqual(len(res2), 2)
        current_ids = {obj["id"] for obj in res2}
        self.assertEqual(current_ids, {id1, id2}, "Both IDs must be preserved without swapping or duplication")

    def test_different_colors_tracked_independently(self):
        """3. Objects of different colours should be tracked independently, even when close."""
        t = CentroidTracker(max_disappeared=5, max_distance=80.0)

        # Frame 1: RED object at (100, 100) and BLUE object at (110, 100) (very close!)
        dets1 = [
            {"color": "RED", "x": 100, "y": 100, "bbox": (80, 80, 40, 40), "area": 1600, "direction": "LEFT"},
            {"color": "BLUE", "x": 110, "y": 100, "bbox": (90, 80, 40, 40), "area": 1600, "direction": "LEFT"}
        ]
        res1 = t.update(dets1)
        self.assertEqual(len(res1), 2)
        red_obj = next(o for o in res1 if o["color"] == "RED")
        blue_obj = next(o for o in res1 if o["color"] == "BLUE")
        self.assertNotEqual(red_obj["id"], blue_obj["id"])

        # Frame 2: RED moves right, BLUE moves left
        dets2 = [
            {"color": "RED", "x": 120, "y": 100, "bbox": (100, 80, 40, 40), "area": 1600, "direction": "LEFT"},
            {"color": "BLUE", "x": 90, "y": 100, "bbox": (70, 80, 40, 40), "area": 1600, "direction": "LEFT"}
        ]
        res2 = t.update(dets2)
        red_obj2 = next(o for o in res2 if o["color"] == "RED")
        blue_obj2 = next(o for o in res2 if o["color"] == "BLUE")

        # Crucial check: RED track did not match BLUE detection and vice versa
        self.assertEqual(red_obj2["id"], red_obj["id"], "RED track must only associate with RED detection")
        self.assertEqual(blue_obj2["id"], blue_obj["id"], "BLUE track must only associate with BLUE detection")

    def test_temporarily_occluded_object_retains_id(self):
        """4. A temporarily occluded object should retain its ID within the configured frame limit."""
        max_disappeared = 4
        t = CentroidTracker(max_disappeared=max_disappeared, max_distance=80.0)

        # Frame 1: Object appears at (200, 200)
        res1 = t.update([{"color": "GREEN", "x": 200, "y": 200, "bbox": (180, 180, 40, 40), "area": 1600, "direction": "CENTER"}])
        orig_id = res1[0]["id"]

        # Frames 2-4: Object temporarily occluded (3 empty frames <= 4)
        for _ in range(3):
            res = t.update([])
            self.assertEqual(len(res), 0)
            self.assertIn(orig_id, t.tracked_objects, "Track must be retained in memory during temporary occlusion")

        # Frame 5: Object reappears at (205, 202) (close to where it was occluded)
        res_reappear = t.update([{"color": "GREEN", "x": 205, "y": 202, "bbox": (185, 182, 40, 40), "area": 1600, "direction": "CENTER"}])
        self.assertEqual(len(res_reappear), 1)
        self.assertEqual(res_reappear[0]["id"], orig_id, "Reappearing object must retain its original ID within frame limit")

        # Now test exceeding the limit:
        # 5 empty frames (> max_disappeared = 4)
        for _ in range(5):
            t.update([])

        self.assertNotIn(orig_id, t.tracked_objects, "Track must be deregistered after exceeding max_disappeared frames")

        # Reappear after deregistration -> must receive a NEW ID
        res_new = t.update([{"color": "GREEN", "x": 205, "y": 202, "bbox": (185, 182, 40, 40), "area": 1600, "direction": "CENTER"}])
        self.assertEqual(len(res_new), 1)
        self.assertNotEqual(res_new[0]["id"], orig_id, "Object reappearing after deregistration must receive a new ID")

    def test_new_object_receives_new_id(self):
        """5. A new object appearing alongside existing ones should receive a new distinct ID."""
        t = CentroidTracker(max_disappeared=5, max_distance=80.0)

        # Frame 1: Only Object 1
        res1 = t.update([{"color": "YELLOW", "x": 100, "y": 100, "bbox": (80, 80, 40, 40), "area": 1600, "direction": "LEFT"}])
        id1 = res1[0]["id"]

        # Frame 2: Object 1 is still there, and Object 2 appears
        res2 = t.update([
            {"color": "YELLOW", "x": 100, "y": 100, "bbox": (80, 80, 40, 40), "area": 1600, "direction": "LEFT"},
            {"color": "YELLOW", "x": 400, "y": 300, "bbox": (380, 280, 40, 40), "area": 1600, "direction": "RIGHT"}
        ])
        self.assertEqual(len(res2), 2)
        obj1 = next(o for o in res2 if o["x"] == 100)
        obj2 = next(o for o in res2 if o["x"] == 400)

        self.assertEqual(obj1["id"], id1)
        self.assertNotEqual(obj2["id"], id1, "New object must receive a new unique ID")
        self.assertGreater(obj2["id"], id1)

    def test_full_pipeline_with_synthetic_frames(self):
        """Test detect_colored_objects with synthetic frames over multiple consecutive steps."""
        test_tracker = CentroidTracker(max_disappeared=5, max_distance=80.0, movement_threshold=3.0)

        # Step 1: Draw a red rectangle at (100, 100, 50, 50)
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame1, (100, 100), (150, 150), (0, 0, 255), -1)

        tracked1, _ = main.detect_colored_objects(frame1, tracker_instance=test_tracker)
        self.assertEqual(len(tracked1), 1)
        id1 = tracked1[0]["id"]
        self.assertEqual(tracked1[0]["color"], "RED")
        self.assertEqual(tracked1[0]["movement"], "STATIONARY")

        # Step 2: Move red rectangle right to (130, 100, 50, 50)
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame2, (130, 100), (180, 150), (0, 0, 255), -1)

        tracked2, _ = main.detect_colored_objects(frame2, tracker_instance=test_tracker)
        self.assertEqual(len(tracked2), 1)
        self.assertEqual(tracked2[0]["id"], id1, "ID must be preserved in synthetic frame pipeline")
        self.assertEqual(tracked2[0]["movement"], "MOVING RIGHT")

    def test_api_status_exposes_tracking_and_movement(self):
        """Verify that /api/status exposes tracking ID and movement information."""
        # 1. When no objects
        main.state["objects"] = []
        main.state["object_count"] = 0
        main.state["object"] = {
            "detected": False,
            "id": None,
            "color": "NONE",
            "direction": "NO TARGET",
            "movement": "NONE",
            "x": 0,
            "y": 0,
            "area": 0,
        }

        st = main.status()
        self.assertEqual(st["object_count"], 0)
        self.assertEqual(st["objects"], [])
        self.assertFalse(st["object"]["detected"])
        self.assertIn("movement", st["object"])
        self.assertIn("id", st["object"])

        # 2. When object tracked
        main.state["objects"] = [{
            "id": 42,
            "color": "BLUE",
            "x": 320,
            "y": 240,
            "direction": "CENTER",
            "movement": "MOVING LEFT",
            "dx": -15,
            "dy": 0,
            "speed": 15.0,
            "area": 2500
        }]
        main.state["object_count"] = 1
        main.state["object"] = {
            "detected": True,
            **main.state["objects"][0]
        }

        st2 = main.status()
        self.assertEqual(st2["object_count"], 1)
        self.assertEqual(st2["objects"][0]["id"], 42)
        self.assertEqual(st2["objects"][0]["movement"], "MOVING LEFT")
        self.assertEqual(st2["object"]["id"], 42)
        self.assertEqual(st2["object"]["movement"], "MOVING LEFT")


if __name__ == "__main__":
    unittest.main()

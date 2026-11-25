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


class TestColorDetectionSystem(unittest.TestCase):
    def setUp(self):
        # Reset state to default before each test
        main.state["objects"] = []
        main.state["object_count"] = 0
        main.state["object"] = {
            "detected": False,
            "color": "NONE",
            "direction": "NO TARGET",
            "x": 0,
            "y": 0,
            "area": 0,
        }

    def test_bbox_distance_calculation(self):
        """Verify Euclidean distance between bounding boxes."""
        # Case A: Overlapping boxes -> distance 0
        b1 = (50, 50, 50, 50)
        b2 = (80, 80, 50, 50)
        self.assertEqual(main.bbox_distance(b1, b2), 0.0)

        # Case B: Box inside another -> distance 0
        b3 = (50, 50, 100, 100)
        b4 = (60, 60, 20, 20)
        self.assertEqual(main.bbox_distance(b3, b4), 0.0)

        # Case C: Horizontally adjacent with vertical overlap
        b5 = (10, 50, 40, 40)   # x: 10..50
        b6 = (65, 50, 40, 40)   # x: 65..105, gap = 15
        self.assertEqual(main.bbox_distance(b5, b6), 15.0)

        # Case D: Vertically adjacent with horizontal overlap
        b7 = (50, 10, 40, 40)   # y: 10..50
        b8 = (50, 70, 40, 40)   # y: 70..110, gap = 20
        self.assertEqual(main.bbox_distance(b7, b8), 20.0)

        # Case E: Diagonal gap
        b9 = (0, 0, 10, 10)     # x: 0..10, y: 0..10
        b10 = (13, 14, 10, 10)  # dx = 3, dy = 4 -> dist = 5.0
        self.assertAlmostEqual(main.bbox_distance(b9, b10), 5.0)

    def test_synthetic_1_fragmented_single_object(self):
        """
        Synthetic Frame 1: One object with multiple fragmented colour regions.
        A single red physical object broken into 3 pieces by gaps of 8 pixels.
        Should be merged into exactly 1 detected object.
        """
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Draw 3 red rectangles (BGR: 0, 0, 255) separated by 8 px gaps
        # Piece 1: (50, 100) to (80, 140) -> 30x40 = 1200 px
        # Piece 2: (88, 100) to (118, 140) -> 30x40 = 1200 px (gap is 8 px)
        # Piece 3: (126, 100) to (156, 140) -> 30x40 = 1200 px (gap is 8 px)
        cv2.rectangle(frame, (50, 100), (80, 140), (0, 0, 255), -1)
        cv2.rectangle(frame, (88, 100), (118, 140), (0, 0, 255), -1)
        cv2.rectangle(frame, (126, 100), (156, 140), (0, 0, 255), -1)

        detected_list, annotated = main.detect_colored_objects(
            frame, min_contour_area=100, min_object_area=500, merge_distance=25
        )

        self.assertEqual(len(detected_list), 1, f"Expected 1 merged object, got {len(detected_list)}")
        obj = detected_list[0]
        self.assertEqual(obj["color"], "RED")
        self.assertEqual(obj["direction"], "LEFT")
        # Center should be around (50 + 156) // 2 = 103, and (100 + 140) // 2 = 120
        self.assertAlmostEqual(obj["x"], 103, delta=5)
        self.assertAlmostEqual(obj["y"], 120, delta=5)
        # Total area should be sum of 3 pieces ≈ 3600
        self.assertGreater(obj["area"], 3000)

    def test_synthetic_2_two_separate_objects_same_color(self):
        """
        Synthetic Frame 2: Two separate objects of the same colour.
        Two red objects separated by 60 pixels (greater than merge threshold).
        Should NOT be merged; should detect 2 separate objects.
        """
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Object 1: (50, 100) to (100, 150) -> width 50, height 50
        cv2.rectangle(frame, (50, 100), (100, 150), (0, 0, 255), -1)

        # Object 2: (160, 100) to (210, 150) -> width 50, height 50
        # Gap between x=100 and x=160 is 60 px (threshold is 25 px)
        cv2.rectangle(frame, (160, 100), (210, 150), (0, 0, 255), -1)

        detected_list, annotated = main.detect_colored_objects(
            frame, min_contour_area=100, min_object_area=500, merge_distance=25
        )

        self.assertEqual(len(detected_list), 2, f"Expected 2 separate objects, got {len(detected_list)}")
        self.assertEqual(detected_list[0]["color"], "RED")
        self.assertEqual(detected_list[1]["color"], "RED")
        # Coordinates should be distinct
        self.assertNotEqual(detected_list[0]["x"], detected_list[1]["x"])

    def test_synthetic_3_two_objects_different_colors(self):
        """
        Synthetic Frame 3: Two objects of different colours.
        One RED object and one BLUE object close together (10 px apart).
        Should NOT be merged despite close proximity.
        """
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Object 1 (RED): (50, 100) to (100, 150)
        cv2.rectangle(frame, (50, 100), (100, 150), (0, 0, 255), -1)

        # Object 2 (BLUE): (110, 100) to (160, 150) -> gap is 10 px
        cv2.rectangle(frame, (110, 100), (160, 150), (255, 0, 0), -1)

        detected_list, annotated = main.detect_colored_objects(
            frame, min_contour_area=100, min_object_area=500, merge_distance=25
        )

        self.assertEqual(len(detected_list), 2, f"Expected 2 objects, got {len(detected_list)}")
        colors = {obj["color"] for obj in detected_list}
        self.assertEqual(colors, {"RED", "BLUE"})

    def test_synthetic_4_no_objects(self):
        """
        Synthetic Frame 4: No objects.
        Empty black frame should return 0 objects without error.
        """
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        detected_list, annotated = main.detect_colored_objects(
            frame, min_contour_area=100, min_object_area=500, merge_distance=25
        )

        self.assertEqual(len(detected_list), 0)

    def test_noise_filtering_below_min_area(self):
        """Verify that small noise specks below MIN_CONTOUR_AREA are discarded."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Draw a tiny red speck (5x5 = 25 px < 100 px)
        cv2.rectangle(frame, (50, 50), (55, 55), (0, 0, 255), -1)

        # Draw an isolated 12x12 speck (= 144 px > 100 px, but < 500 px min_object_area)
        cv2.rectangle(frame, (200, 200), (212, 212), (0, 0, 255), -1)

        detected_list, _ = main.detect_colored_objects(
            frame, min_contour_area=100, min_object_area=500, merge_distance=25
        )

        # Both specks should be rejected (neither is a valid object >= 500)
        self.assertEqual(len(detected_list), 0)

    def test_all_six_colors_detected(self):
        """Verify all 6 supported colors: RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE."""
        # Create a frame with 6 patches, each in a different location
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # We construct HSV colors and convert to BGR to test exact color ranges:
        # RED: H=0, S=200, V=200
        # ORANGE: H=17, S=200, V=200
        # YELLOW: H=30, S=200, V=200
        # GREEN: H=60, S=200, V=200
        # BLUE: H=110, S=200, V=200
        # PURPLE: H=145, S=200, V=200
        color_hsvs = {
            "RED": (0, 200, 200),
            "ORANGE": (17, 200, 200),
            "YELLOW": (30, 200, 200),
            "GREEN": (60, 200, 200),
            "BLUE": (110, 200, 200),
            "PURPLE": (145, 200, 200),
        }

        positions = [
            (30, 30, 70, 70),
            (130, 30, 170, 70),
            (230, 30, 270, 70),
            (330, 30, 370, 70),
            (430, 30, 470, 70),
            (530, 30, 570, 70),
        ]

        for (color_name, hsv_val), (x1, y1, x2, y2) in zip(color_hsvs.items(), positions):
            # Create 1x1 HSV pixel and convert to BGR
            hsv_pixel = np.uint8([[[hsv_val[0], hsv_val[1], hsv_val[2]]]])
            bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
            bgr_color = (int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2]))
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, -1)

        detected_list, _ = main.detect_colored_objects(
            frame, min_contour_area=100, min_object_area=500, merge_distance=25
        )

        detected_colors = {obj["color"] for obj in detected_list}
        self.assertEqual(len(detected_list), 6)
        self.assertEqual(detected_colors, {"RED", "ORANGE", "YELLOW", "GREEN", "BLUE", "PURPLE"})

    def test_state_consistency_with_detection(self):
        """Verify that updating state with detection results is consistent with /api/status."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a large green object (BGR: 0, 255, 0)
        cv2.rectangle(frame, (250, 150), (350, 250), (0, 255, 0), -1)

        detected_list, _ = main.detect_colored_objects(frame)
        self.assertEqual(len(detected_list), 1)

        # Simulate state update as in mjpeg_frames
        main.state["objects"] = detected_list
        main.state["object_count"] = len(detected_list)
        primary = max(detected_list, key=lambda obj: obj["area"])
        main.state["object"] = {"detected": True, **primary}

        status_res = main.status()
        self.assertEqual(status_res["object_count"], 1)
        self.assertEqual(len(status_res["objects"]), 1)
        self.assertTrue(status_res["object"]["detected"])
        self.assertEqual(status_res["object"]["color"], "GREEN")
        self.assertEqual(status_res["object"]["direction"], "CENTER")


if __name__ == "__main__":
    unittest.main()

import os
import sys
import time
import unittest
import numpy as np
import cv2

# Ensure walle_software_starter is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(BASE_DIR, "walle_software_starter")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from person_tracker import (
    PersonTracker, PersonObservation, AppearanceReID, ReacquisitionStatus, extract_appearance_features
)
from person_detector import (
    PersonDetectorBase, CVPersonDetector, MockPersonDetector, PersonDetectorManager, DetectorStatus
)
from follow_controller import (
    FollowController, FollowMode, FollowState, FollowDecision
)
import main
from main import (
    DetectorModeCommand, BenchmarkCommand, FollowPersonTargetCommand
)


class TestPersonDetectionAndTracking(unittest.TestCase):
    """
    Unit and integration tests for Phase 2: Real-time Person Detection & Tracking.
    Validates detector interfaces, model fallback, ID persistence, multi-person tracking,
    appearance Re-ID, occlusion/reacquisition, API endpoints, and safety constraints.
    """

    def setUp(self):
        self.tracker = PersonTracker(max_disappeared_frames=5, stale_timeout_sec=0.8, reid_threshold=0.70)
        self.mock_detector = MockPersonDetector()
        self.manager = PersonDetectorManager(prefer_real=False)
        self.manager.set_mode("SIMULATION")
        self.controller = FollowController(
            person_tracker=self.tracker,
            target_distance_m=1.5,
            stale_timeout_sec=0.8
        )

    # 1. Person detection interface
    def test_person_detection_interface(self):
        # Base interface
        self.assertTrue(issubclass(CVPersonDetector, PersonDetectorBase))
        self.assertTrue(issubclass(MockPersonDetector, PersonDetectorBase))

        # Observation structure
        obs = PersonObservation(
            person_id=1,
            bbox=(100, 100, 80, 200),
            confidence=0.92,
            direction="CENTER"
        )
        self.assertEqual(obs.id, 1)
        self.assertEqual(obs.center, (140, 200))
        self.assertEqual(obs.confidence, 0.92)
        self.assertEqual(obs.direction, "CENTER")
        self.assertTrue(obs.is_valid)
        self.assertIsNone(obs.to_dict()["distance_m"])
        self.assertIn("UNAVAILABLE", obs.to_dict()["distance_status"])

    # 2. Mock detector functionality
    def test_mock_detector_functionality(self):
        p1 = PersonObservation(person_id=1, bbox=(120, 80, 100, 240), confidence=0.88, direction="CENTER")
        self.mock_detector.set_people([p1])
        self.assertEqual(self.mock_detector.status, DetectorStatus.SIMULATION)

        # Detect
        results = self.mock_detector.detect()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, 1)
        self.assertEqual(results[0].bbox, (120, 80, 100, 240))

        # Health
        health = self.mock_detector.get_health()
        self.assertEqual(health["status"], DetectorStatus.SIMULATION)
        self.assertTrue(health["fps"] > 0)

        # Benchmark
        bench = self.mock_detector.benchmark(iterations=5)
        self.assertEqual(bench["status"], DetectorStatus.SIMULATION)
        self.assertTrue(bench["achievable_fps"] > 0)

    # 3. Real detector loading and unavailable-model fallback
    def test_real_detector_loading_and_unavailable_model_fallback(self):
        # Given a missing model path, must NOT crash
        detector = CVPersonDetector(model_path="models/non_existent_test_model.onnx")
        self.assertEqual(detector.status, DetectorStatus.UNAVAILABLE)
        self.assertIsNotNone(detector.error_message)
        self.assertIn("Model file not found", detector.error_message)

        # Calling detect on unavailable detector returns empty list safely
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.detect(test_frame)
        self.assertEqual(dets, [])

        # Health check reflects unavailable status
        health = detector.get_health()
        self.assertEqual(health["status"], DetectorStatus.UNAVAILABLE)
        self.assertFalse(health["model_exists"])

        # Benchmark on unavailable detector returns error safely
        bench = detector.benchmark(iterations=5)
        self.assertEqual(bench["status"], DetectorStatus.UNAVAILABLE)

        # PersonDetectorManager mode switching
        mgr = PersonDetectorManager(prefer_real=True)
        mgr.set_mode("SIMULATION")
        self.assertEqual(mgr.mode, "SIMULATION")
        mgr.set_mode("AUTO")
        self.assertEqual(mgr.mode, "AUTO")
        with self.assertRaises(ValueError):
            mgr.set_mode("INVALID_MODE")

    # 4. Person ID persistence
    def test_person_id_persistence(self):
        # Frame 1: Person appears at (100, 100)
        p_frame1 = PersonObservation(person_id=0, bbox=(100, 100, 80, 200), confidence=0.90)
        tracked = self.tracker.update([p_frame1])
        self.assertEqual(len(tracked), 1)
        assigned_id = tracked[0].id
        self.assertGreater(assigned_id, 0)

        # Frame 2: Person moves slightly to (110, 105)
        p_frame2 = PersonObservation(person_id=0, bbox=(110, 105, 80, 200), confidence=0.91)
        tracked = self.tracker.update([p_frame2])
        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0].id, assigned_id, "ID must persist across slight movements")

        # Frame 3: Person moves to (125, 110)
        p_frame3 = PersonObservation(person_id=0, bbox=(125, 110, 80, 200), confidence=0.89)
        tracked = self.tracker.update([p_frame3])
        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0].id, assigned_id, "ID must remain identical")

    # 5. Multiple-person detection
    def test_multiple_person_detection(self):
        # Two distinct people in frame
        p1 = PersonObservation(person_id=0, bbox=(80, 100, 70, 180), confidence=0.92)
        p2 = PersonObservation(person_id=0, bbox=(420, 100, 70, 180), confidence=0.90)
        tracked = self.tracker.update([p1, p2])
        self.assertEqual(len(tracked), 2)
        id1 = tracked[0].id
        id2 = tracked[1].id
        self.assertNotEqual(id1, id2, "Separate people must receive distinct IDs")

        # Move both slightly
        p1_moved = PersonObservation(person_id=0, bbox=(85, 102, 70, 180), confidence=0.91)
        p2_moved = PersonObservation(person_id=0, bbox=(425, 104, 70, 180), confidence=0.89)
        tracked = self.tracker.update([p1_moved, p2_moved])
        self.assertEqual(len(tracked), 2)
        current_ids = {t.id for t in tracked}
        self.assertEqual(current_ids, {id1, id2})

    # 6. Person target selection
    def test_person_target_selection(self):
        # Two people: ID 1 smaller, ID 2 larger
        p1 = PersonObservation(person_id=1, bbox=(100, 100, 50, 100), confidence=0.90)   # area 5000
        p2 = PersonObservation(person_id=2, bbox=(300, 100, 100, 200), confidence=0.90)  # area 20000
        self.tracker.update([p1, p2])

        # Auto selection chooses largest bounding box (ID 2)
        self.assertIsNone(self.tracker.selected_person_id)
        auto_target = self.tracker.get_selected_target()
        self.assertEqual(auto_target.id, 2)

        # Lock onto ID 1 explicitly
        self.tracker.select_person(1)
        self.assertEqual(self.tracker.selected_person_id, 1)
        locked_target = self.tracker.get_selected_target()
        self.assertEqual(locked_target.id, 1)

        # Clear target reverts to auto selection
        self.tracker.clear_target()
        self.assertIsNone(self.tracker.selected_person_id)
        self.assertEqual(self.tracker.get_selected_target().id, 2)

    # 7. Target disappearance
    def test_target_disappearance(self):
        p1 = PersonObservation(person_id=1, bbox=(100, 100, 80, 200), confidence=0.90)
        self.tracker.update([p1])
        self.tracker.select_person(1)
        self.assertEqual(self.tracker.reacquisition_status, ReacquisitionStatus.TRACKING)

        # Person disappears for more than max_disappeared_frames (5 frames)
        for _ in range(7):
            self.tracker.update([])

        # Target should now be lost and deregistered
        self.assertEqual(self.tracker.reacquisition_status, ReacquisitionStatus.LOST)
        self.assertIsNone(self.tracker.get_selected_target())
        self.assertEqual(len(self.tracker.get_tracked_people()), 0)

    # 8. Temporary occlusion
    def test_temporary_occlusion(self):
        self.controller.set_mode(FollowMode.PERSON)
        self.controller.set_enabled(True)

        p1 = PersonObservation(person_id=1, bbox=(100, 100, 80, 200), confidence=0.90, direction="CENTER")
        self.tracker.update([p1])
        self.tracker.select_person(1)

        # Occluded for 2 frames (< 5 max frames)
        self.tracker.update([])
        self.tracker.update([])

        # Reacquisition status must be OCCLUDED
        self.assertEqual(self.tracker.reacquisition_status, ReacquisitionStatus.OCCLUDED)

        # Controller evaluates: must command STOP while awaiting reacquisition (never wander!)
        cmd = self.controller.evaluate()
        self.assertEqual(self.controller.decision, FollowDecision.STOP)
        self.assertEqual(self.controller.state, FollowState.ACQUIRING)
        self.assertEqual(cmd["direction"], "stop")

    # 9. Reacquisition behavior with appearance Re-ID
    def test_reacquisition_behavior_with_appearance_reid(self):
        # Test appearance feature extraction helper
        test_img = np.zeros((400, 400, 3), dtype=np.uint8)
        test_img[100:300, 100:200] = [180, 50, 50]  # Colored region
        feat1 = extract_appearance_features(test_img, (100, 100, 100, 200))
        self.assertIsNotNone(feat1)
        self.assertEqual(len(feat1), 128)

        # Frame 1: Person appears with feat1
        p1 = PersonObservation(person_id=1, bbox=(100, 100, 100, 200), confidence=0.92, features=feat1)
        self.tracker.update([p1])
        self.tracker.select_person(1)
        self.assertEqual(self.tracker.reacquisition_status, ReacquisitionStatus.TRACKING)

        # Frame 2 & 3: Temporary occlusion
        self.tracker.update([])
        self.tracker.update([])
        self.assertEqual(self.tracker.reacquisition_status, ReacquisitionStatus.OCCLUDED)

        # Frame 4: Person reappears at slightly shifted location with matching features
        p1_reappear = PersonObservation(person_id=0, bbox=(120, 105, 100, 200), confidence=0.91, features=feat1)
        tracked = self.tracker.update([p1_reappear])

        # Must reacquire persistent ID 1 via Re-ID matching
        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0].id, 1, "Should reacquire ID 1 using appearance features")
        self.assertEqual(self.tracker.reacquisition_status, ReacquisitionStatus.TRACKING)

    # 10. Stale detection safety
    def test_stale_detection_safety(self):
        self.controller.set_mode(FollowMode.PERSON)
        self.controller.set_enabled(True)

        # Detection from 3 seconds ago (stale timeout 0.8s)
        stale_obs = PersonObservation(
            person_id=1,
            bbox=(200, 100, 100, 200),
            confidence=0.95,
            timestamp=time.time() - 3.0
        )
        self.tracker.update([stale_obs])

        # Stale detection must result in STOP
        cmd = self.controller.evaluate()
        self.assertEqual(self.controller.decision, FollowDecision.STOP)
        self.assertEqual(cmd["direction"], "stop")

    # 11. Dashboard & API integration
    def test_dashboard_api_integration(self):
        # GET /api/person/detected
        res = main.get_detected_people()
        self.assertTrue(res["ok"])
        self.assertIn("people", res)
        self.assertIn("detector", res)

        # GET /api/person/detector/health
        health_res = main.get_person_detector_health()
        self.assertTrue(health_res["ok"])
        self.assertIn("status", health_res["detector"])
        self.assertIn("inference_time_ms", health_res["detector"])
        self.assertIn("fps", health_res["detector"])

        # POST /api/person/detector/mode
        mode_res = main.set_person_detector_mode(DetectorModeCommand(mode="SIMULATION"))
        self.assertTrue(mode_res["ok"])
        self.assertEqual(mode_res["detector"]["manager_mode"], "SIMULATION")

        # POST /api/person/detector/benchmark
        bench_res = main.run_person_detector_benchmark(BenchmarkCommand(iterations=5))
        self.assertTrue(bench_res["ok"])
        self.assertIn("achievable_fps", bench_res["benchmark"])

        # POST /api/person/target/auto
        auto_res = main.set_person_target_auto()
        self.assertTrue(auto_res["ok"])

        # POST /api/person/target/clear
        clear_res = main.clear_person_target()
        self.assertTrue(clear_res["ok"])

        # Global status includes person_detector telemetry
        status_res = main.status()
        self.assertIn("follow", status_res)
        self.assertIn("person_detector", status_res["follow"])

    # 12. No forward movement without calibrated distance
    def test_no_forward_movement_without_calibrated_distance(self):
        self.controller.set_mode(FollowMode.PERSON)
        self.controller.set_enabled(True)

        # Large person centered in frame
        p_large = PersonObservation(
            person_id=1,
            bbox=(150, 50, 340, 380),  # large bbox
            confidence=0.98,
            direction="CENTER"
        )
        self.tracker.update([p_large])

        # Evaluate controller: Forward movement MUST NOT be generated on bounding-box size
        cmd = self.controller.evaluate()
        self.assertNotEqual(self.controller.decision, FollowDecision.MOVE_FORWARD)
        self.assertEqual(self.controller.decision, FollowDecision.HOLD)
        self.assertEqual(self.controller.state, FollowState.HOLDING_DISTANCE)
        self.assertEqual(cmd["direction"], "stop")

        # Alignment steering is allowed
        p_left = PersonObservation(person_id=1, bbox=(30, 50, 200, 300), confidence=0.95, direction="LEFT")
        self.tracker.update([p_left])
        cmd = self.controller.evaluate()
        self.assertEqual(self.controller.decision, FollowDecision.TURN_LEFT)
        self.assertEqual(cmd["direction"], "left")

    # 13. Emergency STOP priority
    def test_emergency_stop_priority(self):
        self.controller.set_mode(FollowMode.PERSON)
        self.controller.set_enabled(True)
        p = PersonObservation(person_id=1, bbox=(50, 50, 100, 200), confidence=0.95, direction="LEFT")
        self.tracker.update([p])
        self.controller.evaluate()

        # Stop via controller emergency stop
        stop_cmd = self.controller.emergency_stop()
        self.assertFalse(self.controller.enabled)
        self.assertEqual(self.controller.decision, FollowDecision.STOP)
        self.assertEqual(self.controller.state, FollowState.MANUAL_OVERRIDE)
        self.assertEqual(stop_cmd["direction"], "stop")

        # Stop via main.py /api/stop
        self.controller.set_enabled(True)
        main.stop()
        self.assertFalse(main.follow_controller.enabled)


class TestPhase4ARealYOLOPersonDetection(unittest.TestCase):
    """
    Dedicated test suite for Phase 4A: Real YOLO Person Detection.
    Verifies model loading, controlled YOLO decoding, person-class filtering,
    NMS deduplication, detector health reporting, and tracker integration.
    """

    def setUp(self):
        self.model_path = "models/yolov8n.onnx"
        self.detector = CVPersonDetector(
            model_path=self.model_path,
            input_width=320,
            input_height=320,
            conf_threshold=0.50,
            nms_threshold=0.40
        )

    def test_real_model_loading(self):
        """Verifies that YOLOv8n ONNX model loads successfully with OpenCV DNN."""
        self.assertEqual(self.detector.status, DetectorStatus.REAL)
        self.assertIsNotNone(self.detector.net)
        health = self.detector.get_health()
        self.assertTrue(health["model_loaded"])
        self.assertTrue(health["model_exists"])
        self.assertIsNone(health["error"])
        self.assertEqual(health["input_size"], [320, 320])

    def test_yolo_output_decoding_controlled_data(self):
        """
        Tests YOLO output decoding on a synthetic controlled tensor of shape (1, 84, 4).
        Anchor 0: Valid person (conf 0.92)
        Anchor 1: Duplicate person (conf 0.88, overlapping anchor 0)
        Anchor 2: Person below threshold (conf 0.30)
        Anchor 3: Car (conf 0.95, person conf 0.10)
        """
        tensor = np.zeros((1, 84, 4), dtype=np.float32)

        # Anchor 0: cx=160, cy=160, w=60, h=120, person score = 0.92
        tensor[0, 0, 0] = 160.0
        tensor[0, 1, 0] = 160.0
        tensor[0, 2, 0] = 60.0
        tensor[0, 3, 0] = 120.0
        tensor[0, 4, 0] = 0.92  # class 0 = person

        # Anchor 1: cx=162, cy=158, w=58, h=118, person score = 0.88
        tensor[0, 0, 1] = 162.0
        tensor[0, 1, 1] = 158.0
        tensor[0, 2, 1] = 58.0
        tensor[0, 3, 1] = 118.0
        tensor[0, 4, 1] = 0.88  # class 0 = person

        # Anchor 2: cx=50, cy=50, w=30, h=30, person score = 0.30 (below 0.50 threshold)
        tensor[0, 0, 2] = 50.0
        tensor[0, 1, 2] = 50.0
        tensor[0, 2, 2] = 30.0
        tensor[0, 3, 2] = 30.0
        tensor[0, 4, 2] = 0.30  # class 0 = person

        # Anchor 3: cx=250, cy=200, w=80, h=80, car score = 0.95, person score = 0.10
        tensor[0, 0, 3] = 250.0
        tensor[0, 1, 3] = 200.0
        tensor[0, 2, 3] = 80.0
        tensor[0, 3, 3] = 80.0
        tensor[0, 4, 3] = 0.10  # person
        tensor[0, 6, 3] = 0.95  # car (class 2)

        boxes, confidences = self.detector.decode_predictions(tensor, frame_w=640, frame_h=480)

        # Only Anchors 0 and 1 should pass decoding (Anchor 2 low conf, Anchor 3 not a person)
        self.assertEqual(len(boxes), 2)
        self.assertEqual(len(confidences), 2)
        self.assertAlmostEqual(confidences[0], 0.92, places=2)
        self.assertAlmostEqual(confidences[1], 0.88, places=2)

        # Coordinate scaling: frame is 640x480, model input 320x320 -> scale is 2.0x in x, 1.5x in y
        # Anchor 0: cx=160 -> scaled_cx=320, cy=160 -> scaled_cy=240, w=60 -> 120, h=120 -> 180
        # x = 320 - 60 = 260, y = 240 - 90 = 150
        x, y, w, h = boxes[0]
        self.assertEqual(x, 260)
        self.assertEqual(y, 150)
        self.assertEqual(w, 120)
        self.assertEqual(h, 180)

    def test_nms_deduplication(self):
        """Verifies that Non-Maximum Suppression suppresses overlapping detections."""
        boxes = [[260, 150, 120, 180], [262, 148, 118, 178]]
        confidences = [0.92, 0.88]

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.50, 0.40)
        flat = indices.flatten() if hasattr(indices, "flatten") else indices
        self.assertEqual(len(flat), 1)
        self.assertEqual(int(flat[0]), 0, "Highest confidence box should be kept")

    def test_coco_person_only_filtering(self):
        """Verifies that non-person objects with high confidence are ignored."""
        tensor = np.zeros((1, 84, 2), dtype=np.float32)
        # Anchor 0: Dog (class 16) with 0.98 confidence, person score 0.05
        tensor[0, 0:4, 0] = [100, 100, 50, 50]
        tensor[0, 4, 0] = 0.05
        tensor[0, 20, 0] = 0.98  # Dog

        # Anchor 1: Backpack (class 24) with 0.90 confidence, person score 0.02
        tensor[0, 0:4, 1] = [200, 200, 40, 60]
        tensor[0, 4, 1] = 0.02
        tensor[0, 28, 1] = 0.90  # Backpack

        boxes, confidences = self.detector.decode_predictions(tensor, frame_w=640, frame_h=480)
        self.assertEqual(len(boxes), 0, "Non-person classes must be completely filtered out")

    def test_detector_health_reporting(self):
        """Verifies detector health reporting including model, camera and inference flags."""
        health = self.detector.get_health()
        self.assertIn("model_loaded", health)
        self.assertIn("camera_connected", health)
        self.assertIn("inference_active", health)
        self.assertIn("error", health)
        self.assertTrue(health["model_loaded"])
        self.assertFalse(health["camera_connected"])

        # Inject simulated camera connection checker
        self.detector.set_camera_checker(lambda: True)
        self.assertTrue(self.detector.get_health()["camera_connected"])
        self.detector.set_camera_checker(lambda: False)
        self.assertFalse(self.detector.get_health()["camera_connected"])

    def test_unavailable_model_explicit_error(self):
        """Verifies that missing model triggers UNAVAILABLE status and explicit error without crash."""
        bad_detector = CVPersonDetector(model_path="models/non_existent_file.onnx")
        self.assertEqual(bad_detector.status, DetectorStatus.UNAVAILABLE)
        health = bad_detector.get_health()
        self.assertFalse(health["model_loaded"])
        self.assertIsNotNone(health["error"])
        self.assertIn("Model file not found", health["error"])

        # Detection on unavailable detector returns empty list safely
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = bad_detector.detect(test_frame)
        self.assertEqual(dets, [])

    def test_no_silent_mock_detection_in_real_or_auto(self):
        """Verifies that REAL and AUTO modes never silently fall back to mock detections."""
        mgr = PersonDetectorManager(model_path="models/non_existent_file.onnx", prefer_real=True)
        # Even if synthetic people are set on mock detector, active detector in AUTO/REAL must not return them
        mgr.mock_detector.set_people([
            PersonObservation(person_id=99, bbox=(10, 10, 50, 100), confidence=0.99)
        ])
        mgr.set_mode("REAL")
        self.assertEqual(len(mgr.detect(np.zeros((480, 640, 3), dtype=np.uint8))), 0)

        mgr.set_mode("AUTO")
        self.assertEqual(len(mgr.detect(np.zeros((480, 640, 3), dtype=np.uint8))), 0)

        # Only explicit SIMULATION mode returns mock detections
        mgr.set_mode("SIMULATION")
        self.assertEqual(len(mgr.detect(np.zeros((480, 640, 3), dtype=np.uint8))), 1)

    def test_real_inference_execution_and_fps(self):
        """Verifies real forward inference pass on CPU and latency/FPS telemetry."""
        test_frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        dets = self.detector.detect(test_frame)
        self.assertIsInstance(dets, list)

        health = self.detector.get_health()
        self.assertGreater(health["inference_count"], 0)
        self.assertGreater(health["last_latency_ms"], 0.0)
        self.assertGreater(health["fps"], 0.0)

        # Benchmark
        bench = self.detector.benchmark(iterations=3)
        self.assertEqual(bench["status"], DetectorStatus.REAL)
        self.assertGreater(bench["achievable_fps"], 0.0)

    def test_person_tracker_integration_with_real_detections(self):
        """Verifies that detections are ingested by PersonTracker, establishing persistent ID and direction."""
        tracker = PersonTracker(max_disappeared_frames=5)
        obs = PersonObservation(
            person_id=0,
            bbox=(260, 150, 120, 180),
            confidence=0.92,
            features=[0.1] * 128
        )
        tracked = tracker.update([obs], frame_width=640, is_detection_frame=True)
        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0].id, 1)
        self.assertEqual(tracked[0].direction, "CENTER")

        # Second frame: shifted left
        obs_left = PersonObservation(
            person_id=0,
            bbox=(50, 150, 120, 180),
            confidence=0.90,
            features=[0.1] * 128
        )
        tracked2 = tracker.update([obs_left], frame_width=640, is_detection_frame=True)
        self.assertEqual(len(tracked2), 1)
        self.assertEqual(tracked2[0].id, 1, "ID must persist across frames")
        self.assertEqual(tracked2[0].direction, "LEFT")


if __name__ == "__main__":
    unittest.main()

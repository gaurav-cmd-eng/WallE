import os
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import cv2

try:
    from person_tracker import PersonObservation, extract_appearance_features
except ImportError:
    from walle_software_starter.person_tracker import PersonObservation, extract_appearance_features


class DetectorStatus:
    REAL = "REAL"
    SIMULATION = "SIMULATION"
    UNAVAILABLE = "UNAVAILABLE"


class PersonDetectorBase:
    """Abstract base class for person detectors."""
    def detect(self, frame: Optional[np.ndarray]) -> List[PersonObservation]:
        raise NotImplementedError

    def get_health(self) -> Dict[str, Any]:
        raise NotImplementedError

    def benchmark(self, iterations: int = 10) -> Dict[str, Any]:
        raise NotImplementedError


class CVPersonDetector(PersonDetectorBase):
    """
    Lightweight, CPU-compatible real person detector using OpenCV DNN.
    Designed for Raspberry Pi 4 (4GB RAM) execution.
    Supports COCO pretrained models in ONNX format (YOLOv8n, YOLOv5n)
    or MobileNet-SSD.
    Gracefully handles missing model files and errors without crashing.
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 config_path: Optional[str] = None,
                 input_width: Optional[int] = None,
                 input_height: Optional[int] = None,
                 conf_threshold: Optional[float] = None,
                 nms_threshold: Optional[float] = None,
                 person_class_id: int = 0):

        # Configurable paths with environment variable fallbacks
        self.model_path = model_path or os.getenv("PERSON_MODEL_PATH", "models/yolov8n.onnx")
        self.config_path = config_path or os.getenv("PERSON_CONFIG_PATH", None)
        self.input_width = input_width or int(os.getenv("PERSON_INPUT_WIDTH", "320"))
        self.input_height = input_height or int(os.getenv("PERSON_INPUT_HEIGHT", "320"))
        self.conf_threshold = conf_threshold if conf_threshold is not None else float(os.getenv("PERSON_CONFIDENCE_THRESHOLD", "0.50"))
        self.nms_threshold = nms_threshold if nms_threshold is not None else float(os.getenv("PERSON_NMS_THRESHOLD", "0.40"))
        self.person_class_id = person_class_id  # COCO class 0 is Person

        # Network and runtime state
        self.net: Optional[cv2.dnn.Net] = None
        self.status = DetectorStatus.UNAVAILABLE
        self.error_message: Optional[str] = None
        self.backend = "OpenCV-DNN-CPU"
        self.resolved_model_path: Optional[str] = None
        self.camera_checker: Optional[Callable[[], bool]] = None

        # Performance & Benchmark metrics
        self.inference_time_ms: float = 0.0
        self.last_latency_ms: float = 0.0
        self.fps: float = 0.0
        self.inference_count: int = 0
        self.last_inference_timestamp: float = 0.0

        # Attempt model load
        self.load_model()

    def set_camera_checker(self, checker: Callable[[], bool]) -> None:
        """Sets a callable to query camera connection state."""
        self.camera_checker = checker

    def _resolve_model_path(self, path: Optional[str]) -> Optional[str]:
        """Resolves model path across candidate directories."""
        if not path:
            return None
        if os.path.isabs(path):
            return path if os.path.exists(path) else None

        candidates = [
            path,
            os.path.join(os.getcwd(), path),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), path),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", path),
            os.path.join(os.getcwd(), "models", os.path.basename(path)),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", os.path.basename(path)),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", os.path.basename(path)),
        ]
        for c in candidates:
            if c and os.path.exists(c):
                return os.path.abspath(c)
        return None

    def load_model(self) -> bool:
        """
        Attempts to load the neural network model using OpenCV DNN.
        If file is missing or invalid, sets status to UNAVAILABLE without crashing.
        """
        resolved = self._resolve_model_path(self.model_path)
        if not resolved or not os.path.exists(resolved):
            self.net = None
            self.resolved_model_path = None
            self.status = DetectorStatus.UNAVAILABLE
            self.error_message = f"Model file not found at '{self.model_path}'. Configure PERSON_MODEL_PATH or place model in models/."
            return False

        try:
            self.resolved_model_path = resolved
            if self.config_path and os.path.exists(self.config_path):
                self.net = cv2.dnn.readNet(resolved, self.config_path)
            elif resolved.endswith(".onnx"):
                self.net = cv2.dnn.readNetFromONNX(resolved)
            else:
                self.net = cv2.dnn.readNet(resolved)

            # Enforce CPU execution for Raspberry Pi compatibility
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            except Exception:
                pass
            try:
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            except Exception:
                pass

            self.status = DetectorStatus.REAL
            self.error_message = None
            return True
        except Exception as exc:
            self.net = None
            self.resolved_model_path = None
            self.status = DetectorStatus.UNAVAILABLE
            self.error_message = f"Failed to load OpenCV DNN model from '{self.model_path}': {exc}"
            return False

    def decode_predictions(self, output: np.ndarray, frame_w: int, frame_h: int) -> Tuple[List[List[int]], List[float]]:
        """
        Decodes raw model output tensor into candidate bounding boxes and confidences.
        Strictly filters for COCO person class (class 0).
        """
        boxes: List[List[int]] = []
        confidences: List[float] = []

        if output is None or len(output.shape) < 2:
            return boxes, confidences

        # Architecture 1: YOLOv8 / YOLOv5 / YOLOv7 style (1, 84, N) or (1, N, 84)
        if len(output.shape) == 3:
            if output.shape[1] in (84, 85):
                # Channel-first format (1, 84, N) -> transpose to (N, 84)
                data = output[0].transpose((1, 0))
            elif output.shape[2] in (84, 85):
                # Channel-last format (1, N, 84)
                data = output[0]
            elif output.shape[1] < output.shape[2]:
                data = output[0].transpose((1, 0))
            else:
                data = output[0]

            x_scale = frame_w / float(self.input_width)
            y_scale = frame_h / float(self.input_height)

            for row in data:
                # In YOLOv8: row[0:4] = (cx, cy, w, h), row[4:] = class_scores (80 classes)
                # In YOLOv5: row[0:4] = (cx, cy, w, h), row[4] = obj_conf, row[5:] = class_scores
                if len(row) > 80:
                    if len(row) == 84:  # YOLOv8 (no separate objectness score)
                        class_scores = row[4:]
                        person_score = float(class_scores[self.person_class_id])
                        best_class_id = int(np.argmax(class_scores))
                    else:  # YOLOv5 / v7
                        obj_conf = float(row[4])
                        class_scores = row[5:]
                        person_score = obj_conf * float(class_scores[self.person_class_id])
                        best_class_id = int(np.argmax(class_scores))

                    # Filter for COCO person class exceeding threshold
                    if person_score >= self.conf_threshold and best_class_id == self.person_class_id:
                        cx, cy, w, h = row[0:4]
                        x = int((cx - 0.5 * w) * x_scale)
                        y = int((cy - 0.5 * h) * y_scale)
                        w = int(w * x_scale)
                        h = int(h * y_scale)

                        # Bound check
                        x = max(0, min(frame_w - 1, x))
                        y = max(0, min(frame_h - 1, y))
                        w = max(10, min(frame_w - x, w))
                        h = max(10, min(frame_h - y, h))

                        boxes.append([x, y, w, h])
                        confidences.append(person_score)

        # Architecture 2: SSD format (1, 1, N, 7)
        elif len(output.shape) == 4 and output.shape[1] == 1 and output.shape[3] == 7:
            for i in range(output.shape[2]):
                conf = float(output[0, 0, i, 2])
                class_id = int(output[0, 0, i, 1])
                if class_id == self.person_class_id and conf >= self.conf_threshold:
                    x1 = int(output[0, 0, i, 3] * frame_w)
                    y1 = int(output[0, 0, i, 4] * frame_h)
                    x2 = int(output[0, 0, i, 5] * frame_w)
                    y2 = int(output[0, 0, i, 6] * frame_h)
                    w = max(10, x2 - x1)
                    h = max(10, y2 - y1)
                    boxes.append([x1, y1, w, h])
                    confidences.append(conf)

        return boxes, confidences

    def detect(self, frame: Optional[np.ndarray]) -> List[PersonObservation]:
        """
        Performs forward inference on the provided BGR image frame.
        Filters specifically for the COCO person class.
        Extracts appearance features for re-identification.
        Returns a list of PersonObservation instances.
        """
        if frame is None or self.status != DetectorStatus.REAL or self.net is None:
            return []

        t_start = time.perf_counter()
        frame_h, frame_w = frame.shape[:2]

        try:
            # Preprocess into 4D blob (normalization: 1/255.0, resize, swapRB to RGB)
            blob = cv2.dnn.blobFromImage(
                frame,
                scalefactor=1.0 / 255.0,
                size=(self.input_width, self.input_height),
                mean=(0, 0, 0),
                swapRB=True,
                crop=False
            )
            self.net.setInput(blob)
            outputs = self.net.forward()
        except Exception as exc:
            self.error_message = f"Inference execution error: {exc}"
            return []

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        # Update running performance metrics
        self.last_latency_ms = round(latency_ms, 2)
        if self.inference_time_ms == 0.0:
            self.inference_time_ms = latency_ms
        else:
            self.inference_time_ms = 0.85 * self.inference_time_ms + 0.15 * latency_ms
        self.fps = round(1000.0 / self.inference_time_ms if self.inference_time_ms > 0 else 0.0, 1)
        self.inference_count += 1
        self.last_inference_timestamp = time.time()

        # Parse outputs across standard detection architectures
        if isinstance(outputs, (list, tuple)):
            output = outputs[0]
        else:
            output = outputs

        boxes, confidences = self.decode_predictions(output, frame_w, frame_h)

        if not boxes:
            return []

        # Apply Non-Maximum Suppression to deduplicate boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        detections: List[PersonObservation] = []
        now = time.time()

        if len(indices) > 0:
            flat_indices = indices.flatten() if hasattr(indices, "flatten") else indices
            for idx in flat_indices:
                i = int(idx)
                x, y, w, h = boxes[i]
                conf = float(confidences[i])
                # Extract appearance feature vector for Re-ID matching
                features = extract_appearance_features(frame, (x, y, w, h))
                obs = PersonObservation(
                    person_id=0,  # ID assignment will be done persistently by PersonTracker
                    bbox=(x, y, w, h),
                    confidence=conf,
                    features=features,
                    timestamp=now,
                    is_valid=True
                )
                detections.append(obs)

        return detections

    def get_health(self) -> Dict[str, Any]:
        """Provides detector status, model info, and performance telemetry."""
        model_loaded = (self.status == DetectorStatus.REAL and self.net is not None)
        camera_conn = self.camera_checker() if callable(self.camera_checker) else False
        now = time.time()
        inference_active = (self.inference_count > 0 and (now - self.last_inference_timestamp < 5.0))
        return {
            "status": self.status,
            "model_path": self.model_path,
            "resolved_model_path": self.resolved_model_path,
            "model_exists": os.path.exists(self.resolved_model_path or self.model_path) if self.model_path else False,
            "model_loaded": model_loaded,
            "camera_connected": camera_conn,
            "inference_active": inference_active,
            "backend": self.backend,
            "input_size": [self.input_width, self.input_height],
            "conf_threshold": self.conf_threshold,
            "nms_threshold": self.nms_threshold,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "last_latency_ms": round(self.last_latency_ms, 2),
            "fps": self.fps,
            "inference_count": self.inference_count,
            "error": self.error_message
        }

    def benchmark(self, iterations: int = 10) -> Dict[str, Any]:
        """
        Runs CPU inference benchmark on a synthetic test image.
        Measures minimum, maximum, average latency, and achievable FPS on the host CPU.
        """
        if self.status != DetectorStatus.REAL or self.net is None:
            return {
                "status": self.status,
                "error": self.error_message or "Model not loaded. Cannot benchmark.",
                "achievable_fps": 0.0,
                "avg_latency_ms": 0.0
            }

        # Create synthetic 640x480 test image
        test_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        # Warmup pass
        self.detect(test_frame)

        latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            self.detect(test_frame)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        avg_lat = sum(latencies) / len(latencies)
        return {
            "status": self.status,
            "iterations": iterations,
            "min_latency_ms": round(min(latencies), 2),
            "max_latency_ms": round(max(latencies), 2),
            "avg_latency_ms": round(avg_lat, 2),
            "achievable_fps": round(1000.0 / avg_lat, 1) if avg_lat > 0 else 0.0,
            "input_size": [self.input_width, self.input_height]
        }


class MockPersonDetector(PersonDetectorBase):
    """
    Simulation detector for testing without physical camera or neural net weights.
    Generates synthetic person detections or accepts programmatic test cases.
    """
    def __init__(self):
        self.status = DetectorStatus.SIMULATION
        self.synthetic_people: List[PersonObservation] = []
        self.inference_time_ms: float = 2.1
        self.fps: float = 475.0
        self.inference_count: int = 0
        self.last_inference_timestamp: float = 0.0

    def set_people(self, people: List[PersonObservation]):
        self.synthetic_people = people

    def detect(self, frame: Optional[np.ndarray] = None) -> List[PersonObservation]:
        now = time.time()
        self.inference_count += 1
        self.last_inference_timestamp = now
        results = []
        for p in self.synthetic_people:
            # Extract features if frame is provided and features are not set
            features = p.features
            if features is None and frame is not None:
                features = extract_appearance_features(frame, p.bbox)
            results.append(PersonObservation(
                person_id=p.id,
                bbox=p.bbox,
                confidence=p.confidence,
                center=p.center,
                direction=p.direction,
                features=features,
                timestamp=now,
                is_valid=p.is_valid
            ))
        return results

    def get_health(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "model_path": "mock://synthetic",
            "resolved_model_path": None,
            "model_exists": True,
            "model_loaded": False,
            "camera_connected": False,
            "inference_active": False,
            "backend": "Mock-Simulation-Engine",
            "input_size": [320, 320],
            "conf_threshold": 0.50,
            "nms_threshold": 0.40,
            "inference_time_ms": self.inference_time_ms,
            "last_latency_ms": self.inference_time_ms,
            "fps": self.fps,
            "inference_count": self.inference_count,
            "error": None
        }

    def benchmark(self, iterations: int = 10) -> Dict[str, Any]:
        return {
            "status": self.status,
            "iterations": iterations,
            "min_latency_ms": 1.5,
            "max_latency_ms": 2.8,
            "avg_latency_ms": 2.1,
            "achievable_fps": self.fps,
            "input_size": [320, 320],
            "note": "Mock detector simulation benchmark"
        }


class PersonDetectorManager:
    """
    Unified Person Detector Manager.
    Orchestrates real CV detector and mock simulation detector.
    Never silently emits mock detections when real detector or camera is unavailable.
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 prefer_real: bool = True):
        self.real_detector = CVPersonDetector(model_path=model_path)
        self.mock_detector = MockPersonDetector()
        self.camera_checker: Optional[Callable[[], bool]] = None

        # Mode: "REAL", "SIMULATION", or "AUTO"
        if prefer_real and self.real_detector.status == DetectorStatus.REAL:
            self.mode = "REAL"
        elif prefer_real and self.real_detector.status == DetectorStatus.UNAVAILABLE:
            self.mode = "AUTO"  # In AUTO, will use real when available, but never emits fake mock detections
        else:
            self.mode = "SIMULATION"

    def set_camera_checker(self, checker: Callable[[], bool]) -> None:
        """Sets a callable to query camera connection state across all detectors."""
        self.camera_checker = checker
        self.real_detector.set_camera_checker(checker)

    def set_mode(self, mode: str) -> None:
        mode_up = mode.upper()
        if mode_up in ("REAL", "SIMULATION", "AUTO"):
            self.mode = mode_up
        else:
            raise ValueError(f"Invalid detector mode '{mode}'. Expected REAL, SIMULATION, or AUTO.")

    def get_active_detector(self) -> PersonDetectorBase:
        if self.mode == "SIMULATION":
            return self.mock_detector
        elif self.mode == "REAL":
            return self.real_detector
        else:  # AUTO
            # If real detector is available, use real detector.
            # If unavailable, do not silently report mock detections; return real_detector (which safely reports UNAVAILABLE).
            return self.real_detector

    def detect(self, frame: Optional[np.ndarray]) -> List[PersonObservation]:
        detector = self.get_active_detector()
        return detector.detect(frame)

    def get_health(self) -> Dict[str, Any]:
        active = self.get_active_detector()
        health = active.get_health()
        camera_conn = self.camera_checker() if callable(self.camera_checker) else False
        health["mode"] = self.mode
        health["manager_mode"] = self.mode
        health["real_status"] = self.real_detector.status
        health["real_error"] = self.real_detector.error_message
        health["model_loaded"] = (self.real_detector.status == DetectorStatus.REAL and self.real_detector.net is not None)
        health["camera_connected"] = camera_conn
        now = time.time()
        health["inference_active"] = (active.inference_count > 0 and (now - active.last_inference_timestamp < 5.0))
        # Clear error reporting: if real detector is unavailable, report its error
        if self.real_detector.status == DetectorStatus.UNAVAILABLE:
            health["error"] = self.real_detector.error_message
        return health

    def benchmark(self, iterations: int = 10) -> Dict[str, Any]:
        active = self.get_active_detector()
        res = active.benchmark(iterations=iterations)
        res["active_detector"] = active.status
        return res


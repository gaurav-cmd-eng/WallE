import os
import time
import math
from typing import Dict, List, Optional, Tuple, Any


class PersonObservation:
    """
    Represents a detected person observation from a vision system or detector.
    NOTE: Person detection is separate from colour-based blob detection.
    """
    def __init__(self,
                 person_id: int,
                 bbox: Tuple[int, int, int, int],  # (x, y, w, h)
                 confidence: float,
                 center: Optional[Tuple[int, int]] = None,
                 direction: Optional[str] = None,
                 features: Optional[List[float]] = None,
                 timestamp: Optional[float] = None,
                 is_valid: bool = True):
        self.id = person_id
        self.bbox = bbox
        x, y, w, h = bbox
        self.center = center if center is not None else (x + w // 2, y + h // 2)
        self.confidence = float(confidence)
        self.features = features  # Appearance ReID embedding vector
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.is_valid = is_valid
        self.direction = direction or "CENTER"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "bbox": list(self.bbox),
            "center": list(self.center),
            "confidence": round(self.confidence, 3),
            "direction": self.direction,
            "has_reid": self.features is not None,
            "timestamp": round(self.timestamp, 3),
            "is_valid": self.is_valid,
            # Explicit constraint: real metric distance is unavailable without calibrated range sensor
            "distance_m": None,
            "distance_status": "UNAVAILABLE (Awaiting Range Sensor)"
        }


class AppearanceReID:
    """
    Interface for appearance-based person re-identification.
    Computes cosine similarity between appearance feature vectors to associate
    tracks across temporary occlusions and re-entries.
    """
    @staticmethod
    def compute_similarity(feat_a: Optional[List[float]], feat_b: Optional[List[float]]) -> float:
        if not feat_a or not feat_b or len(feat_a) != len(feat_b):
            return 0.0

        dot = sum(a * b for a, b in zip(feat_a, feat_b))
        norm_a = math.sqrt(sum(a * a for a in feat_a))
        norm_b = math.sqrt(sum(b * b for b in feat_b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot / (norm_a * norm_b)

    @classmethod
    def find_best_match(cls,
                        target_features: List[float],
                        candidates: List[PersonObservation],
                        similarity_threshold: float = 0.70) -> Optional[PersonObservation]:
        best_sim = -1.0
        best_candidate = None
        for cand in candidates:
            if cand.features:
                sim = cls.compute_similarity(target_features, cand.features)
                if sim > best_sim:
                    best_sim = sim
                    best_candidate = cand
        if best_sim >= similarity_threshold:
            return best_candidate
        return None


class ReacquisitionStatus:
    IDLE = "IDLE"
    TRACKING = "TRACKING"
    OCCLUDED = "OCCLUDED"
    LOST = "LOST"


def extract_appearance_features(frame: Any, bbox: Tuple[int, int, int, int]) -> Optional[List[float]]:
    """
    Extracts a normalized color-histogram appearance embedding (128-d)
    from a person's bounding box crop for appearance-based re-identification.
    """
    if frame is None:
        return None
    try:
        import cv2
        x, y, w, h = bbox
        frame_h, frame_w = frame.shape[:2]
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(frame_w, int(x + w))
        y2 = min(frame_h, int(y + h))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten().tolist()
    except Exception:
        return None


class PersonTracker:
    """
    Modular Person Tracker.
    Maintains persistent person IDs, supports manual and auto target selection,
    handles temporary occlusion and target reacquisition using appearance ReID,
    and calculates horizontal direction relative to camera frame width.
    """

    def __init__(self,
                 max_disappeared_frames: int = 30,
                 stale_timeout_sec: float = 1.0,
                 reid_threshold: float = 0.70,
                 frame_width: int = 640):
        self.max_disappeared_frames = max_disappeared_frames
        self.stale_timeout_sec = stale_timeout_sec
        self.reid_threshold = reid_threshold
        self.frame_width = frame_width

        # Persistent tracks
        self.tracked_people: Dict[int, PersonObservation] = {}
        self.disappeared_counts: Dict[int, int] = {}
        self.last_seen_timestamps: Dict[int, float] = {}

        # Target selection
        self.selected_person_id: Optional[int] = None
        self.reacquisition_status: str = ReacquisitionStatus.IDLE
        self.next_person_id: int = 1

        # ReID feature storage for reacquisition: person_id -> feature vector
        self.stored_features: Dict[int, List[float]] = {}

    def select_person(self, person_id: Optional[int]) -> None:
        """Lock onto a specific person ID, or pass None for automatic selection."""
        self.selected_person_id = person_id
        if person_id is not None and person_id in self.tracked_people:
            self.reacquisition_status = ReacquisitionStatus.TRACKING
        elif person_id is not None:
            self.reacquisition_status = ReacquisitionStatus.LOST
        else:
            self.reacquisition_status = ReacquisitionStatus.IDLE

    def clear_target(self) -> None:
        """Clears target lock, reverting to auto-selection."""
        self.select_person(None)

    def set_auto_target(self) -> None:
        """Enables automatic person target selection."""
        self.select_person(None)

    def get_tracked_people(self) -> List[PersonObservation]:
        """Returns all currently visible and fresh tracked people."""
        now = time.time()
        return [
            p for p_id, p in self.tracked_people.items()
            if self.disappeared_counts.get(p_id, 0) == 0 and (now - self.last_seen_timestamps.get(p_id, 0.0) <= self.stale_timeout_sec)
        ]

    def compute_direction(self, center_x: int, frame_width: Optional[int] = None) -> str:
        """Calculates horizontal direction (LEFT, CENTER, RIGHT) relative to frame width."""
        fw = frame_width or self.frame_width
        left_bound = fw * 0.35
        right_bound = fw * 0.65
        if center_x < left_bound:
            return "LEFT"
        elif center_x > right_bound:
            return "RIGHT"
        return "CENTER"

    def update(self,
               detections: List[PersonObservation],
               frame_width: Optional[int] = None,
               timestamp: Optional[float] = None,
               is_detection_frame: bool = True) -> List[PersonObservation]:
        """
        Updates persistent tracks with new person detections.
        Matches detections to existing tracks via ID or spatial/ReID association.
        On skipped inference frames (is_detection_frame=False), keeps existing tracks active.
        """
        now = timestamp if timestamp is not None else time.time()
        fw = frame_width or self.frame_width

        # If this is a skipped frame (stride interpolation), return active tracks without incrementing disappeared
        if not is_detection_frame:
            return self.get_tracked_people()

        # Ensure directions and validity on raw detections
        for det in detections:
            det.direction = self.compute_direction(det.center[0], fw)
            # Stale rejection on incoming detection
            if now - det.timestamp > self.stale_timeout_sec:
                det.is_valid = False

        valid_detections = [d for d in detections if d.is_valid]

        # Case 1: No current tracked people
        if not self.tracked_people:
            for det in valid_detections:
                p_id = det.id if det.id > 0 else self.next_person_id
                self.next_person_id = max(self.next_person_id, p_id + 1)
                det.id = p_id
                self.tracked_people[p_id] = det
                self.disappeared_counts[p_id] = 0
                self.last_seen_timestamps[p_id] = now
                if det.features:
                    self.stored_features[p_id] = det.features
            self._update_target_status(now)
            return list(self.tracked_people.values())

        # Case 2: No new detections in current frame
        if not valid_detections:
            for p_id in list(self.tracked_people.keys()):
                self.disappeared_counts[p_id] += 1
                if self.disappeared_counts[p_id] > self.max_disappeared_frames:
                    self._deregister(p_id)
            self._update_target_status(now)
            return [p for p in self.tracked_people.values() if self.disappeared_counts[p.id] == 0]

        # Case 3: Match existing tracks to detections
        existing_ids = list(self.tracked_people.keys())
        matched_det_indices = set()
        matched_track_ids = set()

        # Step 3a: Direct ID match if detector provides persistent ID
        for idx, det in enumerate(valid_detections):
            if det.id in self.tracked_people:
                p_id = det.id
                self.tracked_people[p_id] = det
                self.disappeared_counts[p_id] = 0
                self.last_seen_timestamps[p_id] = now
                if det.features:
                    self.stored_features[p_id] = det.features
                matched_det_indices.add(idx)
                matched_track_ids.add(p_id)

        # Step 3b: Match remaining via ReID or spatial distance
        unmatched_tracks = [t_id for t_id in existing_ids if t_id not in matched_track_ids]
        unmatched_dets = [idx for idx in range(len(valid_detections)) if idx not in matched_det_indices]

        for t_id in unmatched_tracks:
            # Check ReID if available
            stored_feat = self.stored_features.get(t_id)
            best_match_idx = None
            best_score = -1.0

            for d_idx in unmatched_dets:
                cand = valid_detections[d_idx]
                if stored_feat and cand.features:
                    sim = AppearanceReID.compute_similarity(stored_feat, cand.features)
                    if sim > best_score and sim >= self.reid_threshold:
                        best_score = sim
                        best_match_idx = d_idx
                else:
                    # Centroid distance fallback
                    prev_center = self.tracked_people[t_id].center
                    curr_center = cand.center
                    dist = math.hypot(prev_center[0] - curr_center[0], prev_center[1] - curr_center[1])
                    if dist <= 120.0:  # pixels
                        best_match_idx = d_idx
                        break

            if best_match_idx is not None and best_match_idx in unmatched_dets:
                det = valid_detections[best_match_idx]
                det.id = t_id
                self.tracked_people[t_id] = det
                self.disappeared_counts[t_id] = 0
                self.last_seen_timestamps[t_id] = now
                if det.features:
                    self.stored_features[t_id] = det.features
                unmatched_dets.remove(best_match_idx)
                matched_track_ids.add(t_id)

        # Step 3c: Register remaining unmatched detections as new tracks
        for d_idx in unmatched_dets:
            det = valid_detections[d_idx]
            p_id = det.id if det.id > 0 and det.id not in self.tracked_people else self.next_person_id
            self.next_person_id = max(self.next_person_id, p_id + 1)
            det.id = p_id
            self.tracked_people[p_id] = det
            self.disappeared_counts[p_id] = 0
            self.last_seen_timestamps[p_id] = now
            if det.features:
                self.stored_features[p_id] = det.features

        # Step 3d: Increment disappeared for unmatched tracks
        for t_id in existing_ids:
            if t_id not in matched_track_ids:
                self.disappeared_counts[t_id] += 1
                if self.disappeared_counts[t_id] > self.max_disappeared_frames:
                    self._deregister(t_id)

        self._update_target_status(now)
        return [p for p in self.tracked_people.values() if self.disappeared_counts[p.id] == 0]

    def _deregister(self, person_id: int):
        self.tracked_people.pop(person_id, None)
        self.disappeared_counts.pop(person_id, None)
        self.last_seen_timestamps.pop(person_id, None)

    def _update_target_status(self, current_time: float):
        target = self.get_selected_target(current_time)
        if target is None:
            if self.selected_person_id is not None:
                # Was locked on a person who disappeared
                disappeared = self.disappeared_counts.get(self.selected_person_id, 999)
                if 0 < disappeared <= self.max_disappeared_frames:
                    self.reacquisition_status = ReacquisitionStatus.OCCLUDED
                else:
                    self.reacquisition_status = ReacquisitionStatus.LOST
            else:
                self.reacquisition_status = ReacquisitionStatus.IDLE
        else:
            self.reacquisition_status = ReacquisitionStatus.TRACKING

    def get_selected_target(self, current_time: Optional[float] = None, allow_stale: bool = False) -> Optional[PersonObservation]:
        """
        Returns active target person observation if currently visible and fresh.
        Returns None if target is lost, occluded, or stale (unless allow_stale=True).
        """
        now = current_time if current_time is not None else time.time()

        # If specific ID selected:
        if self.selected_person_id is not None:
            if self.selected_person_id in self.tracked_people:
                if self.disappeared_counts.get(self.selected_person_id, 0) == 0:
                    last_seen = self.last_seen_timestamps.get(self.selected_person_id, 0.0)
                    if allow_stale or (now - last_seen <= self.stale_timeout_sec):
                        return self.tracked_people[self.selected_person_id]
            return None

        # Auto-selection: largest visible person by bounding-box area
        visible = [
            p for p_id, p in self.tracked_people.items()
            if self.disappeared_counts.get(p_id, 0) == 0 and (allow_stale or (now - self.last_seen_timestamps.get(p_id, 0.0) <= self.stale_timeout_sec))
        ]
        if not visible:
            return None

        # Sort by bounding-box area (w * h)
        return max(visible, key=lambda p: p.bbox[2] * p.bbox[3])

    def get_status(self) -> Dict[str, Any]:
        target = self.get_selected_target()
        all_visible = [
            p.to_dict() for p_id, p in self.tracked_people.items()
            if self.disappeared_counts.get(p_id, 0) == 0
        ]
        return {
            "people": all_visible,
            "count": len(all_visible),
            "selected_person_id": self.selected_person_id,
            "target": target.to_dict() if target else None,
            "reacquisition_status": self.reacquisition_status,
            "max_disappeared_frames": self.max_disappeared_frames,
            "stale_timeout_sec": self.stale_timeout_sec
        }


class MockPersonDetector:
    """
    Simulation detector for testing person follow functionality without a real
    neural network (e.g. YOLO/MediaPipe). Allows injecting synthetic person detections.
    """
    def __init__(self):
        self.synthetic_people: List[PersonObservation] = []

    def set_people(self, people: List[PersonObservation]):
        self.synthetic_people = people

    def detect(self, frame=None) -> List[PersonObservation]:
        # Return deep copy or fresh timestamps for synthetic detections
        now = time.time()
        results = []
        for p in self.synthetic_people:
            results.append(PersonObservation(
                person_id=p.id,
                bbox=p.bbox,
                confidence=p.confidence,
                center=p.center,
                direction=p.direction,
                features=p.features,
                timestamp=now,
                is_valid=p.is_valid
            ))
        return results

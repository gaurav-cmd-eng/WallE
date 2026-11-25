import math
import os
from typing import Dict, List, Optional, Tuple


class TrackedObject:
    """Represents an active tracked object with persistent ID and motion history."""

    def __init__(self, object_id: int, color: str, centroid: Tuple[int, int],
                 bbox: Tuple[int, int, int, int], area: int, fov_direction: str):
        self.object_id = object_id
        self.color = color
        self.centroid = centroid  # (x, y)
        self.bbox = bbox          # (x, y, w, h)
        self.area = area
        self.fov_direction = fov_direction  # "LEFT", "CENTER", "RIGHT"
        self.disappeared_count = 0
        self.history = [centroid]  # List of recent centroids for motion tracking
        self.max_history = 10
        self.movement = "STATIONARY"
        self.dx = 0
        self.dy = 0
        self.speed = 0.0

    def update(self, centroid: Tuple[int, int], bbox: Tuple[int, int, int, int],
               area: int, fov_direction: str, movement_threshold: float = 3.0):
        prev_x, prev_y = self.centroid
        new_x, new_y = centroid

        self.dx = new_x - prev_x
        self.dy = new_y - prev_y
        self.speed = round(math.hypot(self.dx, self.dy), 1)

        self.centroid = centroid
        self.bbox = bbox
        self.area = area
        self.fov_direction = fov_direction
        self.disappeared_count = 0

        self.history.append(centroid)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        # Estimate movement direction
        self.movement = self._estimate_movement(movement_threshold)

    def mark_missed(self):
        self.disappeared_count += 1
        self.dx = 0
        self.dy = 0
        self.speed = 0.0
        self.movement = "STATIONARY"

    def _estimate_movement(self, threshold: float) -> str:
        if self.speed < threshold:
            return "STATIONARY"

        dx, dy = self.dx, self.dy
        abs_dx, abs_dy = abs(dx), abs(dy)

        # If one direction is dominant (more than 2x the other)
        if abs_dx > 2 * abs_dy:
            return "MOVING RIGHT" if dx > 0 else "MOVING LEFT"
        elif abs_dy > 2 * abs_dx:
            return "MOVING DOWN" if dy > 0 else "MOVING UP"
        else:
            h_dir = "RIGHT" if dx > 0 else "LEFT"
            v_dir = "DOWN" if dy > 0 else "UP"
            return f"MOVING {v_dir}-{h_dir}"

    def to_dict(self) -> dict:
        return {
            "id": self.object_id,
            "color": self.color,
            "x": self.centroid[0],
            "y": self.centroid[1],
            "bbox": self.bbox,
            "direction": self.fov_direction,
            "movement": self.movement,
            "dx": self.dx,
            "dy": self.dy,
            "speed": self.speed,
            "area": self.area
        }


class CentroidTracker:
    """
    Lightweight, color-aware centroid tracker for persistent multi-object tracking.
    Matches detections across consecutive frames to maintain consistent IDs and estimate motion.
    """

    def __init__(self,
                 max_disappeared: Optional[int] = None,
                 max_distance: Optional[float] = None,
                 movement_threshold: Optional[float] = None):
        # Allow configuration from env vars if not passed explicitly
        if max_disappeared is None:
            max_disappeared = int(os.getenv("MAX_DISAPPEARED_FRAMES", "15"))
        if max_distance is None:
            max_distance = float(os.getenv("MAX_TRACKING_DISTANCE", "80.0"))
        if movement_threshold is None:
            movement_threshold = float(os.getenv("MOVEMENT_THRESHOLD", "3.0"))

        self.next_id = 1
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.movement_threshold = movement_threshold

    def register(self, color: str, centroid: Tuple[int, int],
                 bbox: Tuple[int, int, int, int], area: int, fov_direction: str) -> TrackedObject:
        obj = TrackedObject(
            object_id=self.next_id,
            color=color,
            centroid=centroid,
            bbox=bbox,
            area=area,
            fov_direction=fov_direction
        )
        self.tracked_objects[self.next_id] = obj
        self.next_id += 1
        return obj

    def deregister(self, object_id: int):
        if object_id in self.tracked_objects:
            del self.tracked_objects[object_id]

    def reset(self):
        """Reset all tracked objects and ID counter."""
        self.tracked_objects.clear()
        self.next_id = 1

    def update(self, detections: List[dict]) -> List[dict]:
        """
        Updates tracks using current frame detections.
        Each detection is a dict with keys:
            {'color': str, 'x': int, 'y': int, 'bbox': (x, y, w, h), 'area': int, 'direction': str}
        Returns:
            List of dicts representing currently visible tracked objects in this frame.
        """
        # If no detections in this frame
        if not detections:
            to_remove = []
            for obj_id, track in self.tracked_objects.items():
                track.mark_missed()
                if track.disappeared_count > self.max_disappeared:
                    to_remove.append(obj_id)
            for obj_id in to_remove:
                self.deregister(obj_id)
            return []

        # If no existing tracks, register all detections as new tracks
        if not self.tracked_objects:
            visible_objects = []
            for det in detections:
                centroid = (det["x"], det["y"])
                obj = self.register(
                    color=det["color"],
                    centroid=centroid,
                    bbox=det.get("bbox", (det["x"], det["y"], 0, 0)),
                    area=det.get("area", 0),
                    fov_direction=det.get("direction", "CENTER")
                )
                visible_objects.append(obj.to_dict())
            return visible_objects

        # Process detections by colour to ensure tracking is colour-isolated
        detections_by_color: Dict[str, List[dict]] = {}
        for det in detections:
            color = det["color"]
            if color not in detections_by_color:
                detections_by_color[color] = []
            detections_by_color[color].append(det)

        tracks_by_color: Dict[str, List[TrackedObject]] = {}
        for track in self.tracked_objects.values():
            color = track.color
            if color not in tracks_by_color:
                tracks_by_color[color] = []
            tracks_by_color[color].append(track)

        visible_objects = []
        all_colors = set(detections_by_color.keys()).union(tracks_by_color.keys())

        for color in all_colors:
            color_dets = detections_by_color.get(color, [])
            color_tracks = tracks_by_color.get(color, [])

            if not color_tracks:
                # All detections of this colour are new
                for det in color_dets:
                    centroid = (det["x"], det["y"])
                    obj = self.register(
                        color=color,
                        centroid=centroid,
                        bbox=det.get("bbox", (det["x"], det["y"], 0, 0)),
                        area=det.get("area", 0),
                        fov_direction=det.get("direction", "CENTER")
                    )
                    visible_objects.append(obj.to_dict())
                continue

            if not color_dets:
                # No detections for this colour in current frame
                for track in color_tracks:
                    track.mark_missed()
                continue

            # Compute pair distances between tracks and detections of this colour
            pairs = []
            for t_idx, track in enumerate(color_tracks):
                tx, ty = track.centroid
                for d_idx, det in enumerate(color_dets):
                    dx = det["x"] - tx
                    dy = det["y"] - ty
                    dist = math.hypot(dx, dy)
                    pairs.append((dist, t_idx, d_idx))

            # Sort pairs in ascending order of distance
            pairs.sort(key=lambda p: p[0])

            used_tracks = set()
            used_dets = set()

            for dist, t_idx, d_idx in pairs:
                if t_idx in used_tracks or d_idx in used_dets:
                    continue
                if dist > self.max_distance:
                    continue

                track = color_tracks[t_idx]
                det = color_dets[d_idx]

                track.update(
                    centroid=(det["x"], det["y"]),
                    bbox=det.get("bbox", (det["x"], det["y"], 0, 0)),
                    area=det.get("area", 0),
                    fov_direction=det.get("direction", "CENTER"),
                    movement_threshold=self.movement_threshold
                )

                used_tracks.add(t_idx)
                used_dets.add(d_idx)
                visible_objects.append(track.to_dict())

            # Unmatched tracks of this colour mark missed
            for t_idx, track in enumerate(color_tracks):
                if t_idx not in used_tracks:
                    track.mark_missed()

            # Unmatched detections of this colour get new unique IDs
            for d_idx, det in enumerate(color_dets):
                if d_idx not in used_dets:
                    centroid = (det["x"], det["y"])
                    obj = self.register(
                        color=color,
                        centroid=centroid,
                        bbox=det.get("bbox", (det["x"], det["y"], 0, 0)),
                        area=det.get("area", 0),
                        fov_direction=det.get("direction", "CENTER")
                    )
                    visible_objects.append(obj.to_dict())

        # Deregister any tracked objects that exceeded max_disappeared frames
        to_remove = [
            obj_id for obj_id, track in self.tracked_objects.items()
            if track.disappeared_count > self.max_disappeared
        ]
        for obj_id in to_remove:
            self.deregister(obj_id)

        # Sort visible_objects by ID
        visible_objects.sort(key=lambda o: o["id"])
        return visible_objects

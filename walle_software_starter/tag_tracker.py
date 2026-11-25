"""
WALL-E Dual-Mode Autonomous Following System
Module: tag_tracker.py (Phase 3: Advanced Pocket Tag Tracking & UWB/BLE Simulation)

Hardware-independent wireless pocket tag tracking engine supporting:
- Ultra-Wideband (UWB) precision metric ranging.
- Bluetooth Low Energy (BLE) RSSI proximity indicator (non-metric).
- Single UWB receiver directional safety (range does not imply direction).
- Independent directional estimation interface (Camera, AoA, Multilateration).
- Tag identity, pairing, authentication, connection state, and battery simulation.
- Deterministic packet loss, measurement noise, and stale signal detection.
"""

import time
import math
import random
from typing import Dict, List, Optional, Any


class RadioSource:
    UWB = "UWB"
    BLE = "BLE"


class ProximityLevel:
    NEAR = "NEAR"
    MODERATE = "MODERATE"
    FAR = "FAR"
    LOST = "LOST"
    IMMEDIATE = "IMMEDIATE"  # Kept for backward compatibility


class DirectionSource:
    NONE = "NONE"
    EXTERNAL = "EXTERNAL"
    CAMERA_ASSISTED = "CAMERA_ASSISTED"
    AOA = "AOA"
    MULTILATERATION = "MULTILATERATION"


class TagReading:
    """
    Structured wireless tag observation.
    UWB provides calibrated metric distance (is_metric=True).
    BLE provides rough proximity from RSSI (is_metric=False).
    Direction is intentionally separated from single-antenna radio receivers.
    """
    def __init__(self,
                 tag_id: str,
                 source: str,
                 distance_m: Optional[float],
                 is_metric: bool,
                 signal_quality: float,
                 raw_rssi: Optional[float] = None,
                 smoothed_rssi: Optional[float] = None,
                 proximity_level: Optional[str] = None,
                 direction: Optional[str] = None,
                 connected: bool = True,
                 valid: bool = True,
                 battery_pct: Optional[float] = 100.0,
                 error_reason: Optional[str] = None,
                 timestamp: Optional[float] = None,
                 direction_timestamp: Optional[float] = None,
                 direction_valid: Optional[bool] = None,
                 direction_ttl_sec: Optional[float] = 0.5):
        self.tag_id = str(tag_id)
        self.source = source.upper()
        self.distance_m = distance_m
        self.is_metric = is_metric
        self.signal_quality = float(signal_quality)
        self.raw_rssi = raw_rssi
        self.smoothed_rssi = smoothed_rssi if smoothed_rssi is not None else raw_rssi
        self.proximity_level = proximity_level
        self.raw_direction = direction.upper() if direction and direction.upper() in ("LEFT", "CENTER", "RIGHT") else (direction if direction is not None else None)
        self.direction = self.raw_direction
        self.connected = connected
        self.valid = valid
        self.battery_pct = float(battery_pct) if battery_pct is not None else 100.0
        self.error_reason = error_reason
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.direction_timestamp = direction_timestamp
        self.direction_ttl_sec = float(direction_ttl_sec) if direction_ttl_sec is not None else 0.5
        if direction_valid is not None:
            self.direction_valid = bool(direction_valid)
        else:
            self.direction_valid = bool(self.direction and self.direction in ("LEFT", "CENTER", "RIGHT"))
        self.direction_available = self.direction_valid

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tag_id": self.tag_id,
            "source": self.source,
            "distance_m": round(self.distance_m, 2) if self.distance_m is not None else None,
            "is_metric": self.is_metric,
            "signal_quality": round(self.signal_quality, 1),
            "raw_rssi": round(self.raw_rssi, 1) if self.raw_rssi is not None else None,
            "smoothed_rssi": round(self.smoothed_rssi, 1) if self.smoothed_rssi is not None else None,
            "proximity_level": self.proximity_level,
            "direction": self.direction or "UNKNOWN",
            "direction_available": self.direction_available,
            "direction_valid": self.direction_valid,
            "direction_timestamp": round(self.direction_timestamp, 3) if self.direction_timestamp is not None else None,
            "direction_ttl_sec": self.direction_ttl_sec,
            "connected": self.connected,
            "valid": self.valid,
            "battery_pct": round(self.battery_pct, 1),
            "error_reason": self.error_reason,
            "timestamp": round(self.timestamp, 3)
        }


class DirectionProvider:
    """Base interface for independent directional estimation sources."""
    def get_direction(self, tag_id: str, now: Optional[float] = None) -> Optional[str]:
        return None

    def is_available(self, tag_id: Optional[str] = None, now: Optional[float] = None) -> bool:
        return False

    def get_source_type(self) -> str:
        return DirectionSource.NONE


class ManualDirectionProvider(DirectionProvider):
    """External directional source provided manually or via vision bridge with TTL and tag scoping."""
    def __init__(self,
                 direction: Optional[str] = None,
                 tag_id: Optional[str] = None,
                 timestamp: Optional[float] = None,
                 ttl_sec: float = 0.5):
        self.direction = direction.upper() if direction and direction.upper() in ("LEFT", "CENTER", "RIGHT") else None
        self.tag_id = str(tag_id) if tag_id is not None else None
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.ttl_sec = float(ttl_sec)

    def set_direction(self,
                      direction: Optional[str],
                      tag_id: Optional[str] = None,
                      timestamp: Optional[float] = None,
                      ttl_sec: Optional[float] = None) -> None:
        self.direction = direction.upper() if direction and direction.upper() in ("LEFT", "CENTER", "RIGHT") else None
        if tag_id is not None:
            self.tag_id = str(tag_id)
        self.timestamp = timestamp if timestamp is not None else time.time()
        if ttl_sec is not None:
            self.ttl_sec = float(ttl_sec)

    def get_direction(self, tag_id: str, now: Optional[float] = None) -> Optional[str]:
        current_time = now if now is not None else time.time()
        if not self.is_available(tag_id, current_time):
            return None
        return self.direction

    def is_available(self, tag_id: Optional[str] = None, now: Optional[float] = None) -> bool:
        current_time = now if now is not None else time.time()
        if not self.direction or self.direction not in ("LEFT", "CENTER", "RIGHT"):
            return False
        if self.tag_id is not None and tag_id is not None and str(tag_id) != self.tag_id:
            return False
        if (current_time - self.timestamp) > self.ttl_sec:
            return False
        return True

    def get_source_type(self) -> str:
        return DirectionSource.EXTERNAL


class AoADirectionProvider(DirectionProvider):
    """
    Angle-of-Arrival (AoA) multi-antenna simulated direction provider.
    Estimates tag bearing from phase differences across an antenna array.
    """
    def __init__(self, deadzone_deg: float = 15.0, ttl_sec: float = 0.5):
        self.deadzone_deg = deadzone_deg
        self.azimuth_deg: Optional[float] = None
        self.timestamp: Optional[float] = None
        self.ttl_sec: float = float(ttl_sec)

    def set_azimuth(self, azimuth_deg: Optional[float], timestamp: Optional[float] = None) -> None:
        self.azimuth_deg = azimuth_deg
        self.timestamp = timestamp if timestamp is not None else time.time()

    def get_direction(self, tag_id: str, now: Optional[float] = None) -> Optional[str]:
        current_time = now if now is not None else time.time()
        if not self.is_available(tag_id, current_time):
            return None
        if self.azimuth_deg < -self.deadzone_deg:
            return "LEFT"
        elif self.azimuth_deg > self.deadzone_deg:
            return "RIGHT"
        return "CENTER"

    def is_available(self, tag_id: Optional[str] = None, now: Optional[float] = None) -> bool:
        if self.azimuth_deg is None:
            return False
        if self.timestamp is not None and now is not None:
            if (now - self.timestamp) > self.ttl_sec:
                return False
        return True

    def get_source_type(self) -> str:
        return DirectionSource.AOA


class WirelessTagTracker:
    """
    Base class for wireless tag tracking (UWB / BLE).
    Validates readings and rejects stale or disconnected data.
    """
    def __init__(self, source_type: str, stale_timeout_sec: float = 1.0):
        self.source_type = source_type.upper()
        self.stale_timeout_sec = stale_timeout_sec
        self.connected = False
        self.last_reading: Optional[TagReading] = None
        self.last_update_time: float = 0.0

    def validate_reading(self, reading: TagReading, now: Optional[float] = None) -> bool:
        """Validates connection, freshness and data range."""
        current_time = now if now is not None else time.time()

        if not reading.connected:
            return False

        if (current_time - reading.timestamp) > self.stale_timeout_sec:
            return False

        if reading.distance_m is not None:
            if math.isnan(reading.distance_m) or reading.distance_m < 0.0:
                return False

        return True

    def process_reading(self, reading: TagReading, now: Optional[float] = None) -> TagReading:
        current_time = now if now is not None else time.time()
        is_valid = self.validate_reading(reading, current_time)
        reading.valid = is_valid
        if not is_valid and reading.error_reason is None:
            if not reading.connected:
                reading.error_reason = "DISCONNECTED"
            elif (current_time - reading.timestamp) > self.stale_timeout_sec:
                reading.error_reason = "STALE"
            elif reading.distance_m is not None and (math.isnan(reading.distance_m) or reading.distance_m < 0.0):
                reading.error_reason = "INVALID_DISTANCE"

        self.connected = reading.connected
        self.last_reading = reading
        self.last_update_time = reading.timestamp
        return reading

    def get_latest(self, now: Optional[float] = None) -> Optional[TagReading]:
        if not self.last_reading:
            return None
        current_time = now if now is not None else time.time()
        if (current_time - self.last_reading.timestamp) > self.stale_timeout_sec:
            self.last_reading.valid = False
            self.last_reading.error_reason = "STALE"
        return self.last_reading


class UWBTagTracker(WirelessTagTracker):
    """
    Ultra-Wideband (UWB) tracking implementation.
    Provides precise, calibrated metric distance (e.g. 0.1m - 50m).
    Single receiver: does NOT provide direction independently.
    Supports configurable update frequency, measurement noise, and packet loss.
    """
    def __init__(self,
                 stale_timeout_sec: float = 1.0,
                 min_distance: float = 0.1,
                 max_distance: float = 50.0,
                 update_frequency_hz: float = 10.0,
                 noise_std_dev_m: float = 0.0,
                 packet_loss_rate: float = 0.0):
        super().__init__(source_type=RadioSource.UWB, stale_timeout_sec=stale_timeout_sec)
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.update_frequency_hz = update_frequency_hz
        self.noise_std_dev_m = noise_std_dev_m
        self.packet_loss_rate = packet_loss_rate
        self._rng = random.Random(42)

    def set_simulation_params(self,
                              noise_std_dev_m: Optional[float] = None,
                              packet_loss_rate: Optional[float] = None,
                              seed: Optional[int] = None) -> None:
        if noise_std_dev_m is not None:
            self.noise_std_dev_m = float(noise_std_dev_m)
        if packet_loss_rate is not None:
            self.packet_loss_rate = max(0.0, min(1.0, float(packet_loss_rate)))
        if seed is not None:
            self._rng = random.Random(seed)

    def validate_reading(self, reading: TagReading, now: Optional[float] = None) -> bool:
        base_valid = super().validate_reading(reading, now)
        if not base_valid:
            return False

        # UWB MUST provide metric distance within supported range
        if reading.distance_m is None:
            return False

        if reading.distance_m < self.min_distance or reading.distance_m > self.max_distance:
            return False

        return True

    def calculate_signal_quality(self, distance_m: float) -> float:
        """Computes signal quality (0-100%) based on distance and maximum range."""
        if distance_m < self.min_distance or distance_m > self.max_distance:
            return 0.0
        # Linearly scales from 98% down to 25% near max distance
        ratio = max(0.0, min(1.0, (distance_m - self.min_distance) / (self.max_distance - self.min_distance)))
        quality = 98.0 - (ratio * 73.0)
        return max(5.0, min(100.0, quality))

    def record_measurement(self,
                           tag_id: str,
                           distance_m: float,
                           signal_quality: Optional[float] = None,
                           direction: Optional[str] = None,
                           connected: bool = True,
                           battery_pct: Optional[float] = 100.0,
                           timestamp: Optional[float] = None) -> TagReading:
        now = timestamp if timestamp is not None else time.time()

        # Simulate packet loss
        if self.packet_loss_rate > 0.0 and self._rng.random() < self.packet_loss_rate:
            reading = TagReading(
                tag_id=tag_id,
                source=RadioSource.UWB,
                distance_m=None,
                is_metric=True,
                signal_quality=0.0,
                direction=None,
                connected=connected,
                valid=False,
                battery_pct=battery_pct,
                error_reason="PACKET_LOSS",
                timestamp=now
            )
            return self.process_reading(reading, now)

        # Apply measurement noise if configured
        effective_dist = distance_m
        if self.noise_std_dev_m > 0.0 and distance_m >= 0.0:
            noise = self._rng.gauss(0.0, self.noise_std_dev_m)
            effective_dist = max(0.0, distance_m + noise)

        # Compute signal quality if omitted
        if signal_quality is None:
            sig_qual = self.calculate_signal_quality(effective_dist)
        else:
            sig_qual = float(signal_quality)

        # Out-of-range detection
        is_out_of_range = (effective_dist < self.min_distance or effective_dist > self.max_distance)
        err = "OUT_OF_RANGE" if is_out_of_range else None

        reading = TagReading(
            tag_id=tag_id,
            source=RadioSource.UWB,
            distance_m=effective_dist,
            is_metric=True,
            signal_quality=sig_qual,
            direction=direction,
            direction_timestamp=now if direction else None,
            direction_valid=bool(direction and direction.upper() in ("LEFT", "CENTER", "RIGHT")),
            direction_ttl_sec=0.5,
            connected=connected,
            valid=(not is_out_of_range),
            battery_pct=battery_pct,
            error_reason=err,
            timestamp=now
        )
        return self.process_reading(reading, now)


class BLETagTracker(WirelessTagTracker):
    """
    Bluetooth Low Energy (BLE) tracking implementation.
    RSSI is treated as a rough proximity indicator ONLY, not calibrated metric distance.
    is_metric is explicitly False.
    Provides configurable exponential smoothing, noise simulation, and packet loss.
    """
    def __init__(self,
                 stale_timeout_sec: float = 1.0,
                 tx_power_at_1m: float = -59.0,
                 path_loss_exponent: float = 2.5,
                 near_threshold_rssi: float = -68.0,
                 moderate_threshold_rssi: float = -80.0,
                 smoothing_alpha: float = 0.3,
                 noise_std_dev_db: float = 0.0,
                 packet_loss_rate: float = 0.0):
        super().__init__(source_type=RadioSource.BLE, stale_timeout_sec=stale_timeout_sec)
        self.tx_power = tx_power_at_1m
        self.path_loss_exponent = path_loss_exponent
        self.near_threshold_rssi = near_threshold_rssi
        self.moderate_threshold_rssi = moderate_threshold_rssi
        self.smoothing_alpha = smoothing_alpha
        self.noise_std_dev_db = noise_std_dev_db
        self.packet_loss_rate = packet_loss_rate
        self.smoothed_rssi: Optional[float] = None
        self._rng = random.Random(42)

    def set_simulation_params(self,
                              noise_std_dev_db: Optional[float] = None,
                              packet_loss_rate: Optional[float] = None,
                              smoothing_alpha: Optional[float] = None,
                              seed: Optional[int] = None) -> None:
        if noise_std_dev_db is not None:
            self.noise_std_dev_db = float(noise_std_dev_db)
        if packet_loss_rate is not None:
            self.packet_loss_rate = max(0.0, min(1.0, float(packet_loss_rate)))
        if smoothing_alpha is not None:
            self.smoothing_alpha = max(0.01, min(1.0, float(smoothing_alpha)))
        if seed is not None:
            self._rng = random.Random(seed)

    def estimate_proximity_level(self, rssi: float, connected: bool = True) -> str:
        if not connected:
            return ProximityLevel.LOST
        if rssi >= self.near_threshold_rssi:
            return ProximityLevel.NEAR
        elif rssi >= self.moderate_threshold_rssi:
            return ProximityLevel.MODERATE
        return ProximityLevel.FAR

    def estimate_rough_distance(self, rssi: float) -> float:
        """Rough RSSI log-distance approximation for uncalibrated display only."""
        if rssi == 0:
            return 99.0
        ratio = (self.tx_power - rssi) / (10.0 * self.path_loss_exponent)
        return round(10.0 ** ratio, 2)

    def record_measurement(self,
                           tag_id: str,
                           rssi: float,
                           direction: Optional[str] = None,
                           connected: bool = True,
                           battery_pct: Optional[float] = 100.0,
                           timestamp: Optional[float] = None) -> TagReading:
        now = timestamp if timestamp is not None else time.time()

        # Simulate packet loss
        if self.packet_loss_rate > 0.0 and self._rng.random() < self.packet_loss_rate:
            reading = TagReading(
                tag_id=tag_id,
                source=RadioSource.BLE,
                distance_m=None,
                is_metric=False,
                signal_quality=0.0,
                raw_rssi=None,
                smoothed_rssi=self.smoothed_rssi,
                proximity_level=ProximityLevel.LOST,
                direction=None,
                connected=connected,
                valid=False,
                battery_pct=battery_pct,
                error_reason="PACKET_LOSS",
                timestamp=now
            )
            return self.process_reading(reading, now)

        # Apply RSSI noise if configured
        effective_rssi = float(rssi)
        if self.noise_std_dev_db > 0.0:
            effective_rssi += self._rng.gauss(0.0, self.noise_std_dev_db)

        # Apply exponential moving average smoothing
        if self.smoothed_rssi is None:
            self.smoothed_rssi = effective_rssi
        else:
            self.smoothed_rssi = (self.smoothing_alpha * effective_rssi) + ((1.0 - self.smoothing_alpha) * self.smoothed_rssi)

        prox = self.estimate_proximity_level(effective_rssi, connected=connected)
        rough_dist = self.estimate_rough_distance(self.smoothed_rssi)
        sig_quality = max(0.0, min(100.0, (self.smoothed_rssi + 100.0) * (100.0 / 60.0)))

        reading = TagReading(
            tag_id=tag_id,
            source=RadioSource.BLE,
            distance_m=rough_dist,
            is_metric=False,  # BLE RSSI is NOT a calibrated metric distance
            signal_quality=sig_quality,
            raw_rssi=effective_rssi,
            smoothed_rssi=self.smoothed_rssi,
            proximity_level=prox,
            direction=direction,
            direction_timestamp=now if direction else None,
            direction_valid=bool(direction and direction.upper() in ("LEFT", "CENTER", "RIGHT")),
            direction_ttl_sec=0.5,
            connected=connected,
            valid=connected,
            battery_pct=battery_pct,
            error_reason=None if connected else "DISCONNECTED",
            timestamp=now
        )
        return self.process_reading(reading, now)


class TagInfo:
    """Represents a paired or known physical pocket tag in the registry."""
    def __init__(self,
                 tag_id: str,
                 name: str = "",
                 paired: bool = True,
                 authenticated: bool = True,
                 connected: bool = True,
                 battery_pct: float = 100.0,
                 source_type: str = "UWB",
                 last_seen: float = 0.0):
        self.tag_id = str(tag_id)
        self.name = name or f"Tag {tag_id}"
        self.paired = paired
        self.authenticated = authenticated
        self.connected = connected
        self.battery_pct = max(0.0, min(100.0, float(battery_pct)))
        self.source_type = source_type.upper()
        self.last_seen = last_seen if last_seen > 0 else time.time()
        self.signal_quality = 100.0
        self.status = "CONNECTED" if connected else "DISCONNECTED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tag_id": self.tag_id,
            "name": self.name,
            "paired": self.paired,
            "authenticated": self.authenticated,
            "connected": self.connected,
            "battery_pct": round(self.battery_pct, 1),
            "source_type": self.source_type,
            "last_seen": round(self.last_seen, 3),
            "signal_quality": round(self.signal_quality, 1),
            "status": self.status
        }


class TagTrackerManager:
    """
    Unified manager for wireless tags.
    Houses both UWB and BLE trackers behind a common interface.
    Manages tag identity, pairing, authentication, independent direction providers,
    and lost/reacquisition states.
    """
    def __init__(self, default_source: str = RadioSource.UWB, direction_ttl_sec: float = 0.5):
        self.uwb_tracker = UWBTagTracker()
        self.ble_tracker = BLETagTracker()
        self.active_source = default_source.upper()
        self.selected_tag_id: Optional[str] = None
        self.direction_ttl_sec: float = float(direction_ttl_sec)
        self.external_directions: Dict[str, Dict[str, Any]] = {}
        self.external_direction: Optional[str] = None  # Assisted by camera/antenna array
        self.directional_provider: Optional[DirectionProvider] = None

        # Tag Identity & Pairing Registry
        self.tag_registry: Dict[str, TagInfo] = {}
        self.enforce_pairing: bool = False
        self._init_default_tags()

        # Reacquisition state tracking
        self.tag_state: str = "IDLE"  # "IDLE", "CONNECTED", "LOST", "REACQUIRING"
        self.was_lost: bool = False

    def _init_default_tags(self) -> None:
        """Registers default known tag identities to maintain seamless testing."""
        default_ids = ["TAG_01", "TAG_UWB_01", "TAG_BLE_01", "TAG_API_01", "TAG_UWB_42"]
        for tid in default_ids:
            self.tag_registry[tid] = TagInfo(
                tag_id=tid,
                name=f"Pocket Tag {tid}",
                paired=True,
                authenticated=True,
                connected=True,
                battery_pct=100.0
            )

    def set_source(self, source: str) -> None:
        source_up = source.upper()
        if source_up in (RadioSource.UWB, RadioSource.BLE):
            self.active_source = source_up

    def set_direction_ttl(self, ttl_sec: float) -> None:
        """Sets the direction time-to-live in seconds."""
        self.direction_ttl_sec = max(0.01, float(ttl_sec))

    def select_tag(self, tag_id: Optional[str]) -> None:
        self.selected_tag_id = str(tag_id) if tag_id is not None else None

    def set_directional_provider(self, provider: Optional[DirectionProvider]) -> None:
        self.directional_provider = provider

    def set_external_direction(self,
                               direction: Optional[str],
                               tag_id: Optional[str] = None,
                               timestamp: Optional[float] = None,
                               ttl_sec: Optional[float] = None) -> None:
        """
        Direction from external vision or antenna positioning with TTL and tag scoping.
        Associates direction with a specific tag and timestamp.
        """
        now = timestamp if timestamp is not None else time.time()
        ttl = float(ttl_sec) if ttl_sec is not None else self.direction_ttl_sec
        dir_val = direction.upper() if direction and direction.upper() in ("LEFT", "CENTER", "RIGHT") else None

        if dir_val is None:
            self.clear_external_direction(tag_id)
            return

        target_tag = str(tag_id) if tag_id is not None else (self.selected_tag_id or "__DEFAULT__")
        self.external_directions[target_tag] = {
            "direction": dir_val,
            "tag_id": target_tag,
            "timestamp": now,
            "ttl_sec": ttl
        }
        self.external_direction = dir_val
        self.directional_provider = ManualDirectionProvider(
            direction=dir_val,
            tag_id=target_tag if target_tag != "__DEFAULT__" else None,
            timestamp=now,
            ttl_sec=ttl
        )

    def clear_external_direction(self, tag_id: Optional[str] = None) -> None:
        """Explicitly clears external direction data for a tag or all tags."""
        if tag_id is not None:
            tid = str(tag_id)
            self.external_directions.pop(tid, None)
            if self.selected_tag_id == tid or tid == "__DEFAULT__":
                self.external_directions.pop("__DEFAULT__", None)
                self.external_direction = None
                self.directional_provider = None
        else:
            self.external_directions.clear()
            self.external_direction = None
            self.directional_provider = None

    def get_directional_data(self, tag_id: str, now: Optional[float] = None) -> Dict[str, Any]:
        """
        Retrieves directional data for a specific tag.
        Validates tag association and checks TTL freshness.
        """
        current_time = now if now is not None else time.time()

        # 1. Check directional provider
        if self.directional_provider and self.directional_provider.is_available(tag_id, current_time):
            d = self.directional_provider.get_direction(tag_id, current_time)
            ts = getattr(self.directional_provider, "timestamp", current_time)
            ttl = getattr(self.directional_provider, "ttl_sec", self.direction_ttl_sec)
            return {
                "direction": d,
                "direction_available": bool(d in ("LEFT", "CENTER", "RIGHT")),
                "direction_valid": bool(d in ("LEFT", "CENTER", "RIGHT")),
                "direction_timestamp": ts,
                "direction_ttl_sec": ttl,
                "source": self.directional_provider.get_source_type()
            }

        # 2. Check tag-specific external direction
        entry = self.external_directions.get(str(tag_id))
        if entry is None and "__DEFAULT__" in self.external_directions:
            entry = self.external_directions["__DEFAULT__"]

        if entry:
            dir_val = entry["direction"]
            ts = entry["timestamp"]
            ttl = entry.get("ttl_sec", self.direction_ttl_sec)
            age = current_time - ts
            if age <= ttl and dir_val in ("LEFT", "CENTER", "RIGHT"):
                return {
                    "direction": dir_val,
                    "direction_available": True,
                    "direction_valid": True,
                    "direction_timestamp": ts,
                    "direction_ttl_sec": ttl,
                    "source": DirectionSource.EXTERNAL
                }
            else:
                # Stale / expired direction
                return {
                    "direction": None,
                    "direction_available": False,
                    "direction_valid": False,
                    "direction_timestamp": ts,
                    "direction_ttl_sec": ttl,
                    "source": DirectionSource.EXTERNAL
                }

        return {
            "direction": None,
            "direction_available": False,
            "direction_valid": False,
            "direction_timestamp": None,
            "direction_ttl_sec": self.direction_ttl_sec,
            "source": DirectionSource.NONE
        }

    def pair_tag(self, tag_id: str, name: str = "", authenticated: bool = True) -> TagInfo:
        tag_id = str(tag_id)
        info = TagInfo(tag_id=tag_id, name=name, paired=True, authenticated=authenticated, connected=True)
        self.tag_registry[tag_id] = info
        return info

    def unpair_tag(self, tag_id: str) -> bool:
        tag_id = str(tag_id)
        if tag_id in self.tag_registry:
            self.tag_registry[tag_id].paired = False
            self.tag_registry[tag_id].status = "UNPAIRED"
            return True
        return False

    def disconnect_tag(self, tag_id: str) -> bool:
        tag_id = str(tag_id)
        if tag_id in self.tag_registry:
            self.tag_registry[tag_id].connected = False
            self.tag_registry[tag_id].status = "DISCONNECTED"
            return True
        return False

    def reconnect_tag(self, tag_id: str) -> bool:
        tag_id = str(tag_id)
        if tag_id in self.tag_registry:
            self.tag_registry[tag_id].connected = True
            self.tag_registry[tag_id].status = "CONNECTED"
            return True
        return False

    def is_tag_authorized(self, tag_id: str) -> bool:
        if not self.enforce_pairing:
            return True
        info = self.tag_registry.get(tag_id)
        return bool(info and info.paired and info.authenticated)

    def get_paired_tags(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.tag_registry.values() if t.paired]

    def get_active_tracker(self) -> WirelessTagTracker:
        if self.active_source == RadioSource.BLE:
            return self.ble_tracker
        return self.uwb_tracker

    def get_current_reading(self, now: Optional[float] = None) -> Optional[TagReading]:
        current_time = now if now is not None else time.time()
        tracker = self.get_active_tracker()
        reading = tracker.get_latest(current_time)
        if not reading:
            if self.tag_state == "CONNECTED":
                self.tag_state = "LOST"
                self.was_lost = True
            return None

        # Filter by selected tag if specified
        if self.selected_tag_id and reading.tag_id != self.selected_tag_id:
            return None

        # Check pairing authentication
        if self.enforce_pairing and not self.is_tag_authorized(reading.tag_id):
            reading.valid = False
            reading.error_reason = "UNAUTHORIZED_TAG"
            return None

        # Check tag connection in registry
        tag_info = self.tag_registry.get(reading.tag_id)
        if tag_info and not tag_info.connected:
            reading.connected = False
            reading.valid = False
            reading.error_reason = "DISCONNECTED"

        # Update registry state if present
        if tag_info and reading.valid and reading.connected:
            tag_info.last_seen = reading.timestamp
            tag_info.signal_quality = reading.signal_quality
            tag_info.battery_pct = reading.battery_pct
            tag_info.status = "CONNECTED"

        # Merge directional provider or external direction if packet has no native direction
        raw_dir = getattr(reading, "raw_direction", None)
        if raw_dir is None or raw_dir == "UNKNOWN":
            dir_data = self.get_directional_data(reading.tag_id, current_time)
            reading.direction = dir_data["direction"]
            reading.direction_available = dir_data["direction_available"]
            reading.direction_valid = dir_data["direction_valid"]
            reading.direction_timestamp = dir_data["direction_timestamp"]
            reading.direction_ttl_sec = dir_data["direction_ttl_sec"]
        else:
            # Reading has native direction attached in packet (e.g. packet AoA)
            reading.direction_timestamp = getattr(reading, "direction_timestamp", None) or reading.timestamp
            reading.direction_ttl_sec = getattr(reading, "direction_ttl_sec", None) or self.direction_ttl_sec
            age = current_time - reading.direction_timestamp
            is_valid = bool(raw_dir in ("LEFT", "CENTER", "RIGHT") and age <= reading.direction_ttl_sec)
            reading.direction = raw_dir if is_valid else None
            reading.direction_available = is_valid
            reading.direction_valid = is_valid

        # Manage lost / reacquisition state transitions
        if reading.connected and reading.valid:
            if self.was_lost:
                self.tag_state = "REACQUIRING"
                self.was_lost = False
            else:
                self.tag_state = "CONNECTED"
        else:
            self.tag_state = "LOST"
            self.was_lost = True

        return reading

    def get_status(self) -> Dict[str, Any]:
        tracker = self.get_active_tracker()
        current = self.get_current_reading()
        dir_source = (self.directional_provider.get_source_type()
                      if self.directional_provider and self.directional_provider.is_available()
                      else (DirectionSource.EXTERNAL if self.external_direction else DirectionSource.NONE))

        return {
            "active_source": self.active_source,
            "selected_tag_id": self.selected_tag_id,
            "connected": tracker.connected,
            "tag_state": self.tag_state,
            "external_direction": self.external_direction,
            "direction_ttl_sec": self.direction_ttl_sec,
            "directional_source": dir_source,
            "direction_available": bool(current and current.direction_available),
            "direction_valid": bool(current and current.direction_valid),
            "direction_timestamp": current.direction_timestamp if current else None,
            "reading": current.to_dict() if current else None,
            "is_metric": current.is_metric if current else (self.active_source == RadioSource.UWB),
            "stale_timeout_sec": tracker.stale_timeout_sec,
            "paired_tags_count": len([t for t in self.tag_registry.values() if t.paired]),
            "enforce_pairing": self.enforce_pairing,
            "simulation_mode": True
        }


class SimulatedTagDevice:
    """
    Simulation utility to emit virtual UWB or BLE tag readings for testing
    without physical radio transceivers.
    """
    def __init__(self, manager: TagTrackerManager):
        self.manager = manager

    def simulate_uwb(self,
                     tag_id: str = "TAG_UWB_01",
                     distance_m: float = 1.5,
                     signal_quality: float = 98.0,
                     direction: Optional[str] = "CENTER",
                     connected: bool = True,
                     battery_pct: float = 100.0,
                     timestamp: Optional[float] = None) -> TagReading:
        return self.manager.uwb_tracker.record_measurement(
            tag_id=tag_id,
            distance_m=distance_m,
            signal_quality=signal_quality,
            direction=direction,
            connected=connected,
            battery_pct=battery_pct,
            timestamp=timestamp
        )

    def simulate_ble(self,
                     tag_id: str = "TAG_BLE_01",
                     rssi: float = -65.0,
                     direction: Optional[str] = "CENTER",
                     connected: bool = True,
                     battery_pct: float = 100.0,
                     timestamp: Optional[float] = None) -> TagReading:
        return self.manager.ble_tracker.record_measurement(
            tag_id=tag_id,
            rssi=rssi,
            direction=direction,
            connected=connected,
            battery_pct=battery_pct,
            timestamp=timestamp
        )

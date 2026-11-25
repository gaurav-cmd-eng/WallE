"""
WALL-E Dual-Mode Autonomous Following System
Module: tag_simulation.py (Phase 3: Deterministic Simulation Scenarios)

Provides reproducible, deterministic simulation scenarios for UWB ranging,
BLE proximity detection, tag identity management, and FollowController safety.
Uses fixed seeds to guarantee repeatability without physical hardware.
"""

import time
from typing import Dict, List, Optional, Any

try:
    from tag_tracker import (
        TagTrackerManager, UWBTagTracker, BLETagTracker, RadioSource,
        TagReading, SimulatedTagDevice, ProximityLevel
    )
    from follow_controller import (
        FollowController, FollowMode, FollowState, FollowDecision
    )
    from person_tracker import PersonTracker, PersonObservation
except ImportError:
    from walle_software_starter.tag_tracker import (
        TagTrackerManager, UWBTagTracker, BLETagTracker, RadioSource,
        TagReading, SimulatedTagDevice, ProximityLevel
    )
    from walle_software_starter.follow_controller import (
        FollowController, FollowMode, FollowState, FollowDecision
    )
    from walle_software_starter.person_tracker import PersonTracker, PersonObservation


class DeterministicTagSimulator:
    """
    Executes reproducible test scenarios covering UWB ranging, BLE proximity,
    tag lifecycle management, and safety controller behavior.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = seed
        self.person_tracker = PersonTracker(max_disappeared_frames=5, stale_timeout_sec=0.8)
        self.tag_tracker = TagTrackerManager()
        self.controller = FollowController(
            person_tracker=self.person_tracker,
            tag_tracker=self.tag_tracker,
            target_distance_m=1.5,
            distance_tolerance_m=0.3,
            min_safe_distance_m=0.8,
            approach_threshold_m=1.8,
            hold_min_m=1.2,
            hold_max_m=1.8,
            stale_timeout_sec=0.8
        )
        self.sim_tag = SimulatedTagDevice(self.tag_tracker)
        self.tag_tracker.uwb_tracker.set_simulation_params(seed=self.seed)
        self.tag_tracker.ble_tracker.set_simulation_params(seed=self.seed)

    # =========================================================================
    # UWB SCENARIOS (1 to 9)
    # =========================================================================

    def scenario_uwb_1_no_direction(self) -> Dict[str, Any]:
        """UWB Scenario 1: Tag at 2.5m with no directional source -> Must HOLD, never MOVE_FORWARD."""
        self.reset()
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.tag_tracker.set_source(RadioSource.UWB)
        self.controller.set_enabled(True)

        # Record measurement without direction
        reading = self.tag_tracker.uwb_tracker.record_measurement(
            tag_id="TAG_01", distance_m=2.5, direction=None, connected=True
        )
        cmd = self.controller.evaluate()

        assert reading.distance_m == 2.5
        assert not reading.direction_available
        assert self.controller.decision == FollowDecision.HOLD, f"Expected HOLD without direction, got {self.controller.decision}"
        assert self.controller.state == FollowState.HOLDING_DISTANCE
        assert cmd["direction"] == "stop"

        return {
            "scenario": "uwb_1_no_direction",
            "distance_m": reading.distance_m,
            "direction": reading.direction,
            "direction_available": reading.direction_available,
            "decision": self.controller.decision,
            "state": self.controller.state,
            "passed": True
        }

    def scenario_uwb_2_with_direction(self) -> Dict[str, Any]:
        """UWB Scenario 2: Tag at 2.5m with valid independent direction -> MOVE_FORWARD."""
        self.reset()
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.tag_tracker.set_source(RadioSource.UWB)
        self.controller.set_enabled(True)

        reading = self.sim_tag.simulate_uwb(tag_id="TAG_01", distance_m=2.5, direction="CENTER", connected=True)
        cmd = self.controller.evaluate()

        assert self.controller.decision == FollowDecision.MOVE_FORWARD
        assert self.controller.state == FollowState.FOLLOWING
        assert cmd["direction"] == "forward"

        return {
            "scenario": "uwb_2_with_direction",
            "distance_m": reading.distance_m,
            "direction": reading.direction,
            "decision": self.controller.decision,
            "state": self.controller.state,
            "passed": True
        }

    def scenario_uwb_3_holding_distance(self) -> Dict[str, Any]:
        """UWB Scenario 3: Tag at 1.5m (within 1.2m - 1.8m deadband) -> HOLD."""
        self.reset()
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.tag_tracker.set_source(RadioSource.UWB)
        self.controller.set_enabled(True)

        reading = self.sim_tag.simulate_uwb(tag_id="TAG_01", distance_m=1.5, direction="CENTER", connected=True)
        cmd = self.controller.evaluate()

        assert self.controller.decision == FollowDecision.HOLD
        assert self.controller.state == FollowState.HOLDING_DISTANCE
        assert cmd["direction"] == "stop"

        return {
            "scenario": "uwb_3_holding_distance",
            "distance_m": reading.distance_m,
            "decision": self.controller.decision,
            "state": self.controller.state,
            "passed": True
        }

    def scenario_uwb_4_close_safety_stop(self) -> Dict[str, Any]:
        """UWB Scenario 4: Tag at 0.5m (< 0.8m safety perimeter) -> OBSTACLE_STOP."""
        self.reset()
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.tag_tracker.set_source(RadioSource.UWB)
        self.controller.set_enabled(True)

        reading = self.sim_tag.simulate_uwb(tag_id="TAG_01", distance_m=0.5, direction="CENTER", connected=True)
        cmd = self.controller.evaluate()

        assert self.controller.decision == FollowDecision.STOP
        assert self.controller.state == FollowState.OBSTACLE_STOP
        assert cmd["direction"] == "stop"

        return {
            "scenario": "uwb_4_close_safety_stop",
            "distance_m": reading.distance_m,
            "decision": self.controller.decision,
            "state": self.controller.state,
            "passed": True
        }

    def scenario_uwb_5_fluctuating_noise(self, steps: int = 8, noise_std: float = 0.1) -> Dict[str, Any]:
        """UWB Scenario 5: Fluctuating distance with deterministic measurement noise."""
        self.reset(seed=123)
        self.tag_tracker.uwb_tracker.set_simulation_params(noise_std_dev_m=noise_std, seed=123)
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.tag_tracker.set_source(RadioSource.UWB)
        self.controller.set_enabled(True)

        history = []
        base_dist = 2.2
        for i in range(steps):
            r = self.tag_tracker.uwb_tracker.record_measurement(
                tag_id="TAG_01", distance_m=base_dist, direction="CENTER", connected=True
            )
            cmd = self.controller.evaluate()
            history.append({
                "step": i,
                "measured_m": r.distance_m,
                "valid": r.valid,
                "decision": self.controller.decision
            })

        # All noisy readings around 2.2m should remain valid and above 1.8m
        assert all(h["valid"] for h in history)
        assert all(h["decision"] == FollowDecision.MOVE_FORWARD for h in history)

        return {
            "scenario": "uwb_5_fluctuating_noise",
            "steps": steps,
            "history": history,
            "passed": True
        }

    def scenario_uwb_6_packet_loss(self, steps: int = 10, loss_rate: float = 0.4) -> Dict[str, Any]:
        """UWB Scenario 6: Intermittent packet loss simulation."""
        self.reset(seed=99)
        self.tag_tracker.uwb_tracker.set_simulation_params(packet_loss_rate=loss_rate, seed=99)
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.tag_tracker.set_source(RadioSource.UWB)
        self.controller.set_enabled(True)

        dropped = 0
        received = 0
        for _ in range(steps):
            r = self.tag_tracker.uwb_tracker.record_measurement(
                tag_id="TAG_01", distance_m=2.0, direction="CENTER", connected=True
            )
            if r.error_reason == "PACKET_LOSS":
                dropped += 1
            else:
                received += 1

        assert dropped > 0, "Expected at least one dropped packet"
        assert received > 0, "Expected at least one received packet"

        return {
            "scenario": "uwb_6_packet_loss",
            "steps": steps,
            "loss_rate": loss_rate,
            "dropped": dropped,
            "received": received,
            "passed": True
        }

    def scenario_uwb_7_stale_measurement(self) -> Dict[str, Any]:
        """UWB Scenario 7: Stale measurement older than timeout -> COMMUNICATION_ERROR / STOP."""
        self.reset()
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.tag_tracker.set_source(RadioSource.UWB)
        self.controller.set_enabled(True)

        stale_ts = time.time() - 2.5  # Timeout is 0.8s
        self.tag_tracker.uwb_tracker.record_measurement(
            tag_id="TAG_01", distance_m=2.5, direction="CENTER", connected=True, timestamp=stale_ts
        )
        cmd = self.controller.evaluate()

        assert self.controller.decision == FollowDecision.STOP
        assert self.controller.state == FollowState.COMMUNICATION_ERROR

        return {
            "scenario": "uwb_7_stale_measurement",
            "decision": self.controller.decision,
            "state": self.controller.state,
            "passed": True
        }

    def scenario_uwb_8_out_of_range(self) -> Dict[str, Any]:
        """UWB Scenario 8: Tag distance exceeds max_distance (50m) -> OUT_OF_RANGE."""
        self.reset()
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.tag_tracker.set_source(RadioSource.UWB)
        self.controller.set_enabled(True)

        r = self.tag_tracker.uwb_tracker.record_measurement(
            tag_id="TAG_01", distance_m=65.0, direction="CENTER", connected=True
        )
        assert not r.valid
        assert r.error_reason == "OUT_OF_RANGE"

        cmd = self.controller.evaluate()
        assert self.controller.decision == FollowDecision.STOP
        assert self.controller.state == FollowState.COMMUNICATION_ERROR

        return {
            "scenario": "uwb_8_out_of_range",
            "error_reason": r.error_reason,
            "decision": self.controller.decision,
            "passed": True
        }

    def scenario_uwb_9_disconnect_reconnect(self) -> Dict[str, Any]:
        """UWB Scenario 9: Tag disconnection -> TARGET_LOST; reconnection safely enters ACQUIRING."""
        self.reset()
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.tag_tracker.set_source(RadioSource.UWB)
        self.controller.set_enabled(True)

        # Step 1: Active follow
        self.sim_tag.simulate_uwb(tag_id="TAG_01", distance_m=2.5, direction="CENTER", connected=True)
        self.controller.evaluate()
        assert self.controller.state == FollowState.FOLLOWING

        # Step 2: Tag disconnects
        self.sim_tag.simulate_uwb(tag_id="TAG_01", distance_m=2.5, direction="CENTER", connected=False)
        self.controller.evaluate()
        assert self.controller.decision == FollowDecision.STOP
        assert self.controller.state == FollowState.TARGET_LOST

        # Step 3: Tag reconnects -> Must safely transition through ACQUIRING before FOLLOWING
        self.sim_tag.simulate_uwb(tag_id="TAG_01", distance_m=2.5, direction="CENTER", connected=True)
        cmd_reacq = self.controller.evaluate()
        assert self.controller.state == FollowState.ACQUIRING
        assert cmd_reacq["direction"] == "stop"

        # Step 4: Next step -> resumes FOLLOWING
        self.sim_tag.simulate_uwb(tag_id="TAG_01", distance_m=2.5, direction="CENTER", connected=True)
        cmd_resume = self.controller.evaluate()
        assert self.controller.state == FollowState.FOLLOWING
        assert cmd_resume["direction"] == "forward"

        return {
            "scenario": "uwb_9_disconnect_reconnect",
            "passed": True
        }

    # =========================================================================
    # BLE SCENARIOS (1 to 7)
    # =========================================================================

    def scenario_ble_1_near(self) -> Dict[str, Any]:
        """BLE Scenario 1: Strong RSSI (-55 dBm) -> Proximity NEAR, NO forward movement."""
        self.reset()
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.tag_tracker.set_source(RadioSource.BLE)
        self.controller.set_enabled(True)

        r = self.sim_tag.simulate_ble(tag_id="BLE_01", rssi=-55.0, direction="CENTER", connected=True)
        cmd = self.controller.evaluate()

        assert not r.is_metric
        assert r.proximity_level == ProximityLevel.NEAR
        assert self.controller.decision == FollowDecision.HOLD
        assert self.controller.state == FollowState.HOLDING_DISTANCE
        assert cmd["direction"] == "stop"

        return {
            "scenario": "ble_1_near",
            "proximity_level": r.proximity_level,
            "decision": self.controller.decision,
            "passed": True
        }

    def scenario_ble_2_moderate(self) -> Dict[str, Any]:
        """BLE Scenario 2: Moderate RSSI (-74 dBm) -> Proximity MODERATE, NO forward movement."""
        self.reset()
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.tag_tracker.set_source(RadioSource.BLE)
        self.controller.set_enabled(True)

        r = self.sim_tag.simulate_ble(tag_id="BLE_01", rssi=-74.0, direction="CENTER", connected=True)
        cmd = self.controller.evaluate()

        assert r.proximity_level == ProximityLevel.MODERATE
        assert self.controller.decision == FollowDecision.HOLD
        assert cmd["direction"] == "stop"

        return {
            "scenario": "ble_2_moderate",
            "proximity_level": r.proximity_level,
            "decision": self.controller.decision,
            "passed": True
        }

    def scenario_ble_3_far(self) -> Dict[str, Any]:
        """BLE Scenario 3: Weak RSSI (-85 dBm) -> Proximity FAR, NO forward movement."""
        self.reset()
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.tag_tracker.set_source(RadioSource.BLE)
        self.controller.set_enabled(True)

        r = self.sim_tag.simulate_ble(tag_id="BLE_01", rssi=-85.0, direction="CENTER", connected=True)
        cmd = self.controller.evaluate()

        assert r.proximity_level == ProximityLevel.FAR
        assert self.controller.decision == FollowDecision.HOLD
        assert cmd["direction"] == "stop"

        return {
            "scenario": "ble_3_far",
            "proximity_level": r.proximity_level,
            "decision": self.controller.decision,
            "passed": True
        }

    def scenario_ble_4_lost(self) -> Dict[str, Any]:
        """BLE Scenario 4: Lost BLE signal / disconnected -> TARGET_LOST / STOP."""
        self.reset()
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.tag_tracker.set_source(RadioSource.BLE)
        self.controller.set_enabled(True)

        r = self.sim_tag.simulate_ble(tag_id="BLE_01", rssi=-95.0, direction="CENTER", connected=False)
        cmd = self.controller.evaluate()

        assert r.proximity_level == ProximityLevel.LOST
        assert self.controller.decision == FollowDecision.STOP
        assert self.controller.state == FollowState.TARGET_LOST

        return {
            "scenario": "ble_4_lost",
            "proximity_level": r.proximity_level,
            "decision": self.controller.decision,
            "state": self.controller.state,
            "passed": True
        }

    def scenario_ble_5_fluctuating_smoothing(self, steps: int = 8) -> Dict[str, Any]:
        """BLE Scenario 5: RSSI smoothing dampens signal fluctuations."""
        self.reset(seed=77)
        self.tag_tracker.ble_tracker.set_simulation_params(noise_std_dev_db=4.0, smoothing_alpha=0.25, seed=77)

        raw_readings = []
        smoothed_readings = []
        for _ in range(steps):
            r = self.tag_tracker.ble_tracker.record_measurement(
                tag_id="BLE_01", rssi=-65.0, direction="CENTER", connected=True
            )
            raw_readings.append(r.raw_rssi)
            smoothed_readings.append(r.smoothed_rssi)

        raw_variance = sum((x - -65.0) ** 2 for x in raw_readings) / len(raw_readings)
        smoothed_variance = sum((x - -65.0) ** 2 for x in smoothed_readings) / len(smoothed_readings)

        # Smoothed variance should be lower than raw noisy variance
        assert smoothed_variance <= raw_variance

        return {
            "scenario": "ble_5_fluctuating_smoothing",
            "raw_variance": round(raw_variance, 2),
            "smoothed_variance": round(smoothed_variance, 2),
            "passed": True
        }

    def scenario_ble_6_packet_loss(self, steps: int = 10) -> Dict[str, Any]:
        """BLE Scenario 6: Intermittent packet loss handling."""
        self.reset(seed=88)
        self.tag_tracker.ble_tracker.set_simulation_params(packet_loss_rate=0.4, seed=88)

        dropped = 0
        received = 0
        for _ in range(steps):
            r = self.tag_tracker.ble_tracker.record_measurement(
                tag_id="BLE_01", rssi=-65.0, direction="CENTER", connected=True
            )
            if r.error_reason == "PACKET_LOSS":
                dropped += 1
            else:
                received += 1

        assert dropped > 0
        assert received > 0

        return {
            "scenario": "ble_6_packet_loss",
            "dropped": dropped,
            "received": received,
            "passed": True
        }

    def scenario_ble_7_stale(self) -> Dict[str, Any]:
        """BLE Scenario 7: Stale BLE measurement -> COMMUNICATION_ERROR / STOP."""
        self.reset()
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.tag_tracker.set_source(RadioSource.BLE)
        self.controller.set_enabled(True)

        stale_ts = time.time() - 2.0
        self.tag_tracker.ble_tracker.record_measurement(
            tag_id="BLE_01", rssi=-60.0, direction="CENTER", connected=True, timestamp=stale_ts
        )
        cmd = self.controller.evaluate()

        assert self.controller.decision == FollowDecision.STOP
        assert self.controller.state == FollowState.COMMUNICATION_ERROR

        return {
            "scenario": "ble_7_stale",
            "decision": self.controller.decision,
            "state": self.controller.state,
            "passed": True
        }

    # =========================================================================
    # TAG IDENTITY & CONNECTION SCENARIOS (1 to 6)
    # =========================================================================

    def scenario_tag_1_pairing(self) -> Dict[str, Any]:
        """Tag Scenario 1: Pair a new pocket tag successfully."""
        self.reset()
        info = self.tag_tracker.pair_tag("TAG_USER_99", name="Alice Tag")
        assert info.paired
        assert info.tag_id == "TAG_USER_99"
        assert any(t["tag_id"] == "TAG_USER_99" for t in self.tag_tracker.get_paired_tags())

        return {"scenario": "tag_1_pairing", "tag": info.to_dict(), "passed": True}

    def scenario_tag_2_unpairing(self) -> Dict[str, Any]:
        """Tag Scenario 2: Unpair a tag."""
        self.reset()
        self.tag_tracker.pair_tag("TAG_TEMP", name="Temporary Tag")
        unpaired = self.tag_tracker.unpair_tag("TAG_TEMP")
        assert unpaired
        assert not any(t["tag_id"] == "TAG_TEMP" for t in self.tag_tracker.get_paired_tags())

        return {"scenario": "tag_2_unpairing", "unpaired": unpaired, "passed": True}

    def scenario_tag_3_unknown_rejected(self) -> Dict[str, Any]:
        """Tag Scenario 3: Unknown tag rejected when pairing enforced."""
        self.reset()
        self.tag_tracker.enforce_pairing = True
        self.tag_tracker.select_tag(None)

        # Unpaired tag attempts to send measurement
        self.tag_tracker.uwb_tracker.record_measurement(
            tag_id="TAG_ROGUE_77", distance_m=2.0, direction="CENTER", connected=True
        )
        current = self.tag_tracker.get_current_reading()
        assert current is None, "Unknown tag measurement must be rejected when enforce_pairing=True"

        return {"scenario": "tag_3_unknown_rejected", "rejected": True, "passed": True}

    def scenario_tag_4_invalid_rejected(self) -> Dict[str, Any]:
        """Tag Scenario 4: Invalid tag readings (negative distance) rejected."""
        self.reset()
        r = self.tag_tracker.uwb_tracker.record_measurement(
            tag_id="TAG_01", distance_m=-5.0, direction="CENTER", connected=True
        )
        assert not r.valid
        assert r.error_reason in ("INVALID_DISTANCE", "OUT_OF_RANGE")

        return {"scenario": "tag_4_invalid_rejected", "valid": r.valid, "passed": True}

    def scenario_tag_5_tag_switching(self) -> Dict[str, Any]:
        """Tag Scenario 5: Seamless tag target switching between two paired tags."""
        self.reset()
        self.tag_tracker.pair_tag("TAG_A", name="Tag Alpha")
        self.tag_tracker.pair_tag("TAG_B", name="Tag Bravo")

        # Select TAG_A
        self.tag_tracker.select_tag("TAG_A")
        self.sim_tag.simulate_uwb(tag_id="TAG_A", distance_m=2.0, direction="CENTER")
        r_a = self.tag_tracker.get_current_reading()
        assert r_a and r_a.tag_id == "TAG_A"

        # Now TAG_B emits a measurement, but TAG_A is locked
        self.sim_tag.simulate_uwb(tag_id="TAG_B", distance_m=3.0, direction="CENTER")
        r_blocked = self.tag_tracker.get_current_reading()
        assert r_blocked is None or r_blocked.tag_id == "TAG_A"

        # Switch to TAG_B
        self.tag_tracker.select_tag("TAG_B")
        self.sim_tag.simulate_uwb(tag_id="TAG_B", distance_m=3.0, direction="CENTER")
        r_b = self.tag_tracker.get_current_reading()
        assert r_b and r_b.tag_id == "TAG_B"

        return {"scenario": "tag_5_tag_switching", "passed": True}

    def scenario_tag_6_lost_and_reacquired(self) -> Dict[str, Any]:
        """Tag Scenario 6: Tag lost state tracked, safe transition on reacquisition."""
        self.reset()
        self.sim_tag.simulate_uwb(tag_id="TAG_01", distance_m=2.0, direction="CENTER")
        self.tag_tracker.get_current_reading()
        assert self.tag_tracker.tag_state == "CONNECTED"

        # Tag times out / stale
        now = time.time() + 2.0
        self.tag_tracker.get_current_reading(now=now)
        assert self.tag_tracker.tag_state == "LOST"

        # Tag re-emits reading -> state transitions through REACQUIRING
        self.sim_tag.simulate_uwb(tag_id="TAG_01", distance_m=2.0, direction="CENTER", timestamp=now + 0.1)
        self.tag_tracker.get_current_reading(now=now + 0.1)
        assert self.tag_tracker.tag_state == "REACQUIRING"

        return {"scenario": "tag_6_lost_and_reacquired", "passed": True}

    # =========================================================================
    # CONTROLLER SCENARIOS (1 to 7)
    # =========================================================================

    def scenario_controller_1_manual_override(self) -> Dict[str, Any]:
        """Controller Scenario 1: Manual override during active following immediately commands STOP."""
        self.reset()
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.tag_tracker.set_source(RadioSource.UWB)
        self.controller.set_enabled(True)
        self.sim_tag.simulate_uwb(tag_id="TAG_01", distance_m=2.5, direction="CENTER")
        self.controller.evaluate()
        assert self.controller.decision == FollowDecision.MOVE_FORWARD

        # Engage manual override
        self.controller.emergency_stop()
        assert self.controller.decision == FollowDecision.STOP
        assert self.controller.state == FollowState.MANUAL_OVERRIDE
        assert not self.controller.enabled

        return {"scenario": "controller_1_manual_override", "passed": True}

    def scenario_controller_2_emergency_stop_following(self) -> Dict[str, Any]:
        """Controller Scenario 2: Emergency STOP while following halts rover completely."""
        self.reset()
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.controller.set_enabled(True)
        self.sim_tag.simulate_uwb(tag_id="TAG_01", distance_m=3.0, direction="CENTER")
        self.controller.evaluate()

        cmd = self.controller.emergency_stop()
        assert cmd["direction"] == "stop"
        assert cmd["speed"] == 0
        assert cmd["sent"] is False  # Safety invariant

        return {"scenario": "controller_2_emergency_stop_following", "passed": True}

    def scenario_controller_3_emergency_stop_tag_loss(self) -> Dict[str, Any]:
        """Controller Scenario 3: Emergency STOP during tag loss maintains STOP."""
        self.reset()
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.controller.set_enabled(True)
        self.sim_tag.simulate_uwb(tag_id="TAG_01", distance_m=2.0, direction="CENTER", connected=False)
        self.controller.evaluate()
        assert self.controller.state == FollowState.TARGET_LOST

        self.controller.emergency_stop()
        assert self.controller.decision == FollowDecision.STOP
        assert self.controller.state == FollowState.MANUAL_OVERRIDE

        return {"scenario": "controller_3_emergency_stop_tag_loss", "passed": True}

    def scenario_controller_4_mode_switch(self) -> Dict[str, Any]:
        """Controller Scenario 4: Mode switching resets targets and halts previous mode safely."""
        self.reset()
        self.controller.set_mode(FollowMode.PERSON)
        self.controller.set_enabled(True)
        self.controller.select_person_target(3)

        # Switch to POCKET_TAG
        self.controller.set_mode(FollowMode.POCKET_TAG)
        assert self.controller.mode == FollowMode.POCKET_TAG
        assert not self.controller.enabled
        assert self.controller.decision == FollowDecision.STOP
        assert self.person_tracker.selected_person_id is None

        return {"scenario": "controller_4_mode_switch", "passed": True}

    def scenario_controller_5_uwb_without_direction(self) -> Dict[str, Any]:
        """Controller Scenario 5: Valid UWB distance without direction produces HOLD, never MOVE_FORWARD."""
        self.reset()
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.tag_tracker.set_source(RadioSource.UWB)
        self.controller.set_enabled(True)

        self.tag_tracker.uwb_tracker.record_measurement(
            tag_id="TAG_01", distance_m=2.8, direction=None, connected=True
        )
        cmd = self.controller.evaluate()
        assert self.controller.decision == FollowDecision.HOLD
        assert cmd["direction"] == "stop"

        return {"scenario": "controller_5_uwb_without_direction", "passed": True}

    def scenario_controller_6_ble_only_proximity(self) -> Dict[str, Any]:
        """Controller Scenario 6: BLE-only proximity never generates forward movement."""
        self.reset()
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.tag_tracker.set_source(RadioSource.BLE)
        self.controller.set_enabled(True)

        self.sim_tag.simulate_ble(tag_id="BLE_01", rssi=-50.0, direction="CENTER", connected=True)
        cmd = self.controller.evaluate()
        assert self.controller.decision == FollowDecision.HOLD
        assert cmd["direction"] == "stop"

        return {"scenario": "controller_6_ble_only_proximity", "passed": True}

    def scenario_controller_7_invalid_or_stale(self) -> Dict[str, Any]:
        """Controller Scenario 7: Invalid or stale tag data results in STOP."""
        self.reset()
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.tag_tracker.set_source(RadioSource.UWB)
        self.controller.set_enabled(True)

        # Stale reading
        self.tag_tracker.uwb_tracker.record_measurement(
            tag_id="TAG_01", distance_m=2.0, direction="CENTER", connected=True, timestamp=time.time() - 3.0
        )
        cmd = self.controller.evaluate()
        assert self.controller.decision == FollowDecision.STOP
        assert self.controller.state == FollowState.COMMUNICATION_ERROR

        return {"scenario": "controller_7_invalid_or_stale", "passed": True}

    # =========================================================================
    # BATCH RUNNER
    # =========================================================================

    def run_all_scenarios(self) -> Dict[str, Any]:
        """Executes all 29 deterministic scenarios and returns a full report."""
        results = {}

        # 9 UWB scenarios
        results["uwb_1_no_direction"] = self.scenario_uwb_1_no_direction()
        results["uwb_2_with_direction"] = self.scenario_uwb_2_with_direction()
        results["uwb_3_holding_distance"] = self.scenario_uwb_3_holding_distance()
        results["uwb_4_close_safety_stop"] = self.scenario_uwb_4_close_safety_stop()
        results["uwb_5_fluctuating_noise"] = self.scenario_uwb_5_fluctuating_noise()
        results["uwb_6_packet_loss"] = self.scenario_uwb_6_packet_loss()
        results["uwb_7_stale_measurement"] = self.scenario_uwb_7_stale_measurement()
        results["uwb_8_out_of_range"] = self.scenario_uwb_8_out_of_range()
        results["uwb_9_disconnect_reconnect"] = self.scenario_uwb_9_disconnect_reconnect()

        # 7 BLE scenarios
        results["ble_1_near"] = self.scenario_ble_1_near()
        results["ble_2_moderate"] = self.scenario_ble_2_moderate()
        results["ble_3_far"] = self.scenario_ble_3_far()
        results["ble_4_lost"] = self.scenario_ble_4_lost()
        results["ble_5_fluctuating_smoothing"] = self.scenario_ble_5_fluctuating_smoothing()
        results["ble_6_packet_loss"] = self.scenario_ble_6_packet_loss()
        results["ble_7_stale"] = self.scenario_ble_7_stale()

        # 6 Tag Management scenarios
        results["tag_1_pairing"] = self.scenario_tag_1_pairing()
        results["tag_2_unpairing"] = self.scenario_tag_2_unpairing()
        results["tag_3_unknown_rejected"] = self.scenario_tag_3_unknown_rejected()
        results["tag_4_invalid_rejected"] = self.scenario_tag_4_invalid_rejected()
        results["tag_5_tag_switching"] = self.scenario_tag_5_tag_switching()
        results["tag_6_lost_and_reacquired"] = self.scenario_tag_6_lost_and_reacquired()

        # 7 Controller scenarios
        results["controller_1_manual_override"] = self.scenario_controller_1_manual_override()
        results["controller_2_emergency_stop_following"] = self.scenario_controller_2_emergency_stop_following()
        results["controller_3_emergency_stop_tag_loss"] = self.scenario_controller_3_emergency_stop_tag_loss()
        results["controller_4_mode_switch"] = self.scenario_controller_4_mode_switch()
        results["controller_5_uwb_without_direction"] = self.scenario_controller_5_uwb_without_direction()
        results["controller_6_ble_only_proximity"] = self.scenario_controller_6_ble_only_proximity()
        results["controller_7_invalid_or_stale"] = self.scenario_controller_7_invalid_or_stale()

        total = len(results)
        passed = sum(1 for r in results.values() if r.get("passed"))
        return {
            "total_scenarios": total,
            "passed_scenarios": passed,
            "failed_scenarios": total - passed,
            "all_passed": (total == passed),
            "results": results
        }

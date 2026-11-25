"""
Unit and integration tests for WALL-E Phase 3:
Advanced Pocket Tag Tracking & UWB/BLE Simulation.
"""

import os
import sys
import time
import unittest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(BASE_DIR, "walle_software_starter")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from tag_tracker import (
    TagTrackerManager, UWBTagTracker, BLETagTracker, RadioSource,
    TagReading, SimulatedTagDevice, ProximityLevel, DirectionSource,
    ManualDirectionProvider, AoADirectionProvider
)
from follow_controller import (
    FollowController, FollowMode, FollowState, FollowDecision
)
from person_tracker import PersonTracker, PersonObservation
from tag_simulation import DeterministicTagSimulator
from fastapi import HTTPException
import main


class TestAdvancedUWBRanging(unittest.TestCase):
    """Verifies UWB ranging, signal quality, packet loss, and single-receiver safety."""

    def setUp(self):
        self.tracker = UWBTagTracker(min_distance=0.1, max_distance=50.0, stale_timeout_sec=0.8)
        self.manager = TagTrackerManager(default_source=RadioSource.UWB)
        self.controller = FollowController(
            tag_tracker=self.manager,
            target_distance_m=1.5,
            distance_tolerance_m=0.3,
            min_safe_distance_m=0.8,
            approach_threshold_m=1.8,
            hold_min_m=1.2,
            hold_max_m=1.8,
            stale_timeout_sec=0.8
        )
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.controller.set_enabled(True)

    def test_metric_distance_and_signal_quality(self):
        """UWB provides calibrated metric distance and quality based on range."""
        r_close = self.tracker.record_measurement(tag_id="TAG_01", distance_m=1.5, direction=None)
        self.assertTrue(r_close.is_metric)
        self.assertEqual(r_close.distance_m, 1.5)
        self.assertGreater(r_close.signal_quality, 90.0)

        # Far distance degrades quality naturally
        r_far = self.tracker.record_measurement(tag_id="TAG_01", distance_m=35.0, direction=None)
        self.assertTrue(r_far.valid)
        self.assertLess(r_far.signal_quality, r_close.signal_quality)

    def test_out_of_range_detection(self):
        """Distances beyond max_distance or below min_distance are marked out of range."""
        r_over = self.tracker.record_measurement(tag_id="TAG_01", distance_m=55.0)
        self.assertFalse(r_over.valid)
        self.assertEqual(r_over.error_reason, "OUT_OF_RANGE")

        r_under = self.tracker.record_measurement(tag_id="TAG_01", distance_m=0.05)
        self.assertFalse(r_under.valid)
        self.assertEqual(r_under.error_reason, "OUT_OF_RANGE")

    def test_packet_loss_simulation(self):
        """Packet loss simulation deterministically drops packets when loss rate > 0."""
        self.tracker.set_simulation_params(packet_loss_rate=0.5, seed=42)
        readings = [
            self.tracker.record_measurement(tag_id="TAG_01", distance_m=2.0)
            for _ in range(12)
        ]
        dropped = [r for r in readings if r.error_reason == "PACKET_LOSS"]
        valid = [r for r in readings if r.valid]
        self.assertGreater(len(dropped), 0)
        self.assertGreater(len(valid), 0)

    def test_single_receiver_limitation_never_moves_forward_without_direction(self):
        """CRITICAL: Single UWB range measurement must NEVER trigger forward or turn movement."""
        self.manager.uwb_tracker.record_measurement(
            tag_id="TAG_01", distance_m=2.5, direction=None, connected=True
        )
        cmd = self.controller.evaluate()
        self.assertEqual(self.controller.decision, FollowDecision.HOLD)
        self.assertEqual(self.controller.state, FollowState.HOLDING_DISTANCE)
        self.assertEqual(cmd["direction"], "stop")
        self.assertIn("directional source unavailable", cmd["reason"])

    def test_movement_allowed_with_independent_directional_source(self):
        """Forward approach is enabled only when valid metric distance AND valid direction exist."""
        # Using manual external direction
        self.manager.set_external_direction("CENTER")
        self.manager.uwb_tracker.record_measurement(
            tag_id="TAG_01", distance_m=2.5, direction=None, connected=True
        )
        cmd = self.controller.evaluate()
        self.assertEqual(self.controller.decision, FollowDecision.MOVE_FORWARD)
        self.assertEqual(self.controller.state, FollowState.FOLLOWING)
        self.assertEqual(cmd["direction"], "forward")

    def test_aoa_directional_provider(self):
        """AoA provider resolves azimuth degrees to discrete steering directions."""
        aoa = AoADirectionProvider(deadzone_deg=15.0)
        self.manager.set_directional_provider(aoa)

        aoa.set_azimuth(-25.0)  # Left
        self.manager.uwb_tracker.record_measurement(tag_id="TAG_01", distance_m=2.5, direction=None)
        cmd = self.controller.evaluate()
        self.assertEqual(self.controller.decision, FollowDecision.TURN_LEFT)

        aoa.set_azimuth(30.0)  # Right
        self.manager.uwb_tracker.record_measurement(tag_id="TAG_01", distance_m=2.5, direction=None)
        cmd = self.controller.evaluate()
        self.assertEqual(self.controller.decision, FollowDecision.TURN_RIGHT)

        aoa.set_azimuth(0.0)  # Center
        self.manager.uwb_tracker.record_measurement(tag_id="TAG_01", distance_m=2.5, direction=None)
        cmd = self.controller.evaluate()
        self.assertEqual(self.controller.decision, FollowDecision.MOVE_FORWARD)


class TestAdvancedBLEProximity(unittest.TestCase):
    """Verifies BLE proximity classifications, RSSI smoothing, and non-metric safety."""

    def setUp(self):
        self.tracker = BLETagTracker(stale_timeout_sec=0.8, near_threshold_rssi=-68.0, moderate_threshold_rssi=-80.0)
        self.manager = TagTrackerManager(default_source=RadioSource.BLE)
        self.controller = FollowController(tag_tracker=self.manager, stale_timeout_sec=0.8)
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.controller.set_enabled(True)

    def test_proximity_classifications(self):
        """BLE classifications: NEAR, MODERATE, FAR, LOST."""
        r_near = self.tracker.record_measurement(tag_id="BLE_01", rssi=-55.0)
        self.assertEqual(r_near.proximity_level, ProximityLevel.NEAR)

        r_mod = self.tracker.record_measurement(tag_id="BLE_01", rssi=-74.0)
        self.assertEqual(r_mod.proximity_level, ProximityLevel.MODERATE)

        r_far = self.tracker.record_measurement(tag_id="BLE_01", rssi=-88.0)
        self.assertEqual(r_far.proximity_level, ProximityLevel.FAR)

        r_lost = self.tracker.record_measurement(tag_id="BLE_01", rssi=-95.0, connected=False)
        self.assertEqual(r_lost.proximity_level, ProximityLevel.LOST)

    def test_ble_is_never_metric_distance(self):
        """BLE readings are strictly is_metric=False and never produce MOVE_FORWARD."""
        r = self.tracker.record_measurement(tag_id="BLE_01", rssi=-50.0, direction="CENTER")
        self.assertFalse(r.is_metric)

        self.manager.ble_tracker.record_measurement(tag_id="BLE_01", rssi=-50.0, direction="CENTER")
        cmd = self.controller.evaluate()
        self.assertNotEqual(self.controller.decision, FollowDecision.MOVE_FORWARD)
        self.assertEqual(self.controller.decision, FollowDecision.HOLD)
        self.assertEqual(cmd["direction"], "stop")

    def test_ble_rssi_smoothing(self):
        """Exponential smoothing dampens noisy raw RSSI values."""
        self.tracker.set_simulation_params(smoothing_alpha=0.2)
        # Sequence of fluctuating raw readings
        raws = [-60.0, -85.0, -55.0, -80.0]
        smoothed = []
        for v in raws:
            r = self.tracker.record_measurement(tag_id="BLE_01", rssi=v)
            smoothed.append(r.smoothed_rssi)

        # Smoothed values should lag and moderate rapid fluctuations
        self.assertGreater(smoothed[1], -85.0)
        self.assertLess(smoothed[2], -55.0)


class TestTagIdentityAndConnectionManagement(unittest.TestCase):
    """Verifies pairing, tag authentication, filtering, and reacquisition states."""

    def setUp(self):
        self.manager = TagTrackerManager(default_source=RadioSource.UWB)

    def test_pairing_and_unpairing(self):
        """Tags can be paired, queried, and unpaired."""
        info = self.manager.pair_tag("TAG_ALICE", name="Alice Pocket Tag", authenticated=True)
        self.assertTrue(info.paired)
        self.assertTrue(info.authenticated)
        self.assertEqual(info.name, "Alice Pocket Tag")

        paired = self.manager.get_paired_tags()
        self.assertTrue(any(t["tag_id"] == "TAG_ALICE" for t in paired))

        unpaired = self.manager.unpair_tag("TAG_ALICE")
        self.assertTrue(unpaired)
        paired_after = self.manager.get_paired_tags()
        self.assertFalse(any(t["tag_id"] == "TAG_ALICE" for t in paired_after))

    def test_rejection_of_unauthorized_tags_when_pairing_enforced(self):
        """Unpaired or unknown tags cannot control the rover when enforce_pairing is active."""
        self.manager.enforce_pairing = True
        self.manager.select_tag(None)

        # Authorized paired tag
        self.manager.pair_tag("TAG_AUTH", authenticated=True)
        self.manager.uwb_tracker.record_measurement(tag_id="TAG_AUTH", distance_m=2.0, direction="CENTER")
        r_auth = self.manager.get_current_reading()
        self.assertIsNotNone(r_auth)
        self.assertEqual(r_auth.tag_id, "TAG_AUTH")

        # Rogue unauthenticated tag
        self.manager.uwb_tracker.record_measurement(tag_id="TAG_ROGUE", distance_m=1.0, direction="CENTER")
        r_rogue = self.manager.get_current_reading()
        self.assertIsNone(r_rogue)

    def test_tag_selection_filters_unselected_tags(self):
        """When a tag is locked by ID, measurements from other tags are ignored."""
        self.manager.select_tag("TAG_PRIMARY")

        self.manager.uwb_tracker.record_measurement(tag_id="TAG_SECONDARY", distance_m=3.0, direction="CENTER")
        self.assertIsNone(self.manager.get_current_reading())

        self.manager.uwb_tracker.record_measurement(tag_id="TAG_PRIMARY", distance_m=2.1, direction="CENTER")
        r = self.manager.get_current_reading()
        self.assertIsNotNone(r)
        self.assertEqual(r.tag_id, "TAG_PRIMARY")

    def test_battery_and_connection_tracking(self):
        """Tag battery and connection states are tracked and reported in status."""
        self.manager.pair_tag("TAG_BATT", name="Battery Tag")
        self.manager.uwb_tracker.record_measurement(
            tag_id="TAG_BATT", distance_m=2.0, battery_pct=78.5, connected=True
        )
        r = self.manager.get_current_reading()
        self.assertEqual(r.battery_pct, 78.5)

        status = self.manager.get_status()
        self.assertTrue(status["connected"])
        self.assertEqual(status["reading"]["battery_pct"], 78.5)


class TestDeterministicSimulationScenarios(unittest.TestCase):
    """Executes the full suite of 29 deterministic simulation scenarios."""

    def test_all_29_scenarios_pass_repeatably(self):
        sim = DeterministicTagSimulator(seed=42)
        report = sim.run_all_scenarios()
        self.assertEqual(report["total_scenarios"], 29)
        self.assertEqual(report["passed_scenarios"], 29)
        self.assertEqual(report["failed_scenarios"], 0)
        self.assertTrue(report["all_passed"])

    def test_fastapi_tag_endpoints(self):
        """Verifies new Phase 3 FastAPI endpoints."""
        # 1. Pair tag endpoint
        p_res = main.pair_or_unpair_tag(main.TagPairCommand(tag_id="TAG_API_TEST", name="API Tag", action="pair"))
        self.assertTrue(p_res["ok"])
        self.assertEqual(p_res["tag"]["tag_id"], "TAG_API_TEST")

        # 2. Tag list endpoint
        l_res = main.list_follow_tags()
        self.assertTrue(l_res["ok"])
        self.assertTrue(any(t["tag_id"] == "TAG_API_TEST" for t in l_res["tags"]))

        # 3. Tag direction endpoint
        d_res = main.set_follow_tag_direction(main.TagDirectionCommand(direction="RIGHT", source="CAMERA"))
        self.assertTrue(d_res["ok"])
        self.assertEqual(main.follow_controller.tag_tracker.external_direction, "RIGHT")

        # 4. Simulation scenario runner endpoint
        s_res = main.run_tag_scenario(main.TagScenarioCommand(scenario="all", seed=42))
        self.assertTrue(s_res["ok"])
        self.assertTrue(s_res["report"]["all_passed"])

        # 5. Unpair tag endpoint
        u_res = main.pair_or_unpair_tag(main.TagPairCommand(tag_id="TAG_API_TEST", action="unpair"))
        self.assertTrue(u_res["ok"])
        self.assertTrue(u_res["unpaired"])


class TestPhase32SafetyArchitecture(unittest.TestCase):
    """
    Verification of Phase 3.2 Directional Data TTL & Emergency STOP Latch Architecture.
    """
    def setUp(self):
        self.manager = TagTrackerManager(default_source=RadioSource.UWB, direction_ttl_sec=0.5)
        self.controller = FollowController(
            tag_tracker=self.manager,
            target_distance_m=1.5,
            approach_threshold_m=1.8,
            min_safe_distance_m=0.8
        )
        self.controller.set_mode(FollowMode.POCKET_TAG)
        self.controller.select_tag_target("TAG_01", "UWB")
        self.controller.set_enabled(True)

    def test_fresh_direction_accepted(self):
        """Fresh direction within TTL authorizes forward movement when range > 1.8m."""
        self.manager.set_external_direction("CENTER", tag_id="TAG_01", timestamp=100.0, ttl_sec=0.5)
        self.manager.uwb_tracker.record_measurement(tag_id="TAG_01", distance_m=2.5, direction=None, timestamp=100.1)
        cmd = self.controller.evaluate(now=100.2)
        self.assertEqual(cmd["decision"], FollowDecision.MOVE_FORWARD)
        self.assertEqual(cmd["direction"], "forward")
        self.assertEqual(self.controller.state, FollowState.FOLLOWING)

    def test_expired_direction_rejected(self):
        """Direction measurement older than TTL is strictly rejected, resulting in HOLD."""
        self.manager.set_external_direction("CENTER", tag_id="TAG_01", timestamp=100.0, ttl_sec=0.5)
        self.manager.uwb_tracker.record_measurement(tag_id="TAG_01", distance_m=2.5, direction=None, timestamp=100.1)
        # Evaluated at 100.6s (age = 0.6s > 0.5s TTL)
        cmd = self.controller.evaluate(now=100.6)
        self.assertEqual(cmd["decision"], FollowDecision.HOLD)
        self.assertEqual(cmd["direction"], "stop")
        self.assertEqual(self.controller.state, FollowState.HOLDING_DISTANCE)

    def test_direction_timestamp_not_refreshed_by_uwb_packets(self):
        """Subsequent fresh UWB range packets must NOT refresh an older direction timestamp."""
        self.manager.set_external_direction("RIGHT", tag_id="TAG_01", timestamp=100.0, ttl_sec=0.5)
        # Packet 1 at 100.1 (fresh)
        self.manager.uwb_tracker.record_measurement(tag_id="TAG_01", distance_m=2.5, direction=None, timestamp=100.1)
        cmd1 = self.controller.evaluate(now=100.2)
        self.assertEqual(cmd1["decision"], FollowDecision.TURN_RIGHT)

        # 5 fresh UWB packets at 100.6, 100.7, 100.8...
        for t in [100.6, 100.7, 100.8, 100.9]:
            self.manager.uwb_tracker.record_measurement(tag_id="TAG_01", distance_m=2.5, direction=None, timestamp=t)
            cmd = self.controller.evaluate(now=t)
            # Must remain HOLD because direction expired at 100.5s
            self.assertEqual(cmd["decision"], FollowDecision.HOLD)
            self.assertEqual(cmd["speed"], 0)

    def test_direction_from_another_tag_rejected(self):
        """Direction scoped to TAG_02 must NOT be applied to TAG_01."""
        self.manager.set_external_direction("CENTER", tag_id="TAG_02", timestamp=100.0, ttl_sec=0.5)
        self.manager.uwb_tracker.record_measurement(tag_id="TAG_01", distance_m=2.5, direction=None, timestamp=100.1)
        cmd = self.controller.evaluate(now=100.1)
        self.assertEqual(cmd["decision"], FollowDecision.HOLD)
        self.assertEqual(self.controller.state, FollowState.HOLDING_DISTANCE)

    def test_direction_explicitly_cleared(self):
        """Explicitly cleared direction immediately reverts to HOLD."""
        self.manager.set_external_direction("CENTER", tag_id="TAG_01", timestamp=100.0, ttl_sec=0.5)
        self.manager.uwb_tracker.record_measurement(tag_id="TAG_01", distance_m=2.5, direction=None, timestamp=100.1)
        cmd_before = self.controller.evaluate(now=100.1)
        self.assertEqual(cmd_before["decision"], FollowDecision.MOVE_FORWARD)

        # Clear direction
        self.manager.clear_external_direction(tag_id="TAG_01")
        cmd_after = self.controller.evaluate(now=100.1)
        self.assertEqual(cmd_after["decision"], FollowDecision.HOLD)

    def test_uwb_range_without_fresh_direction_produces_hold(self):
        """UWB range above 1.8m without fresh direction always produces HOLD and speed 0."""
        self.manager.uwb_tracker.record_measurement(tag_id="TAG_01", distance_m=3.5, direction=None, timestamp=100.0)
        cmd = self.controller.evaluate(now=100.0)
        self.assertEqual(cmd["decision"], FollowDecision.HOLD)
        self.assertEqual(cmd["speed"], 0)
        self.assertEqual(self.controller.state, FollowState.HOLDING_DISTANCE)

    def test_ble_cannot_produce_forward_movement(self):
        """BLE tags must never produce MOVE_FORWARD even with fresh CENTER direction."""
        self.controller.select_tag_target("TAG_BLE_01", "BLE")
        self.manager.set_external_direction("CENTER", tag_id="TAG_BLE_01", timestamp=100.0, ttl_sec=0.5)
        self.manager.ble_tracker.record_measurement(tag_id="TAG_BLE_01", rssi=-50.0, direction=None, timestamp=100.1)
        cmd = self.controller.evaluate(now=100.1)
        self.assertNotEqual(cmd["decision"], FollowDecision.MOVE_FORWARD)
        self.assertEqual(cmd["decision"], FollowDecision.HOLD)
        self.assertEqual(cmd["speed"], 0)

    def test_emergency_stop_latch_survives_sensor_updates(self):
        """Emergency STOP latch cannot be cleared by subsequent sensor packets."""
        self.controller.emergency_stop()
        self.assertTrue(self.controller.emergency_stop_latched)
        self.assertFalse(self.controller.enabled)

        # Ingest sensor packets
        self.manager.set_external_direction("CENTER", tag_id="TAG_01", timestamp=100.0)
        self.manager.uwb_tracker.record_measurement(tag_id="TAG_01", distance_m=2.5, direction="CENTER", timestamp=100.1)
        cmd = self.controller.evaluate(now=100.1)
        self.assertEqual(cmd["decision"], FollowDecision.STOP)
        self.assertEqual(self.controller.state, FollowState.MANUAL_OVERRIDE)
        self.assertTrue(self.controller.emergency_stop_latched)

    def test_emergency_stop_latch_survives_mode_switching(self):
        """Switching modes while latched maintains the emergency STOP latch."""
        self.controller.emergency_stop()
        self.assertTrue(self.controller.emergency_stop_latched)

        # Switch to PERSON mode
        self.controller.set_mode(FollowMode.PERSON)
        self.assertTrue(self.controller.emergency_stop_latched)
        self.assertFalse(self.controller.enabled)
        self.assertEqual(self.controller.state, FollowState.MANUAL_OVERRIDE)

        # Switch to MANUAL mode
        self.controller.set_mode(FollowMode.MANUAL)
        self.assertTrue(self.controller.emergency_stop_latched)

    def test_emergency_stop_latch_survives_follow_enable_requests(self):
        """Re-enabling follow is strictly rejected while latched."""
        self.controller.emergency_stop()
        self.assertTrue(self.controller.emergency_stop_latched)

        self.controller.set_enabled(True)
        self.assertFalse(self.controller.enabled)
        self.assertTrue(self.controller.emergency_stop_latched)
        self.assertEqual(self.controller.state, FollowState.MANUAL_OVERRIDE)

    def test_only_explicit_reset_clears_the_latch(self):
        """Only reset_emergency_stop() clears the latch and returns controller to IDLE."""
        self.controller.emergency_stop()
        self.assertTrue(self.controller.emergency_stop_latched)

        self.controller.reset_emergency_stop()
        self.assertFalse(self.controller.emergency_stop_latched)
        self.assertEqual(self.controller.state, FollowState.IDLE)
        self.assertEqual(self.controller.decision, FollowDecision.STOP)
        self.assertFalse(self.controller.enabled)

    def test_following_does_not_automatically_resume_after_reset(self):
        """Reset leaves follow disabled; explicit re-enable & target selection required."""
        self.controller.emergency_stop()
        self.controller.reset_emergency_stop()
        self.assertFalse(self.controller.enabled)

        # Sensor packets arrive after reset
        self.manager.uwb_tracker.record_measurement(tag_id="TAG_01", distance_m=2.5, direction="CENTER")
        cmd = self.controller.evaluate()
        self.assertEqual(cmd["decision"], FollowDecision.STOP)
        self.assertEqual(self.controller.state, FollowState.IDLE)

        # Requires explicit target selection & enable
        self.controller.select_tag_target("TAG_01", "UWB")
        self.controller.set_enabled(True)
        self.manager.set_external_direction("CENTER", tag_id="TAG_01")
        self.manager.uwb_tracker.record_measurement(tag_id="TAG_01", distance_m=2.5, direction=None)
        cmd2 = self.controller.evaluate()
        self.assertEqual(cmd2["decision"], FollowDecision.MOVE_FORWARD)

    def test_person_following_regression(self):
        """Person-following aligns with LEFT/RIGHT and holds on CENTER."""
        self.controller.set_mode(FollowMode.PERSON)
        self.controller.set_enabled(True)
        p = PersonObservation(person_id=1, bbox=(50, 100, 150, 300), confidence=0.95, direction="LEFT")
        self.controller.person_tracker.update([p])
        cmd = self.controller.evaluate()
        self.assertEqual(cmd["decision"], FollowDecision.TURN_LEFT)

    def test_existing_tag_pairing_and_reacquisition_regression(self):
        """Tag pairing and reacquisition flow cleanly through ACQUIRING to CONNECTED."""
        self.manager.enforce_pairing = True
        self.manager.pair_tag("TAG_NEW", "New Tag")
        self.assertTrue(self.manager.is_tag_authorized("TAG_NEW"))
        self.assertFalse(self.manager.is_tag_authorized("TAG_UNAUTH"))

    def test_rest_api_safety_endpoints(self):
        """Verifies REST endpoints for safety reset, latch status, and direction clear."""
        # Test stop engages latch
        main.stop()
        self.assertTrue(main.follow_controller.emergency_stop_latched)

        # Test enable rejected with HTTP 400 while latched
        with self.assertRaises(HTTPException) as ctx:
            main.set_follow_enable(main.FollowEnableCommand(enabled=True))
        self.assertEqual(ctx.exception.status_code, 400)

        # Test reset clears latch
        r_res = main.reset_safety_latch()
        self.assertTrue(r_res["ok"])
        self.assertFalse(r_res["emergency_stop_latched"])
        self.assertFalse(main.follow_controller.emergency_stop_latched)

        # Test direction with TTL and clear endpoint
        d_res = main.set_follow_tag_direction(main.TagDirectionCommand(direction="LEFT", ttl_sec=0.4))
        self.assertTrue(d_res["ok"])
        c_res = main.clear_follow_tag_direction()
        self.assertTrue(c_res["ok"])
        self.assertIsNone(main.follow_controller.tag_tracker.external_direction)


if __name__ == "__main__":
    unittest.main()

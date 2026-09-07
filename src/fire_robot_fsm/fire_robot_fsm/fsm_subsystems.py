from __future__ import annotations

import math
from typing import Any


class DoorTargetSubFsm:
    """Door target selection and final blue-evidence policy.

    The main FSM owns mission state transitions. This sub-FSM owns the narrower
    question "which blue door candidate is currently actionable?" so false door
    opens can be debugged without touching the whole mission flow.
    """

    def __init__(self, node: Any):
        self.node = node

    def explore_blue_candidates(self):
        n = self.node
        blue_candidates = [
            d for d in n.detected_doors
            if d.door_color == 'blue'
            and not n._is_door_opened_for_observation(d)
            and not n._is_door_abandoned_for_observation(d)
            and n._is_door_recent_observation(d)
            and not n._is_door_physically_opened(d)
            and not n._is_door_observation_suppressed(d)
            and not n._is_door_physically_abandoned(d)
            and not n._is_blue_candidate_blocked_by_red(d)
            and n._is_blue_door_at_valid_wall_position(d)
            and not n._direct_blue_target_handle_invalid_reason(d)
            and n._is_blue_door_target_ready(d)
            and n._is_blue_door_ready_to_interrupt_explore(d)
        ]
        raw_safe_doors = [
            d for d in blue_candidates
            if not n._is_door_failed_for_observation(d)
        ]
        if n._use_observed_blue_clusters_as_targets:
            observed_targets = n._stable_unopened_observed_blue_targets()
            raw_safe_doors = n._merge_blue_target_lists(
                raw_safe_doors, observed_targets)

        safe_doors = [
            d for d in raw_safe_doors
            if n._door_has_map_identity(d)
            and n._blue_door_survives_axis_adjusted_opened_filter(d)
        ]
        return blue_candidates, raw_safe_doors, safe_doors

    def live_blue_evidence_matches_target(
            self, door, candidate, target_points, candidate_points,
            max_dist: float) -> bool:
        n = self.node
        if not door.door_id.startswith('observed_blue_'):
            return self._direct_blue_evidence_matches_target(
                door, candidate, max_dist)

        target_wall = n._door_handle_xy(door)
        candidate_wall = n._door_handle_xy(candidate)
        candidate_raw_wall = n._raw_blue_handle_xy_for_evidence(candidate)
        if (
                candidate_raw_wall is not None
                and math.hypot(
                    target_wall[0] - candidate_raw_wall[0],
                    target_wall[1] - candidate_raw_wall[1])
                < math.hypot(
                    target_wall[0] - candidate_wall[0],
                    target_wall[1] - candidate_wall[1])):
            candidate_wall = candidate_raw_wall
        target_progress = n._axis_progress_xy(target_wall[0], target_wall[1])
        candidate_progress = n._axis_progress_xy(candidate_wall[0], candidate_wall[1])
        target_lateral = n._axis_lateral_xy(target_wall[0], target_wall[1])
        candidate_lateral = n._axis_lateral_xy(candidate_wall[0], candidate_wall[1])
        progress_gap = abs(candidate_progress - target_progress)
        lateral_gap = abs(candidate_lateral - target_lateral)

        opening_aligned = n._blue_target_open_pose_aligned_for_safe_memory(door)
        open_progress_cap = 1.15 if opening_aligned else 1.35
        open_lateral_cap = 0.95 if opening_aligned else 1.10
        if not bool(getattr(door, 'handle_detected', False)):
            open_progress_cap = min(open_progress_cap, 1.05)
            open_lateral_cap = min(open_lateral_cap, 0.90)
        max_progress_gap = max(
            0.20,
            min(
                getattr(n, '_observed_blue_fresh_evidence_max_progress_gap_m', 0.95),
                open_progress_cap))
        max_lateral_gap = max(
            0.20,
            min(
                getattr(n, '_observed_blue_fresh_evidence_max_lateral_gap_m', 0.85),
                open_lateral_cap))
        target_xy = n._door_identity_xy(door)
        candidate_xy = n._door_identity_xy(candidate)
        pre_exit_station_match = False
        pre_exit_strict = n._blue_xy_requires_pre_nav_exit_fresh_evidence(
            target_xy)
        if pre_exit_strict:
            identity_progress = n._axis_progress_xy(target_xy[0], target_xy[1])
            candidate_identity_progress = n._axis_progress_xy(
                candidate_xy[0], candidate_xy[1])
            identity_lateral = n._axis_lateral_xy(target_xy[0], target_xy[1])
            candidate_identity_lateral = n._axis_lateral_xy(
                candidate_xy[0], candidate_xy[1])
            identity_dist = math.hypot(
                target_xy[0] - candidate_xy[0],
                target_xy[1] - candidate_xy[1])
            wall_dist = math.hypot(
                target_wall[0] - candidate_wall[0],
                target_wall[1] - candidate_wall[1])
            strict_progress_gap = max(
                0.20,
                min(
                    max_progress_gap,
                    max(0.70, n._observed_physical_door_merge_dist_m + 0.80)))
            strict_lateral_gap = max(
                0.20,
                min(
                    max_lateral_gap,
                    max(0.65, n._axis_door_side_standoff_m + 0.75)))
            station_dist_limit = max(
                0.45,
                min(
                    max_dist,
                    math.hypot(strict_progress_gap, strict_lateral_gap)))
            identity_match = (
                identity_dist <= station_dist_limit
                and abs(candidate_identity_progress - identity_progress)
                <= strict_progress_gap
                and abs(candidate_identity_lateral - identity_lateral)
                <= strict_lateral_gap)
            wall_match = (
                wall_dist <= station_dist_limit
                and progress_gap <= strict_progress_gap
                and lateral_gap <= strict_lateral_gap)
            if not (identity_match or wall_match):
                n.get_logger().info(
                    f'Fresh blue evidence rejected for {door.door_id}: '
                    f'pre-exit live candidate {candidate.door_id} is not the '
                    f'same physical station '
                    f'(identity_dist={identity_dist:.2f}/{station_dist_limit:.2f}, '
                    f'wall_dist={wall_dist:.2f}/{station_dist_limit:.2f}, '
                    f'progress_gap={progress_gap:.2f}/{strict_progress_gap:.2f}, '
                    f'lateral_gap={lateral_gap:.2f}/{strict_lateral_gap:.2f}).',
                    throttle_duration_sec=3.0)
                return False
            pre_exit_station_match = True

        target_side = target_lateral - n._explore_center_y
        candidate_side = candidate_lateral - n._explore_center_y
        if (
                abs(target_side) > 0.35
                and abs(candidate_side) > 0.35
                and target_side * candidate_side < 0.0):
            n.get_logger().info(
                f'Fresh blue evidence rejected for {door.door_id}: live '
                f'candidate {candidate.door_id} is on the opposite wall.',
                throttle_duration_sec=3.0)
            return False

        if (
                self._points_near_opened_station(candidate_points)
                and not self._points_near_opened_station([target_wall])):
            n.get_logger().info(
                f'Fresh blue evidence rejected for {door.door_id}: live '
                f'candidate {candidate.door_id} belongs to an already opened '
                'door station.',
                throttle_duration_sec=3.0)
            return False

        if progress_gap > max_progress_gap or lateral_gap > max_lateral_gap:
            if (opening_aligned
                    and self._observed_blue_live_progress_match_is_reliable(
                        door, candidate,
                        target_wall, candidate_wall,
                        progress_gap, lateral_gap,
                        max_progress_gap, max_lateral_gap)):
                return True
            n.get_logger().info(
                f'Fresh blue evidence rejected for {door.door_id}: live '
                f'candidate {candidate.door_id} is not the same mapped station '
                f'(progress_gap={progress_gap:.2f}/{max_progress_gap:.2f}, '
                f'lateral_gap={lateral_gap:.2f}/{max_lateral_gap:.2f}).',
                throttle_duration_sec=3.0)
            return False

        if pre_exit_station_match:
            return True
        return self._any_points_close(target_points, candidate_points, max_dist)

    def _observed_blue_live_progress_match_is_reliable(
            self,
            door,
            candidate,
            target_wall,
            candidate_wall,
            progress_gap: float,
            lateral_gap: float,
            max_progress_gap: float,
            max_lateral_gap: float) -> bool:
        n = self.node
        target_xy = n._door_identity_xy(door)
        if n._blue_xy_opened_station_relation(target_xy) is not None:
            return False
        candidate_xy = n._door_identity_xy(candidate)
        if (
                n._blue_xy_opened_station_relation(candidate_xy) is not None
                and n._blue_xy_opened_station_relation(target_xy) is None):
            return False
        aligned_max_progress_gap = max(
            0.45,
            min(max_progress_gap + 0.35, 1.15))
        opening_aligned = n._blue_target_open_pose_aligned_for_safe_memory(door)
        if progress_gap > aligned_max_progress_gap:
            return False
        max_projection_lateral_gap = max(
            max_lateral_gap + 0.70,
            max_lateral_gap * 1.35,
            getattr(n, '_axis_door_side_standoff_m', 0.0) + 1.15,
            getattr(n, '_observed_blue_fresh_evidence_max_lateral_gap_m', 0.0) + 1.15)
        if opening_aligned and getattr(n, '_opened_blue_door_count')() > 0:
            max_projection_lateral_gap = min(
                max_projection_lateral_gap,
                max_lateral_gap + 0.35,
                max(
                    0.55,
                    getattr(n, '_axis_door_side_standoff_m', 0.0) * 0.65))
        if lateral_gap > max_projection_lateral_gap:
            return False
        if not opening_aligned:
            return False

        if not n._is_xy_at_configured_wall_lateral(target_xy):
            return False

        strong_blue, blue_confidence, blue_count = (
            n._blue_target_observation_strength(door))
        cluster = n._find_observed_blue_cluster(
            target_xy,
            merge_dist=n._observed_candidate_merge_dist())
        cluster_count = 0
        cluster_confidence = 0.0
        if cluster is not None:
            if cluster.get('abandoned', 0.0) >= 1.0:
                return False
            cluster_count = int(cluster.get('count', 0.0))
            cluster_confidence = float(cluster.get('confidence', 0.0))

        evidence_count = max(blue_count, cluster_count)
        evidence_confidence = max(
            blue_confidence,
            cluster_confidence,
            abs(float(door.confidence)),
            float(candidate.confidence))
        min_count = max(3, n._observed_blue_min_observations + 2)
        min_confidence = max(
            0.40,
            max(0.0, n._door_open_fresh_blue_min_confidence),
            max(0.0, n._pre_exit_blue_min_confidence))
        if (
                not strong_blue
                and (
                    evidence_count < min_count
                    or evidence_confidence < min_confidence)):
            return False

        target_lateral = n._axis_lateral_xy(target_xy[0], target_xy[1])
        candidate_lateral = n._axis_lateral_xy(candidate_xy[0], candidate_xy[1])
        target_side = target_lateral - n._explore_center_y
        candidate_side = candidate_lateral - n._explore_center_y
        wall_side_threshold = max(
            0.15,
            getattr(n, '_observed_blue_min_abs_wall_y_m', 0.0) * 0.5)
        if abs(target_side) < wall_side_threshold:
            return False
        if (
                abs(candidate_side) >= wall_side_threshold
                and target_side * candidate_side < 0.0):
            return False
        target_wall_side = (
            n._axis_lateral_xy(target_wall[0], target_wall[1])
            - n._explore_center_y)
        candidate_wall_side = (
            n._axis_lateral_xy(candidate_wall[0], candidate_wall[1])
            - n._explore_center_y)
        if (
                abs(target_wall_side) >= wall_side_threshold
                and abs(candidate_wall_side) >= wall_side_threshold
                and target_wall_side * candidate_wall_side < 0.0):
            return False

        n.get_logger().info(
            f'Fresh blue evidence accepted for {door.door_id}: live progress '
            f'match tolerates projection lateral drift '
            f'(progress_gap={progress_gap:.2f}/{aligned_max_progress_gap:.2f}, '
            f'lateral_gap={lateral_gap:.2f}/{max_projection_lateral_gap:.2f}, '
            f'count={evidence_count}, conf={evidence_confidence:.2f}).',
            throttle_duration_sec=3.0)
        return True

    def _direct_blue_evidence_matches_target(
            self, door, candidate, max_dist: float) -> bool:
        """Confirm direct targets by comparing the same physical station.

        The target and the live candidate each expose several points: a navigation
        pose, a wall/handle point, and sometimes a raw wall evidence point. A false
        side-camera projection can make one target point close to a different
        candidate point even when the physical wall stations are several meters
        apart. For final opening, compare identity-to-identity and
        handle-to-handle instead of accepting any cross-pair distance.
        """
        n = self.node
        target_xy = n._door_identity_xy(door)
        candidate_xy = n._door_identity_xy(candidate)
        target_wall = n._door_handle_xy(door)
        candidate_wall = n._door_handle_xy(candidate)
        candidate_raw_wall = n._raw_blue_handle_xy_for_evidence(candidate)
        if (
                candidate_raw_wall is not None
                and math.hypot(
                    target_wall[0] - candidate_raw_wall[0],
                    target_wall[1] - candidate_raw_wall[1])
                < math.hypot(
                    target_wall[0] - candidate_wall[0],
                    target_wall[1] - candidate_wall[1])):
            candidate_wall = candidate_raw_wall

        target_lateral = n._axis_lateral_xy(target_xy[0], target_xy[1])
        candidate_lateral = n._axis_lateral_xy(candidate_xy[0], candidate_xy[1])
        target_side = target_lateral - n._explore_center_y
        candidate_side = candidate_lateral - n._explore_center_y
        wall_side_threshold = max(
            0.15,
            getattr(n, '_observed_blue_min_abs_wall_y_m', 0.0) * 0.5)
        if (
                abs(target_side) >= wall_side_threshold
                and abs(candidate_side) >= wall_side_threshold
                and target_side * candidate_side < 0.0):
            n.get_logger().info(
                f'Fresh blue evidence rejected for {door.door_id}: live '
                f'candidate {candidate.door_id} is on the opposite wall.',
                throttle_duration_sec=3.0)
            return False

        identity_dist = math.hypot(
            target_xy[0] - candidate_xy[0],
            target_xy[1] - candidate_xy[1])
        wall_dist = math.hypot(
            target_wall[0] - candidate_wall[0],
            target_wall[1] - candidate_wall[1])
        same_detector_family = (
            candidate.door_id in n._door_id_keys(door.door_id)
            or door.door_id in n._door_id_keys(candidate.door_id)
            or n._canonical_door_id(candidate.door_id)
            == n._canonical_door_id(door.door_id))
        if same_detector_family and min(identity_dist, wall_dist) <= max_dist:
            return True

        target_progress = n._axis_progress_xy(target_xy[0], target_xy[1])
        candidate_progress = n._axis_progress_xy(candidate_xy[0], candidate_xy[1])
        target_wall_progress = n._axis_progress_xy(target_wall[0], target_wall[1])
        candidate_wall_progress = n._axis_progress_xy(candidate_wall[0], candidate_wall[1])
        target_wall_lateral = n._axis_lateral_xy(target_wall[0], target_wall[1])
        candidate_wall_lateral = n._axis_lateral_xy(candidate_wall[0], candidate_wall[1])

        max_progress_gap = max(
            0.20,
            getattr(n, '_observed_blue_fresh_evidence_max_progress_gap_m', 0.95))
        max_lateral_gap = max(
            0.20,
            getattr(n, '_observed_blue_fresh_evidence_max_lateral_gap_m', 0.85))
        max_station_dist = max(
            max_dist,
            getattr(n, '_observed_candidate_merge_dist')())

        progress_gap = abs(candidate_progress - target_progress)
        lateral_gap = abs(candidate_lateral - target_lateral)
        wall_progress_gap = abs(candidate_wall_progress - target_wall_progress)
        wall_lateral_gap = abs(candidate_wall_lateral - target_wall_lateral)
        axis_station_dist = max(
            max_station_dist,
            math.hypot(max_progress_gap, max_lateral_gap))
        identity_station_match = (
            identity_dist <= axis_station_dist
            and progress_gap <= max_progress_gap
            and lateral_gap <= max_lateral_gap)
        wall_station_match = (
            wall_dist <= axis_station_dist
            and wall_progress_gap <= max_progress_gap
            and wall_lateral_gap <= max_lateral_gap)
        if identity_station_match or wall_station_match:
            return True

        if n._direct_blue_candidate_matches_observed_wall_memory(
                candidate, target_xy):
            return True

        n.get_logger().info(
            f'Fresh blue evidence rejected for {door.door_id}: live '
            f'candidate {candidate.door_id} is not the same mapped door station '
            f'(identity_dist={identity_dist:.2f}/{axis_station_dist:.2f}, '
            f'wall_dist={wall_dist:.2f}/{axis_station_dist:.2f}, '
            f'progress_gap={progress_gap:.2f}/{max_progress_gap:.2f}, '
            f'lateral_gap={lateral_gap:.2f}/{max_lateral_gap:.2f}, '
            f'wall_progress_gap={wall_progress_gap:.2f}/{max_progress_gap:.2f}, '
            f'wall_lateral_gap={wall_lateral_gap:.2f}/{max_lateral_gap:.2f}).',
            throttle_duration_sec=3.0)
        return False

    @staticmethod
    def _any_points_close(points_a, points_b, max_dist: float) -> bool:
        return any(
            math.hypot(a[0] - b[0], a[1] - b[1]) <= max_dist
            for a in points_a
            for b in points_b)

    def _points_near_opened_station(self, points) -> bool:
        n = self.node
        if not n._opened_blue_door_positions:
            return False
        merge_dist = max(
            0.20,
            getattr(n, '_opened_physical_door_merge_dist_m', 0.0),
            getattr(n, '_observed_physical_door_merge_dist_m', 0.0))
        for xy in points:
            for opened_xy in n._opened_blue_door_positions:
                if math.hypot(
                        xy[0] - opened_xy[0],
                        xy[1] - opened_xy[1]) <= merge_dist:
                    return True

        window = max(
            0.0,
            n._opened_station_blue_suppression_progress_m,
            n._opened_station_same_side_blue_suppression_progress_m)
        if hasattr(n, '_opened_same_wall_remnant_progress_window'):
            window = n._opened_same_wall_remnant_progress_window(window)
        else:
            window = max(0.0, min(window, merge_dist))
        if window <= 0.0:
            return False
        wall_side_threshold = max(
            0.15,
            n._observed_blue_min_abs_wall_y_m * 0.5)
        for xy in points:
            point_progress = n._axis_progress_xy(xy[0], xy[1])
            point_lateral = n._axis_lateral_xy(xy[0], xy[1]) - n._explore_center_y
            point_on_wall = abs(point_lateral) >= wall_side_threshold
            for opened_xy in n._opened_blue_door_positions:
                opened_progress = n._axis_progress_xy(opened_xy[0], opened_xy[1])
                opened_lateral = (
                    n._axis_lateral_xy(opened_xy[0], opened_xy[1])
                    - n._explore_center_y)
                opened_on_wall = abs(opened_lateral) >= wall_side_threshold
                same_wall = (
                    point_on_wall
                    and opened_on_wall
                    and point_lateral * opened_lateral > 0.0)
                if same_wall and abs(point_progress - opened_progress) <= window:
                    return True
        return False


class DoorApproachSubFsm:
    """Door approach result policy.

    Nav2 may report success before the robot finishes rotating toward a side
    door, and it may also report failure when the robot is already close enough
    for low-speed door alignment. This helper keeps that decision separate
    from the mission-level FSM.
    """

    def __init__(self, node: Any):
        self.node = node

    def can_continue_with_fine_alignment(self, reason: str) -> bool:
        n = self.node
        door = n.target_door
        if door is None or door.door_color != 'blue':
            return False

        error = n._door_open_pose_error(door)
        if error is None:
            return False

        dist, yaw_error, forward_error, lateral_error = error
        lateral_abs = abs(lateral_error)
        fine_dist = max(
            n._door_open_fine_control_max_dist_m,
            n._door_open_ready_max_dist_m + 0.28,
            0.52)
        fine_lateral = max(
            n._door_open_fine_lateral_tolerance_m,
            n._door_open_ready_lateral_tolerance_m + 0.16,
            0.36)

        if dist <= fine_dist and lateral_abs <= fine_lateral:
            n.get_logger().warn(
                f'Nav result [{reason}] is close enough for door fine alignment: '
                f'{door.door_id}, dist={dist:.2f}/{fine_dist:.2f}m, '
                f'lateral={lateral_abs:.2f}/{fine_lateral:.2f}m, '
                f'forward={forward_error:.2f}m, '
                f'yaw_error={math.degrees(abs(yaw_error)):.0f}deg.')
            return True

        return False

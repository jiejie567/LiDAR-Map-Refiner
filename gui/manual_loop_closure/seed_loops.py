"""Descriptor retrieval for the Initial Loop Search phase.

This module proposes revisit candidates from Scan Context appearance.  The
caller then performs geometric registration and safety audit before an initial
loop can enter the working graph.  It is not a numbered optimizer stage and it
does not depend on the duplicated-surface audit.  Pure numpy; geometric
verification (GICP) is the caller's responsibility.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Tuple

import numpy as np

try:  # package import (GUI); falls back to leaf import for open3d-free use
    from .scan_context_io import (
        ScanContextConfig,
        _gravity_canonical_rotation,
        make_descriptor_with_mask,
    )
except ImportError:  # pragma: no cover - test/CLI leaf-module path
    from scan_context_io import (  # type: ignore
        ScanContextConfig,
        _gravity_canonical_rotation,
        make_descriptor_with_mask,
    )


@dataclass(frozen=True)
class SeedCandidate:
    source_id: int
    target_id: int
    distance: float
    yaw_deg: float  # rotation of source scan that best aligns it to target


def build_descriptor_stack(
    load_points: Callable[[int], np.ndarray],
    count: int,
    config: ScanContextConfig,
    gravity_up_body: Optional[np.ndarray] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    progress_fn: Optional[Callable[[int, int], None]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (descriptor, validity) stacks for all keyframes.

    ``load_points(i)`` returns the body-frame (N,3) cloud of keyframe ``i``.
    When gravity canonicalization is enabled and per-frame up-vectors are
    given, clouds are rotated exactly as at mapping time.
    """
    descriptors = None
    masks = None
    for index in range(count):
        points = np.asarray(load_points(index), dtype=np.float64)[:, :3]
        if (
            config.gravity_canonicalization_enable
            and gravity_up_body is not None
        ):
            rotation = _gravity_canonical_rotation(gravity_up_body[index])
            points = points @ rotation.T
        descriptor, mask = make_descriptor_with_mask(points, config)
        if descriptors is None:
            descriptors = np.zeros((count,) + descriptor.shape)
            masks = np.zeros((count,) + mask.shape, dtype=bool)
        descriptors[index] = descriptor
        masks[index] = mask
        if progress_fn is not None:
            progress_fn(index + 1, count)
        if log_fn is not None and (index + 1) % 500 == 0:
            log_fn(f"[InitialLoops] descriptors {index + 1}/{count}")
    return descriptors, masks


def build_descriptor_and_prepared_stack(
    load_points: Callable[[int], np.ndarray],
    prepare_points: Callable[[np.ndarray], np.ndarray],
    count: int,
    config: ScanContextConfig,
    gravity_up_body: Optional[np.ndarray] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    progress_fn: Optional[Callable[[int, int], None]] = None,
) -> tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Build descriptors and diagnosis clouds with one source read per frame.

    Descriptor construction and ghost diagnosis both start from the same raw
    local-frame cloud.  Combining them here avoids reading every PCD twice,
    which is particularly costly when a session lives on an external disk.
    Only the compact prepared cloud is retained; raw clouds are released after
    each descriptor has been built.
    """
    prepared: List[Optional[np.ndarray]] = [None] * count

    def load_and_prepare(index: int) -> np.ndarray:
        points = np.asarray(load_points(index))
        prepared[index] = prepare_points(points)
        return points

    descriptors, masks = build_descriptor_stack(
        load_and_prepare,
        count,
        config,
        gravity_up_body,
        log_fn,
        progress_fn,
    )
    if any(cloud is None for cloud in prepared):
        raise RuntimeError("not all keyframe clouds were prepared")
    return descriptors, masks, [cloud for cloud in prepared if cloud is not None]


def descriptor_env_setup(
    base_config: ScanContextConfig, environment: str
) -> tuple[ScanContextConfig, Optional[Tuple[float, float]]]:
    """Seed-retrieval descriptor preset per environment.

    'indoor'  -> UpDown-style only when the recorded session contains valid
                 dual-z geometry; otherwise protected single-layer retrieval.
    'outdoor' -> native Scan Context: single max-height channel, 80 m radius.
    Returns (config, channel_weights or None).
    """
    if environment == "outdoor":
        return (
            replace(
                base_config,
                dual_z_layer_enable=False,
                max_radius=max(base_config.max_radius, 80.0),
            ),
            None,
        )
    if not base_config.dual_z_layer_enable:
        return base_config, None
    return base_config, (0.3, 0.7)


def _unit_columns(
    descriptors: np.ndarray,
    masks: np.ndarray,
    channel_weights: Optional[Tuple[float, float]] = None,
    num_rings: Optional[int] = None,
) -> np.ndarray:
    """Zero invalid cells and normalize each sector column to unit norm.

    With channel_weights, the two stacked dual-z channel blocks are normalized
    per channel and scaled by sqrt(w), so the stacked column dot product equals
    the weighted sum of per-channel cosine similarities.
    """
    clean = np.where(masks, descriptors, 0.0)
    if channel_weights is not None and num_rings is not None:
        blocks = []
        for weight, sl in zip(
            channel_weights,
            (slice(0, num_rings), slice(num_rings, None)),
        ):
            block = clean[:, sl, :]
            norms = np.linalg.norm(block, axis=1, keepdims=True)
            blocks.append(math.sqrt(weight) * block / np.maximum(norms, 1e-12))
        return np.concatenate(blocks, axis=1)
    norms = np.linalg.norm(clean, axis=1, keepdims=True)
    return clean / np.maximum(norms, 1e-12)


def pairwise_alignment(
    unit_a: np.ndarray,
    valid_a: np.ndarray,
    fft_b: np.ndarray,
    fft_valid_b: np.ndarray,
) -> tuple[float, int]:
    """Best mean column-cosine similarity of one frame against one cached frame.

    Uses the correlation theorem: the sum over sectors of column dot products,
    for every circular shift at once, is an inverse FFT of the spectra product.
    Returns (best mean cosine, best shift in sectors).
    """
    fft_a = np.fft.rfft(unit_a, axis=1)
    corr = np.fft.irfft(np.sum(fft_a * np.conj(fft_b), axis=0), n=unit_a.shape[1])
    fft_va = np.fft.rfft(valid_a.astype(np.float64))
    overlap = np.fft.irfft(fft_va * np.conj(fft_valid_b), n=unit_a.shape[1])
    overlap = np.maximum(overlap, 1.0)
    scores = corr / overlap
    best = int(np.argmax(scores))
    return float(scores[best]), best


def _layer_distance(
    query: np.ndarray,
    query_mask: np.ndarray,
    candidate: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    min_joint_rings: int,
    retrieval_height_offset: float,
    sector_support_exponent: float,
) -> Optional[float]:
    """Return the C++ V7 per-layer distance for one fixed yaw.

    Joint-ring support rejects accidental one-cell matches.  The supported
    sector cosine then keeps non-overlapping sectors in the score instead of
    letting them disappear from the denominator.
    """
    support = max(1, int(min_joint_rings))
    query_counts = query_mask.sum(axis=0)
    candidate_counts = candidate_mask.sum(axis=0)
    joint = query_mask & candidate_mask
    joint_counts = joint.sum(axis=0)
    if not np.any(query_mask) or not np.any(candidate_mask):
        return None

    query_values = np.where(
        joint, query + float(retrieval_height_offset), 0.0)
    candidate_values = np.where(
        joint, candidate + float(retrieval_height_offset), 0.0)
    dot = np.sum(query_values * candidate_values, axis=0)
    query_norm = np.sum(query_values * query_values, axis=0)
    candidate_norm = np.sum(candidate_values * candidate_values, axis=0)
    comparable = (
        (joint_counts >= support)
        & (query_norm > 1e-24)
        & (candidate_norm > 1e-24)
    )
    effective_columns = int(np.count_nonzero(comparable))
    if effective_columns == 0:
        return None

    value_similarity = dot[comparable] / np.sqrt(
        query_norm[comparable] * candidate_norm[comparable])
    mask_similarity = joint_counts[comparable] / np.sqrt(
        query_counts[comparable] * candidate_counts[comparable])
    similarity = float(np.mean(value_similarity * mask_similarity))
    query_supported = int(np.count_nonzero(query_counts >= support))
    candidate_supported = int(np.count_nonzero(candidate_counts >= support))
    if query_supported == 0 or candidate_supported == 0:
        return None
    sector_support = effective_columns / math.sqrt(
        float(query_supported * candidate_supported))
    sector_support = max(0.0, min(1.0, sector_support))
    return 1.0 - similarity * math.pow(
        sector_support, max(0.0, float(sector_support_exponent)))


def _descriptor_distance(
    query: np.ndarray,
    query_mask: np.ndarray,
    candidate: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    channel_weights: Optional[Tuple[float, float]],
    num_rings: int,
    min_joint_rings: int,
    retrieval_height_offset: float,
    sector_support_exponent: float,
) -> float:
    if channel_weights is None or query.shape[0] == num_rings:
        distance = _layer_distance(
            query,
            query_mask,
            candidate,
            candidate_mask,
            min_joint_rings=min_joint_rings,
            retrieval_height_offset=retrieval_height_offset,
            sector_support_exponent=sector_support_exponent,
        )
        return float("inf") if distance is None else float(distance)

    weighted_distance = 0.0
    weight_sum = 0.0
    hard_mismatch = False
    for weight, rows in zip(
        channel_weights,
        (slice(0, num_rings), slice(num_rings, 2 * num_rings)),
    ):
        weight = max(0.0, float(weight))
        if weight <= 0.0:
            continue
        q_mask = query_mask[rows]
        c_mask = candidate_mask[rows]
        distance = _layer_distance(
            query[rows],
            q_mask,
            candidate[rows],
            c_mask,
            min_joint_rings=min_joint_rings,
            retrieval_height_offset=retrieval_height_offset,
            sector_support_exponent=sector_support_exponent,
        )
        # As in C++ V7, a layer absent on both sides carries no evidence.  A
        # one-sided absence or insufficient joint support is a hard mismatch.
        if not np.any(q_mask) and not np.any(c_mask):
            continue
        if distance is None:
            hard_mismatch = True
            continue
        weighted_distance += weight * distance
        weight_sum += weight
    if hard_mismatch:
        return 1.0
    if weight_sum <= 1e-12:
        return float("inf")
    return weighted_distance / weight_sum


def _sector_key(
    descriptor: np.ndarray,
    mask: np.ndarray,
    retrieval_height_offset: float,
) -> np.ndarray:
    values = np.where(
        mask, descriptor + float(retrieval_height_offset), 0.0)
    return values.sum(axis=0) / max(1, descriptor.shape[0])


def _yaw_distances(
    query: np.ndarray,
    query_mask: np.ndarray,
    candidate: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    channel_weights: Optional[Tuple[float, float]],
    num_rings: int,
    min_joint_rings: int,
    retrieval_height_offset: float,
    sector_support_exponent: float,
) -> List[tuple[float, int]]:
    """Released Scan Context coarse yaw followed by its local exact window."""
    sectors = query.shape[1]
    query_key = _sector_key(query, query_mask, retrieval_height_offset)
    candidate_key = _sector_key(candidate, candidate_mask, retrieval_height_offset)
    coarse_shift = min(
        range(sectors),
        key=lambda shift: float(np.sum(
            (query_key - np.roll(candidate_key, shift)) ** 2)),
    )
    search_radius = int(round(0.5 * 0.1 * sectors))
    scored: List[tuple[float, int]] = []
    for offset in range(-search_radius, search_radius + 1):
        shift = (coarse_shift + offset) % sectors
        distance = _descriptor_distance(
            query,
            query_mask,
            np.roll(candidate, shift, axis=1),
            np.roll(candidate_mask, shift, axis=1),
            channel_weights=channel_weights,
            num_rings=num_rings,
            min_joint_rings=min_joint_rings,
            retrieval_height_offset=retrieval_height_offset,
            sector_support_exponent=sector_support_exponent,
        )
        if np.isfinite(distance):
            scored.append((float(distance), int(shift)))
    return sorted(scored)


def find_seed_candidates(
    descriptors: np.ndarray,
    masks: np.ndarray,
    *,
    min_index_gap: int = 150,
    ring_key_top_k: int = 10,
    max_distance: float = 0.40,
    max_seeds: int = 5,
    segment_radius: int = 100,
    channel_weights: Optional[Tuple[float, float]] = None,
    num_rings: Optional[int] = None,
    min_joint_rings: int = 2,
    retrieval_height_offset: float = 0.1,
    sector_support_exponent: float = 0.5,
    log_fn: Optional[Callable[[str], None]] = None,
) -> List[SeedCandidate]:
    """Retrieve up to ``max_seeds`` well-separated revisit candidates.

    Ring keys (rotation-invariant per-ring means) prefilter candidates; the
    survivors are scored with the yaw-aligned column-cosine distance. Greedy
    selection enforces one seed per pair of trajectory segments.
    """
    count = descriptors.shape[0]
    sectors = descriptors.shape[2]
    if count == 0:
        return []
    if num_rings is None:
        num_rings = descriptors.shape[1]
    clean = np.where(
        masks, descriptors + float(retrieval_height_offset), 0.0)
    # Match C++ makeRingKey: missing sectors remain zero evidence rather than
    # disappearing from the denominator.
    ring_keys = clean.sum(axis=2) / max(1, sectors)

    key_norm_sq = np.einsum("ij,ij->i", ring_keys, ring_keys)
    scored: List[SeedCandidate] = []
    for query in range(count):
        lo, hi = query - min_index_gap, query + min_index_gap
        allowed = np.ones(count, dtype=bool)
        allowed[max(lo, 0) : min(hi + 1, count)] = False
        allowed[query:] = False  # each unordered pair once (candidate < query)
        if not allowed.any():
            continue
        key_dist = (
            key_norm_sq
            - 2.0 * ring_keys @ ring_keys[query]
            + key_norm_sq[query]
        )
        key_dist[~allowed] = np.inf
        top = np.argpartition(key_dist, min(ring_key_top_k, count - 1))[:ring_key_top_k]
        top = top[np.isfinite(key_dist[top])]
        best_candidate = None
        for candidate in top:
            yaw_scores = _yaw_distances(
                descriptors[query],
                masks[query],
                descriptors[candidate],
                masks[candidate],
                channel_weights=channel_weights,
                num_rings=num_rings,
                min_joint_rings=min_joint_rings,
                retrieval_height_offset=retrieval_height_offset,
                sector_support_exponent=sector_support_exponent,
            )
            if not yaw_scores:
                continue
            distance, shift = yaw_scores[0]
            if distance > max_distance:
                continue
            if best_candidate is None or distance < best_candidate.distance:
                yaw_deg = -math.degrees(2.0 * math.pi * shift / sectors)
                if yaw_deg <= -180.0:
                    yaw_deg += 360.0
                best_candidate = SeedCandidate(
                    source_id=int(max(query, candidate)),
                    target_id=int(min(query, candidate)),
                    distance=distance,
                    yaw_deg=yaw_deg,
                )
        if best_candidate is not None:
            scored.append(best_candidate)
        if log_fn is not None and (query + 1) % 500 == 0:
            log_fn(f"[InitialLoops] matched {query + 1}/{count}")

    scored.sort(key=lambda item: item.distance)
    selected: List[SeedCandidate] = []
    for candidate in scored:
        redundant = any(
            abs(candidate.source_id - chosen.source_id) <= segment_radius
            and abs(candidate.target_id - chosen.target_id) <= segment_radius
            for chosen in selected
        )
        if redundant:
            continue
        selected.append(candidate)
        if len(selected) >= max_seeds:
            break
    return selected


def pair_yaw_peaks(
    descriptors: np.ndarray,
    masks: np.ndarray,
    source_id: int,
    target_id: int,
    channel_weights: Optional[Tuple[float, float]] = None,
    num_rings: Optional[int] = None,
    min_joint_rings: int = 2,
    retrieval_height_offset: float = 0.1,
    sector_support_exponent: float = 0.5,
    top_k: int = 3,
    min_separation_sectors: int = 3,
) -> List[tuple[float, float]]:
    """Top-k yaw hypotheses for one KNOWN pair (relocalization-style).

    Instead of the single argmax of the circular correlation, return up to
    ``top_k`` local peaks separated by at least ``min_separation_sectors``.
    Symmetric scenes (corridors, avenues) put the true yaw in a secondary
    peak often enough that registration should try each hypothesis. Returns
    [(distance, yaw_deg), ...] best-first, same convention as
    ``pair_yaw_alignment``.
    """
    if num_rings is None:
        num_rings = descriptors.shape[1]
    sectors = descriptors.shape[2]
    scored = _yaw_distances(
        descriptors[source_id], masks[source_id],
        descriptors[target_id], masks[target_id],
        channel_weights=channel_weights,
        num_rings=num_rings,
        min_joint_rings=min_joint_rings,
        retrieval_height_offset=retrieval_height_offset,
        sector_support_exponent=sector_support_exponent,
    )
    picks: List[int] = []
    distances = {shift: distance for distance, shift in scored}
    for _, s in scored:
        if len(picks) >= top_k:
            break
        if any(min(abs(s - p), sectors - abs(s - p)) < min_separation_sectors
               for p in picks):
            continue
        picks.append(int(s))
    out = []
    for s in picks:
        yaw_deg = -math.degrees(2.0 * math.pi * s / sectors)
        if yaw_deg <= -180.0:
            yaw_deg += 360.0
        out.append((float(distances[s]), yaw_deg))
    return out


def pair_yaw_alignment(
    descriptors: np.ndarray,
    masks: np.ndarray,
    source_id: int,
    target_id: int,
    channel_weights: Optional[Tuple[float, float]] = None,
    num_rings: Optional[int] = None,
    min_joint_rings: int = 2,
    retrieval_height_offset: float = 0.1,
    sector_support_exponent: float = 0.5,
) -> tuple[float, float]:
    """Descriptor distance and aligning yaw for one KNOWN pair.

    Even when retrieval-level distance is too weak to propose the pair, the
    yaw alignment is often still usable as a registration prior. Returns
    (distance, yaw_deg) with the same convention as SeedCandidate.yaw_deg:
    rotating the source scan by yaw_deg about z aligns it with the target.
    """
    if num_rings is None:
        num_rings = descriptors.shape[1]
    scored = _yaw_distances(
        descriptors[source_id], masks[source_id],
        descriptors[target_id], masks[target_id],
        channel_weights=channel_weights,
        num_rings=num_rings,
        min_joint_rings=min_joint_rings,
        retrieval_height_offset=retrieval_height_offset,
        sector_support_exponent=sector_support_exponent,
    )
    if not scored:
        return float("inf"), 0.0
    distance, shift = scored[0]
    sectors = descriptors.shape[2]
    yaw_deg = -math.degrees(2.0 * math.pi * shift / sectors)
    if yaw_deg <= -180.0:
        yaw_deg += 360.0
    return float(distance), yaw_deg


def ghost_badness(regions: List[dict]) -> float:
    """Scalar map-inconsistency score used for seed probation comparisons.

    Tolerates a missing point count rather than raising: callers pass regions
    from two sources -- the detector, and the BALM report reloaded from disk --
    and a scoring helper should not be the thing that takes the GUI down when
    the second one is thinner than the first.
    """
    return float(
        sum(region.get("point_count", 0)
            * region.get("separation_typical_m", region.get("separation_m", 0.0))
            for region in regions)
    )

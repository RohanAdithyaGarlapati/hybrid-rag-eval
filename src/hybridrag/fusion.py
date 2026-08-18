"""Rank list fusion methods.

* ``reciprocal_rank_fusion`` combines ranked id lists using weighted 1 / (k + rank).
* ``normalized_score_fusion`` min-max normalizes each score list per query, then
  takes a weighted sum.
"""

from __future__ import annotations

from typing import Mapping, Sequence


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    *,
    weights: Sequence[float] | None = None,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists into one ``(id, score)`` list sorted best first.

    Each list contributes ``weight / (k + rank)`` where ``rank`` is 1 based. Items
    absent from a list simply contribute nothing from that list.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights length must match ranked_lists length")
    if k <= 0:
        raise ValueError("k must be positive")

    scores: dict[str, float] = {}
    for weight, ranked in zip(weights, ranked_lists):
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + weight * (1.0 / (k + rank))
    fused = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return fused


def normalized_score_fusion(
    score_maps: Sequence[Mapping[str, float]],
    *,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Min-max normalize each score map, then weighted sum.

    Within one map, scores are rescaled to [0, 1] using that map's own min and max
    (a degenerate map where all scores are equal maps to 1.0). Ids missing from a
    map contribute 0 from it.
    """
    if weights is None:
        weights = [1.0] * len(score_maps)
    if len(weights) != len(score_maps):
        raise ValueError("weights length must match score_maps length")

    normalized: list[dict[str, float]] = []
    for smap in score_maps:
        if not smap:
            normalized.append({})
            continue
        vals = list(smap.values())
        lo, hi = min(vals), max(vals)
        span = hi - lo
        if span == 0.0:
            normalized.append({key: 1.0 for key in smap})
        else:
            normalized.append({key: (val - lo) / span for key, val in smap.items()})

    combined: dict[str, float] = {}
    for weight, nmap in zip(weights, normalized):
        for key, val in nmap.items():
            combined[key] = combined.get(key, 0.0) + weight * val
    fused = sorted(combined.items(), key=lambda kv: (-kv[1], kv[0]))
    return fused

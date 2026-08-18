"""Retrieval metrics computed from a ranked list of doc ids and a gold set.

All functions take ``retrieved`` (an ordered list of doc ids, best first) and
``gold`` (a set or list of relevant doc ids). Ranks are 1 based.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def _gold_set(gold: Iterable[str]) -> set[str]:
    return set(gold)


def recall_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    g = _gold_set(gold)
    if not g:
        return 0.0
    topk = list(retrieved[:k])
    hits = sum(1 for d in g if d in topk)
    return hits / len(g)


def hit_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    g = _gold_set(gold)
    if not g:
        return 0.0
    return 1.0 if any(d in g for d in retrieved[:k]) else 0.0


def precision_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    if k <= 0:
        return 0.0
    g = _gold_set(gold)
    topk = list(retrieved[:k])
    hits = sum(1 for d in topk if d in g)
    return hits / k


def reciprocal_rank(retrieved: Sequence[str], gold: Iterable[str]) -> float:
    g = _gold_set(gold)
    for rank, d in enumerate(retrieved, start=1):
        if d in g:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    g = _gold_set(gold)
    dcg = 0.0
    for rank, d in enumerate(retrieved[:k], start=1):
        rel = 1.0 if d in g else 0.0
        if rel:
            dcg += rel / math.log2(rank + 1)
    return dcg


def ndcg_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    g = _gold_set(gold)
    if not g:
        return 0.0
    dcg = dcg_at_k(retrieved, gold, k)
    ideal_hits = min(len(g), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate_query(
    retrieved: Sequence[str],
    gold: Iterable[str],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """All per query metrics keyed like ``recall@5`` and ``ndcg@10``, plus ``mrr``."""
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(retrieved, gold, k)
        out[f"hit@{k}"] = hit_at_k(retrieved, gold, k)
        out[f"precision@{k}"] = precision_at_k(retrieved, gold, k)
        out[f"ndcg@{k}"] = ndcg_at_k(retrieved, gold, k)
    out["mrr"] = reciprocal_rank(retrieved, gold)
    return out


def aggregate(per_query: Sequence[dict[str, float]]) -> dict[str, float]:
    """Mean each metric across queries. Empty input yields an empty dict."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    n = len(per_query)
    return {key: sum(q[key] for q in per_query) / n for key in keys}

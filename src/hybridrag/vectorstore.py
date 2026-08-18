"""A tiny in memory vector store with cosine search and metadata filtering.

Vectors are L2 normalized on insert so cosine similarity is a plain dot product.
Top k selection uses ``numpy.argpartition`` for an O(n) partial sort before a small
exact sort of the k survivors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return mat / norms


@dataclass
class SearchHit:
    id: str
    score: float
    metadata: dict


class VectorStore:
    def __init__(self, dimension: int) -> None:
        self.dimension = int(dimension)
        self._ids: list[str] = []
        self._meta: list[dict] = []
        self._matrix: np.ndarray = np.zeros((0, self.dimension), dtype=np.float64)

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, ids: list[str], vectors: np.ndarray, metadata: list[dict] | None = None) -> None:
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(f"expected vectors of shape (n, {self.dimension})")
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids and vectors length mismatch")
        metadata = metadata or [{} for _ in ids]
        if len(metadata) != len(ids):
            raise ValueError("metadata and ids length mismatch")
        normed = _l2_normalize(vectors)
        self._ids.extend(ids)
        self._meta.extend(metadata)
        self._matrix = np.vstack([self._matrix, normed]) if len(self._matrix) else normed

    def _rank(self, query: np.ndarray, k: int, mask: np.ndarray | None = None) -> list[SearchHit]:
        if len(self._ids) == 0:
            return []
        q = np.asarray(query, dtype=np.float64).reshape(-1)
        norm = np.linalg.norm(q)
        if norm != 0.0:
            q = q / norm
        sims = self._matrix @ q  # cosine, since both sides are normalized

        idx_pool = np.arange(len(self._ids)) if mask is None else np.nonzero(mask)[0]
        if idx_pool.size == 0:
            return []
        pool_sims = sims[idx_pool]
        k_eff = min(k, idx_pool.size)
        # argpartition puts the k largest (by negated sim) in the first k slots.
        part = np.argpartition(-pool_sims, k_eff - 1)[:k_eff]
        order = part[np.argsort(-pool_sims[part])]
        chosen = idx_pool[order]
        return [
            SearchHit(id=self._ids[i], score=float(sims[i]), metadata=self._meta[i])
            for i in chosen
        ]

    def search(self, query: np.ndarray, k: int = 10) -> list[SearchHit]:
        return self._rank(query, k, mask=None)

    def filtered_search(
        self, query: np.ndarray, k: int, predicate: Callable[[dict], bool]
    ) -> list[SearchHit]:
        mask = np.array([predicate(m) for m in self._meta], dtype=bool)
        return self._rank(query, k, mask=mask)

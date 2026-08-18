"""Embedders behind a small protocol.

* ``SentenceTransformerEmbedder`` wraps ``all-MiniLM-L6-v2`` with a deferred import
  so the heavy dependency is only loaded when actually used. It reports
  ``is_semantic = True``.
* ``HashingEmbedder`` is a dependency free fallback: it hashes character 3 to 5
  grams with blake2b into a signed, log damped, L2 normalized vector. It is fully
  deterministic and reports ``is_semantic = False``.
* ``build_embedder("auto")`` prefers the real model and warns loudly on fallback.
"""

from __future__ import annotations

import hashlib
import warnings
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Minimal embedding interface."""

    @property
    def name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def is_semantic(self) -> bool: ...

    def encode(self, texts: list[str]) -> np.ndarray: ...


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return mat / norms


class HashingEmbedder:
    """Deterministic, dependency free character n gram hashing embedder.

    Each text is turned into character 3, 4, and 5 grams (title and body already
    joined by the caller). Every gram is hashed with blake2b; the low bits pick a
    bucket and one more bit picks a sign, giving a signed feature vector. Counts are
    log damped and the vector is L2 normalized so cosine similarity is a dot product.
    """

    def __init__(self, dimension: int = 512, ngram_range: tuple[int, int] = (3, 5)) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = int(dimension)
        self._lo, self._hi = ngram_range

    @property
    def name(self) -> str:
        return f"hashing-blake2b-{self._lo}-{self._hi}gram-d{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def is_semantic(self) -> bool:
        return False

    def _grams(self, text: str) -> list[str]:
        s = f" {text.lower().strip()} "
        grams: list[str] = []
        for n in range(self._lo, self._hi + 1):
            if len(s) < n:
                continue
            for i in range(len(s) - n + 1):
                grams.append(s[i: i + n])
        return grams

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dimension, dtype=np.float64)
        for gram in self._grams(text):
            digest = hashlib.blake2b(gram.encode(
                "utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "big")
            bucket = h % self._dimension
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[bucket] += sign
        # Log damping preserves sign while compressing large magnitudes.
        vec = np.sign(vec) * np.log1p(np.abs(vec))
        return vec

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float64)
        mat = np.vstack([self._embed_one(t) for t in texts])
        return _l2_normalize(mat)


class SentenceTransformerEmbedder:
    """Wrapper around sentence-transformers all-MiniLM-L6-v2 (deferred import)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None  # lazily constructed
        self._dimension = 384  # known for all-MiniLM-L6-v2 until the model loads

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # deferred import

            self._model = SentenceTransformer(self._model_name)
            _get_dim = getattr(self._model, "get_embedding_dimension",
                               None) or self._model.get_sentence_embedding_dimension
            self._dimension = int(_get_dim())
        return self._model

    @property
    def name(self) -> str:
        return f"sentence-transformers/{self._model_name}"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def is_semantic(self) -> bool:
        return True

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._ensure_model()
        emb = model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float64)


def build_embedder(kind: str = "auto", *, dimension: int = 512) -> Embedder:
    """Build an embedder.

    ``"auto"`` prefers the real semantic model and, if it cannot be imported or the
    model cannot be constructed (for example no network to download weights), warns
    loudly and falls back to the deterministic ``HashingEmbedder``.
    """
    if kind == "hashing":
        return HashingEmbedder(dimension=dimension)
    if kind == "sentence-transformers":
        emb = SentenceTransformerEmbedder()
        emb._ensure_model()  # force load so failures surface here
        return emb
    if kind == "auto":
        try:
            emb = SentenceTransformerEmbedder()
            emb._ensure_model()
            return emb
        except Exception as exc:  # noqa: BLE001 - fallback is the whole point
            warnings.warn(
                "\n"
                "============================================================\n"
                "FALLBACK: sentence-transformers could not be loaded.\n"
                f"Reason: {exc}\n"
                "Using the non semantic HashingEmbedder instead. Reported\n"
                "numbers, especially on the paraphrase split, will be WEAKER\n"
                "and must be read as fallback results, not semantic results.\n"
                "============================================================",
                RuntimeWarning,
                stacklevel=2,
            )
            return HashingEmbedder(dimension=dimension)
    raise ValueError(f"unknown embedder kind: {kind!r}")

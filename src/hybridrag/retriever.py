"""The hybrid retriever tying lexical, dense, and fused retrieval together.

Modes:

* ``lexical``            -- BM25 over chunk text.
* ``dense``              -- cosine over chunk embeddings.
* ``hybrid-rrf``         -- reciprocal rank fusion of the lexical and dense lists.
* ``hybrid-normalized``  -- min-max normalized weighted score fusion.

A candidate pool wider than ``k`` is retrieved from each arm before fusion.

Abstention is thresholded on the dense cosine similarity of the best chunk, not on
the fused score: reciprocal rank fusion scores are position only and carry no
absolute confidence, so they are the wrong signal for deciding whether to answer.

``build_context`` orders the strongest chunks first to mitigate the lost in the
middle effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .bm25 import BM25Index
from .chunking import Chunk, chunk_corpus
from .embeddings import Embedder
from .fusion import normalized_score_fusion, reciprocal_rank_fusion
from .vectorstore import VectorStore

MODES = ("lexical", "dense", "hybrid-rrf", "hybrid-normalized")


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int
    dense_sim: float

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id


@dataclass
class RetrievalResult:
    query: str
    mode: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    abstained: bool = False
    max_dense_sim: float = 0.0

    @property
    def doc_ids(self) -> list[str]:
        """Distinct document ids in retrieved order (deduplicated, order preserving)."""
        seen: set[str] = set()
        out: list[str] = []
        for rc in self.chunks:
            if rc.doc_id not in seen:
                seen.add(rc.doc_id)
                out.append(rc.doc_id)
        return out


class HybridRetriever:
    def __init__(
        self,
        docs: list[dict],
        embedder: Embedder,
        *,
        strategy: str = "overlapping",
        k1: float = 1.5,
        b: float = 0.75,
        abstain_threshold: float = 0.25,
        chunk_kwargs: dict | None = None,
    ) -> None:
        self.embedder = embedder
        self.strategy = strategy
        self.abstain_threshold = abstain_threshold
        self.chunks: list[Chunk] = chunk_corpus(docs, strategy, **(chunk_kwargs or {}))
        self._by_id: dict[str, Chunk] = {c.chunk_id: c for c in self.chunks}

        # Lexical index over chunk indexable_text (title prepended).
        self.bm25 = BM25Index(k1=k1, b=b)
        self.bm25.add_many([(c.chunk_id, c.indexable_text) for c in self.chunks])

        # Dense index.
        vectors = embedder.encode([c.indexable_text for c in self.chunks])
        self.store = VectorStore(dimension=embedder.dimension)
        self.store.add(
            ids=[c.chunk_id for c in self.chunks],
            vectors=vectors,
            metadata=[{"doc_id": c.doc_id, "chunk_id": c.chunk_id} for c in self.chunks],
        )

    # -- arms ---------------------------------------------------------------
    def _lexical(self, query: str, pool: int) -> list[tuple[str, float]]:
        return self.bm25.search(query, k=pool)

    def _dense(self, query: str, pool: int) -> list[tuple[str, float]]:
        qvec = self.embedder.encode([query])[0]
        hits = self.store.search(qvec, k=pool)
        return [(h.id, h.score) for h in hits]

    def _max_dense_sim(self, query: str) -> float:
        qvec = self.embedder.encode([query])[0]
        hits = self.store.search(qvec, k=1)
        return float(hits[0].score) if hits else 0.0

    # -- public -------------------------------------------------------------
    def retrieve(self, query: str, k: int = 5, mode: str = "hybrid-rrf") -> RetrievalResult:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        pool = max(k * 4, 20)

        lex = self._lexical(query, pool)
        dense = self._dense(query, pool)
        dense_sim_by_id = {cid: s for cid, s in dense}
        max_dense_sim = max((s for _, s in dense), default=0.0)

        if mode == "lexical":
            ranked = lex
        elif mode == "dense":
            ranked = dense
        elif mode == "hybrid-rrf":
            fused = reciprocal_rank_fusion(
                [[cid for cid, _ in lex], [cid for cid, _ in dense]]
            )
            ranked = fused
        else:  # hybrid-normalized
            fused = normalized_score_fusion([dict(lex), dict(dense)])
            ranked = fused

        chunks: list[RetrievedChunk] = []
        for rank, (cid, score) in enumerate(ranked[:k], start=1):
            chunk = self._by_id.get(cid)
            if chunk is None:
                continue
            chunks.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(score),
                    rank=rank,
                    dense_sim=float(dense_sim_by_id.get(cid, 0.0)),
                )
            )

        abstained = max_dense_sim < self.abstain_threshold
        return RetrievalResult(
            query=query,
            mode=mode,
            chunks=chunks,
            abstained=abstained,
            max_dense_sim=max_dense_sim,
        )

    def build_context(self, result: RetrievalResult, *, max_chars: int | None = None) -> str:
        """Concatenate retrieved chunks strongest first, within an optional budget."""
        ordered = sorted(result.chunks, key=lambda rc: rc.rank)
        pieces: list[str] = []
        total = 0
        for rc in ordered:
            block = f"[{rc.chunk.title}] {rc.chunk.text}".strip()
            if max_chars is not None and total + len(block) > max_chars and pieces:
                break
            pieces.append(block)
            total += len(block)
        return "\n\n".join(pieces)

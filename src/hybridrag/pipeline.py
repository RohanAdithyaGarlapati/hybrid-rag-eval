"""End to end RAG pipeline: retrieve, abstain or generate, then judge."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from .generator import AnswerGenerator
from .judge import AnthropicJudge
from .retriever import HybridRetriever

ABSTAIN_MESSAGE = "I do not know based on the provided context."


@dataclass
class PipelineResult:
    question: str
    mode: str
    abstained: bool
    max_dense_sim: float
    doc_ids: list[str]
    context: str
    answer: Optional[str]
    answer_source: str  # "generated", "abstained", or "generator-unavailable"
    provider: Optional[str] = None
    model: Optional[str] = None
    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def answer_question(
    retriever: HybridRetriever,
    question: str,
    *,
    k: int = 5,
    mode: str = "hybrid-rrf",
    generator: AnswerGenerator | None = None,
    judge: AnthropicJudge | None = None,
) -> PipelineResult:
    """Run retrieval, then abstain or generate, then optionally judge."""
    generator = generator or AnswerGenerator()
    judge = judge or AnthropicJudge()

    res = retriever.retrieve(question, k=k, mode=mode)
    notes: list[str] = []

    if res.abstained:
        return PipelineResult(
            question=question,
            mode=mode,
            abstained=True,
            max_dense_sim=res.max_dense_sim,
            doc_ids=res.doc_ids,
            context="",
            answer=ABSTAIN_MESSAGE,
            answer_source="abstained",
            notes=["retrieval confidence below threshold; abstained without generating"],
        )

    context = retriever.build_context(res)

    gen = generator.generate(question, context)
    if gen is None:
        notes.append(f"generator unavailable: {generator.reason_unavailable}")
        return PipelineResult(
            question=question,
            mode=mode,
            abstained=False,
            max_dense_sim=res.max_dense_sim,
            doc_ids=res.doc_ids,
            context=context,
            answer=None,
            answer_source="generator-unavailable",
            notes=notes,
        )

    faithfulness = None
    answer_relevance = None
    scores = judge.score(question, context, gen.text)
    if scores is None:
        notes.append(f"judge skipped: {judge.reason_unavailable}")
    else:
        faithfulness = scores.faithfulness
        answer_relevance = scores.answer_relevance

    return PipelineResult(
        question=question,
        mode=mode,
        abstained=False,
        max_dense_sim=res.max_dense_sim,
        doc_ids=res.doc_ids,
        context=context,
        answer=gen.text,
        answer_source="generated",
        provider=gen.provider,
        model=gen.model,
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        notes=notes,
    )

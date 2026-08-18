"""FastAPI service exposing /health, /search, /answer, and /generate with Pydantic validation."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .dataset import load_dataset
from .embeddings import build_embedder
from .generator import AnswerGenerator
from .judge import AnthropicJudge
from .pipeline import answer_question
from .retriever import MODES, HybridRetriever


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query text")
    k: int = Field(5, ge=1, le=50)
    mode: Literal["lexical", "dense", "hybrid-rrf", "hybrid-normalized"] = "hybrid-rrf"


class Hit(BaseModel):
    doc_id: str
    chunk_id: str
    title: str
    score: float
    rank: int


class SearchResponse(BaseModel):
    query: str
    mode: str
    abstained: bool
    max_dense_sim: float
    hits: list[Hit]


class AnswerResponse(BaseModel):
    query: str
    abstained: bool
    context: str
    doc_ids: list[str]


class GenerateResponse(BaseModel):
    question: str
    mode: str
    abstained: bool
    max_dense_sim: float
    doc_ids: list[str]
    context: str
    answer: Optional[str]
    answer_source: str
    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None
    notes: list[str] = Field(default_factory=list)


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    ds = load_dataset()
    embedder = build_embedder("auto")
    return HybridRetriever(ds["corpus"], embedder, strategy="overlapping")


@lru_cache(maxsize=1)
def get_generator() -> AnswerGenerator:
    return AnswerGenerator()


@lru_cache(maxsize=1)
def get_judge() -> AnthropicJudge:
    return AnthropicJudge()


app = FastAPI(title="hybrid-rag-eval", version="0.1.0")


@app.get("/health")
def health() -> dict:
    retriever = get_retriever()
    return {
        "status": "ok",
        "n_chunks": len(retriever.chunks),
        "embedder": retriever.embedder.name,
        "is_semantic": retriever.embedder.is_semantic,
        "modes": list(MODES),
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    retriever = get_retriever()
    if req.mode not in MODES:
        raise HTTPException(status_code=422, detail=f"unknown mode {req.mode!r}")
    result = retriever.retrieve(req.query, k=req.k, mode=req.mode)
    hits = [
        Hit(
            doc_id=rc.chunk.doc_id,
            chunk_id=rc.chunk.chunk_id,
            title=rc.chunk.title,
            score=rc.score,
            rank=rc.rank,
        )
        for rc in result.chunks
    ]
    return SearchResponse(
        query=result.query,
        mode=result.mode,
        abstained=result.abstained,
        max_dense_sim=result.max_dense_sim,
        hits=hits,
    )


@app.post("/answer", response_model=AnswerResponse)
def answer(req: SearchRequest) -> AnswerResponse:
    retriever = get_retriever()
    result = retriever.retrieve(req.query, k=req.k, mode=req.mode)
    context = "" if result.abstained else retriever.build_context(result)
    return AnswerResponse(
        query=result.query,
        abstained=result.abstained,
        context=context,
        doc_ids=result.doc_ids,
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(req: SearchRequest) -> GenerateResponse:
    retriever = get_retriever()
    generator = get_generator()
    judge = get_judge()
    res = answer_question(
        retriever,
        req.query,
        k=req.k,
        mode=req.mode,
        generator=generator,
        judge=judge,
    )
    return GenerateResponse(**res.to_dict())

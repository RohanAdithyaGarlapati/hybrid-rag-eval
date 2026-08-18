"""Command line interface: index, query, and evaluate subcommands."""

from __future__ import annotations

import argparse
import json

from .dataset import load_dataset
from .embeddings import build_embedder
from .evaluate import main as evaluate_main
from .retriever import HybridRetriever


def _build(strategy: str) -> HybridRetriever:
    ds = load_dataset()
    embedder = build_embedder("auto")
    return HybridRetriever(ds["corpus"], embedder, strategy=strategy)


def cmd_index(args: argparse.Namespace) -> int:
    retriever = _build(args.strategy)
    print(f"Indexed {len(retriever.chunks)} chunks from strategy={args.strategy}")
    print(f"Embedder: {retriever.embedder.name} (is_semantic={retriever.embedder.is_semantic})")
    print(f"BM25 documents (chunks): {retriever.bm25.n_docs}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    retriever = _build(args.strategy)
    result = retriever.retrieve(args.text, k=args.k, mode=args.mode)
    payload = {
        "query": result.query,
        "mode": result.mode,
        "abstained": result.abstained,
        "max_dense_sim": round(result.max_dense_sim, 4),
        "doc_ids": result.doc_ids,
        "hits": [
            {"rank": rc.rank, "doc_id": rc.chunk.doc_id, "title": rc.chunk.title, "score": round(rc.score, 4)}
            for rc in result.chunks
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    forward: list[str] = []
    if args.out:
        forward += ["--out", args.out]
    forward += ["--embedder", args.embedder, "--seed", str(args.seed)]
    return evaluate_main(forward)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hybridrag", description="Hybrid RAG eval CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="build and report the index")
    p_index.add_argument("--strategy", default="overlapping", choices=["fixed", "overlapping", "semantic"])
    p_index.set_defaults(func=cmd_index)

    p_query = sub.add_parser("query", help="run a single query")
    p_query.add_argument("text")
    p_query.add_argument("--k", type=int, default=5)
    p_query.add_argument("--mode", default="hybrid-rrf", choices=["lexical", "dense", "hybrid-rrf", "hybrid-normalized"])
    p_query.add_argument("--strategy", default="overlapping", choices=["fixed", "overlapping", "semantic"])
    p_query.set_defaults(func=cmd_query)

    p_eval = sub.add_parser("evaluate", help="run the full experiment grid")
    p_eval.add_argument("--out", default=None)
    p_eval.add_argument("--embedder", default="auto", choices=["auto", "hashing", "sentence-transformers"])
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

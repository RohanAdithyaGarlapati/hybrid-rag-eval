"""The full evaluation harness.

Runs the experiment grid (lexical vs dense vs hybrid-rrf vs hybrid-normalized,
across fixed / overlapping / semantic chunking), reports recall@1/3/5/10, MRR, and
nDCG overall and by question kind, compares every variant against the BM25 baseline
with paired tests plus Holm correction, sweeps the abstention threshold, and writes
a reproducible Markdown report stamped with the commit SHA, dataset hash, and the
embedder name and its is_semantic flag.
"""

from __future__ import annotations

import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import metrics as M
from .dataset import dataset_hash, load_dataset
from .embeddings import Embedder, build_embedder
from .retriever import HybridRetriever
from .stats import bootstrap_ci, holm_bonferroni, paired_test

CHUNKINGS = ("fixed", "overlapping", "semantic")
MODES = ("lexical", "dense", "hybrid-rrf", "hybrid-normalized")
KS = (1, 3, 5, 10)
BASELINE = ("overlapping", "lexical")
COMPARE_METRIC = "mrr"  # per query reciprocal rank drives the paired comparison


@dataclass
class VariantResult:
    strategy: str
    mode: str
    overall: dict[str, float]
    by_kind: dict[str, dict[str, float]]
    per_query_metric: list[float]  # COMPARE_METRIC per answerable query, aligned


def _commit_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def _evaluate_variant(
    retriever: HybridRetriever, mode: str, questions: list[dict]
) -> VariantResult:
    per_query: list[dict[str, float]] = []
    per_kind_rows: dict[str, list[dict[str, float]]] = {}
    compare_vec: list[float] = []
    for q in questions:
        res = retriever.retrieve(q["text"], k=max(KS), mode=mode)
        m = M.evaluate_query(res.doc_ids, q["gold"], ks=KS)
        per_query.append(m)
        per_kind_rows.setdefault(q["kind"], []).append(m)
        compare_vec.append(m[COMPARE_METRIC])
    overall = M.aggregate(per_query)
    by_kind = {kind: M.aggregate(rows) for kind, rows in per_kind_rows.items()}
    return VariantResult(
        strategy=retriever.strategy,
        mode=mode,
        overall=overall,
        by_kind=by_kind,
        per_query_metric=compare_vec,
    )


def run_experiment(
    data_dir: str | Path | None = None,
    embedder: Embedder | None = None,
    *,
    seed: int = 0,
) -> dict:
    ds = load_dataset(data_dir)
    corpus = ds["corpus"]
    questions = ds["questions"]
    unanswerable = ds["unanswerable"]
    embedder = embedder or build_embedder("auto")

    # Build one retriever per chunking strategy; reuse across modes.
    variants: dict[tuple[str, str], VariantResult] = {}
    retrievers: dict[str, HybridRetriever] = {}
    for strategy in CHUNKINGS:
        retriever = HybridRetriever(corpus, embedder, strategy=strategy)
        retrievers[strategy] = retriever
        for mode in MODES:
            variants[(strategy, mode)] = _evaluate_variant(retriever, mode, questions)

    # Paired comparisons vs the BM25 baseline, then Holm correction.
    baseline = variants[BASELINE]
    comparisons: list[dict] = []
    pvals: list[float] = []
    keys: list[tuple[str, str]] = []
    for key, vr in variants.items():
        if key == BASELINE:
            continue
        test = paired_test(vr.per_query_metric, baseline.per_query_metric)
        ci = bootstrap_ci(vr.per_query_metric, baseline.per_query_metric, seed=seed)
        comparisons.append(
            {
                "variant": key,
                "method": test.method,
                "statistic": test.statistic,
                "pvalue": test.pvalue,
                "effect_size": test.effect_size,
                "effect_label": test.effect_label,
                "ci_low": ci["low"],
                "ci_high": ci["high"],
                "mean_diff": ci["point"],
            }
        )
        pvals.append(test.pvalue)
        keys.append(key)
    adjusted = holm_bonferroni(pvals)
    for comp, adj in zip(comparisons, adjusted):
        comp["pvalue_holm"] = adj

    # Abstention sweep using dense max cosine similarity on the overlapping index.
    dense_retriever = retrievers["overlapping"]
    ans_sims = np.array([dense_retriever._max_dense_sim(q["text"]) for q in questions])
    una_sims = np.array([dense_retriever._max_dense_sim(q["text"]) for q in unanswerable])
    sweep = _abstention_sweep(ans_sims, una_sims)

    return {
        "embedder": {"name": embedder.name, "is_semantic": embedder.is_semantic, "dimension": embedder.dimension},
        "dataset_hash": dataset_hash(ds["data_dir"]),
        "commit_sha": _commit_sha(),
        "n_questions": len(questions),
        "n_unanswerable": len(unanswerable),
        "variants": variants,
        "baseline": BASELINE,
        "compare_metric": COMPARE_METRIC,
        "comparisons": comparisons,
        "abstention": sweep,
        "seed": seed,
    }


def _abstention_sweep(ans_sims: np.ndarray, una_sims: np.ndarray) -> dict:
    """Sweep the dense similarity threshold and pick an operating point.

    Answerable questions should NOT abstain; unanswerable ones SHOULD. A question
    abstains when its max dense similarity is below the threshold.
    """
    all_sims = np.concatenate([ans_sims, una_sims]) if len(una_sims) else ans_sims
    lo, hi = float(all_sims.min()), float(all_sims.max())
    thresholds = np.linspace(lo, hi, 41)
    rows = []
    best = None
    for t in thresholds:
        false_abstain = float(np.mean(ans_sims < t)) if len(ans_sims) else 0.0
        correct_abstain = float(np.mean(una_sims < t)) if len(una_sims) else 0.0
        youden = correct_abstain - false_abstain
        row = {
            "threshold": float(t),
            "correct_abstention_rate": correct_abstain,
            "false_abstention_rate": false_abstain,
            "youden_j": youden,
        }
        rows.append(row)
        if best is None or youden > best["youden_j"]:
            best = row
    return {"sweep": rows, "operating_point": best}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(x: float) -> str:
    return f"{x:.3f}"


def render_report(results: dict) -> str:
    emb = results["embedder"]
    lines: list[str] = []
    lines.append("# hybrid-rag-eval experiment report")
    lines.append("")
    lines.append(f"- Commit SHA: `{results['commit_sha']}`")
    lines.append(f"- Dataset SHA256: `{results['dataset_hash']}`")
    lines.append(f"- Embedder: `{emb['name']}` (dimension {emb['dimension']})")
    lines.append(f"- Embedder is_semantic: **{emb['is_semantic']}**")
    lines.append(f"- Answerable questions: {results['n_questions']}  |  Unanswerable: {results['n_unanswerable']}")
    lines.append(f"- Baseline: `{results['baseline'][0]} / {results['baseline'][1]}` (BM25)")
    lines.append(f"- Paired comparison metric: `{results['compare_metric']}`  |  seed: {results['seed']}")
    lines.append("")

    if not emb["is_semantic"]:
        lines.append("> **Honesty note.** These numbers were produced by the NON semantic "
                     "HashingEmbedder fallback, not by a trained sentence model. The paraphrase "
                     "split, which needs meaning level matching, is therefore expected to be "
                     "weaker than it would be with a real dense encoder. This is expected and "
                     "reported honestly, not a bug.")
        lines.append("")

    # Overall metrics table.
    lines.append("## Overall metrics by variant")
    lines.append("")
    header = "| chunking | mode | recall@1 | recall@3 | recall@5 | recall@10 | MRR | nDCG@10 |"
    lines.append(header)
    lines.append("|" + "---|" * 8)
    for strategy in CHUNKINGS:
        for mode in MODES:
            vr = results["variants"][(strategy, mode)]
            o = vr.overall
            lines.append(
                f"| {strategy} | {mode} | {_fmt(o['recall@1'])} | {_fmt(o['recall@3'])} | "
                f"{_fmt(o['recall@5'])} | {_fmt(o['recall@10'])} | {_fmt(o['mrr'])} | {_fmt(o['ndcg@10'])} |"
            )
    lines.append("")

    # By kind (recall@5 and MRR).
    lines.append("## By question kind (recall@5 / MRR)")
    lines.append("")
    lines.append("| chunking | mode | lexical recall@5 | lexical MRR | paraphrase recall@5 | paraphrase MRR |")
    lines.append("|" + "---|" * 6)
    for strategy in CHUNKINGS:
        for mode in MODES:
            vr = results["variants"][(strategy, mode)]
            lex = vr.by_kind.get("lexical", {})
            par = vr.by_kind.get("paraphrase", {})
            lines.append(
                f"| {strategy} | {mode} | {_fmt(lex.get('recall@5', 0.0))} | {_fmt(lex.get('mrr', 0.0))} | "
                f"{_fmt(par.get('recall@5', 0.0))} | {_fmt(par.get('mrr', 0.0))} |"
            )
    lines.append("")

    # Statistical comparisons.
    lines.append("## Paired comparisons vs BM25 baseline (Holm corrected)")
    lines.append("")
    lines.append("| variant | test | p (raw) | p (Holm) | effect r | label | mean diff | 95% CI |")
    lines.append("|" + "---|" * 8)
    for comp in sorted(results["comparisons"], key=lambda c: c["pvalue_holm"]):
        vk = f"{comp['variant'][0]} / {comp['variant'][1]}"
        lines.append(
            f"| {vk} | {comp['method']} | {_fmt(comp['pvalue'])} | {_fmt(comp['pvalue_holm'])} | "
            f"{_fmt(comp['effect_size'])} | {comp['effect_label']} | {_fmt(comp['mean_diff'])} | "
            f"[{_fmt(comp['ci_low'])}, {_fmt(comp['ci_high'])}] |"
        )
    lines.append("")

    # Abstention.
    op = results["abstention"]["operating_point"]
    lines.append("## Abstention operating point")
    lines.append("")
    lines.append(f"- Chosen threshold (dense cosine): **{_fmt(op['threshold'])}**")
    lines.append(f"- Correct abstention rate (unanswerable caught): {_fmt(op['correct_abstention_rate'])}")
    lines.append(f"- False abstention rate (answerable wrongly refused): {_fmt(op['false_abstention_rate'])}")
    lines.append(f"- Youden J (correct minus false): {_fmt(op['youden_j'])}")
    lines.append("")
    lines.append("Abstention is thresholded on dense cosine similarity, not on the fused RRF "
                 "score, because RRF scores are position only and carry no absolute confidence signal.")
    lines.append("")
    return "\n".join(lines)


def write_report(results: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(results), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the hybrid-rag-eval experiment grid.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--embedder", default="auto", choices=["auto", "hashing", "sentence-transformers"])
    parser.add_argument("--out", default=None, help="report output path")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        embedder = build_embedder(args.embedder)
    results = run_experiment(args.data_dir, embedder, seed=args.seed)

    out = args.out
    if out is None:
        repo_root = Path(__file__).resolve().parents[2]
        out = repo_root / "reports" / "experiment_report.md"
    written = write_report(results, out)

    # Console summary.
    emb = results["embedder"]
    print(f"Embedder: {emb['name']} (is_semantic={emb['is_semantic']})")
    for strategy in CHUNKINGS:
        for mode in MODES:
            o = results["variants"][(strategy, mode)].overall
            print(f"  {strategy:12s} {mode:18s} recall@5={o['recall@5']:.3f} "
                  f"mrr={o['mrr']:.3f} ndcg@10={o['ndcg@10']:.3f}")
    op = results["abstention"]["operating_point"]
    print(f"Abstention threshold={op['threshold']:.3f} correct={op['correct_abstention_rate']:.3f} "
          f"false={op['false_abstention_rate']:.3f}")
    print(f"Report written to {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

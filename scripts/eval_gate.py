"""CI quality gate.

Runs the evaluation and fails (non zero exit) if the primary variant's overall
recall@5 falls below the committed threshold. This is what the CI workflow uses to
block a regression from merging.

The committed threshold is intentionally conservative so it holds even when the
non semantic HashingEmbedder fallback is used (no network for the real model).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Make ``src`` importable when run directly from the repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybridrag.embeddings import build_embedder  # noqa: E402
from hybridrag.evaluate import run_experiment, write_report  # noqa: E402

# Committed quality bar: overall recall@5 of the primary variant.
GATE_VARIANT = ("overlapping", "hybrid-rrf")
RECALL5_THRESHOLD = 0.80


def main() -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        embedder = build_embedder("auto")
    results = run_experiment(embedder=embedder, seed=0)
    write_report(results, ROOT / "reports" / "experiment_report.md")

    recall5 = results["variants"][GATE_VARIANT].overall["recall@5"]
    emb = results["embedder"]
    print(f"Embedder: {emb['name']} (is_semantic={emb['is_semantic']})")
    print(f"Gate variant {GATE_VARIANT} recall@5 = {recall5:.3f} (threshold {RECALL5_THRESHOLD:.2f})")

    if recall5 < RECALL5_THRESHOLD:
        print("QUALITY GATE FAILED: recall@5 dropped below the committed threshold.")
        return 1
    print("QUALITY GATE PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

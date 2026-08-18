"""Statistical comparison utilities for paired retrieval evaluation.

* ``paired_test`` runs a paired Wilcoxon signed rank test, falls back to the
  Mann-Whitney U test when Wilcoxon cannot run, and cleanly skips when every pair
  ties (zero difference everywhere).
* ``rank_biserial`` reports a rank based effect size with a conventional label.
* ``bootstrap_ci`` computes a seeded percentile confidence interval by resampling
  the paired observations together.
* ``holm_bonferroni`` applies the step down correction with monotone adjusted p values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    method: str
    statistic: float | None
    pvalue: float
    n: int
    effect_size: float
    effect_label: str


def _effect_label(r: float) -> str:
    a = abs(r)
    if a < 0.1:
        return "negligible"
    if a < 0.3:
        return "small"
    if a < 0.5:
        return "medium"
    return "large"


def rank_biserial(a, b) -> float:
    """Matched pairs rank biserial correlation from signed ranks of the differences.

    ``r = (W_plus - W_minus) / (W_plus + W_minus)`` over the nonzero differences,
    where ``W_plus`` and ``W_minus`` are the sums of ranks of positive and negative
    differences. Returns 0.0 when all pairs tie.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    nz = diff[diff != 0.0]
    if nz.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nz))
    w_plus = float(ranks[nz > 0].sum())
    w_minus = float(ranks[nz < 0].sum())
    total = w_plus + w_minus
    if total == 0.0:
        return 0.0
    return (w_plus - w_minus) / total


def paired_test(a, b) -> TestResult:
    """Paired Wilcoxon signed rank test with graceful fallbacks."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired samples must have equal length")
    n = a.size
    diff = a - b
    r = rank_biserial(a, b)

    # All pairs tie: nothing to test.
    if np.all(diff == 0.0):
        return TestResult("skipped-all-tied", None, 1.0, n, 0.0, "negligible")

    # Wilcoxon needs at least one nonzero difference; try it first.
    try:
        res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return TestResult("wilcoxon", float(res.statistic), float(res.pvalue), n, r, _effect_label(r))
    except ValueError:
        pass

    # Fall back to the unpaired Mann-Whitney U test.
    try:
        res = stats.mannwhitneyu(a, b, alternative="two-sided")
        return TestResult("mann-whitney-u", float(res.statistic), float(res.pvalue), n, r, _effect_label(r))
    except ValueError:
        return TestResult("skipped-degenerate", None, 1.0, n, r, _effect_label(r))


def bootstrap_ci(
    a,
    b,
    *,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Seeded percentile bootstrap CI for the mean paired difference (a - b).

    Pairs are resampled together (the same index chosen for both a and b) so the
    pairing is preserved. Returns the point estimate and the interval bounds.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired samples must have equal length")
    diff = a - b
    n = diff.size
    point = float(diff.mean()) if n else 0.0
    if n == 0:
        return {"point": 0.0, "low": 0.0, "high": 0.0, "ci": ci}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, 100.0 * alpha))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return {"point": point, "low": low, "high": high, "ci": ci}


def holm_bonferroni(pvalues) -> list[float]:
    """Holm-Bonferroni step down adjusted p values, in the input order.

    Sorts ascending, scales each by the number of remaining tests, enforces a
    monotonically nondecreasing sequence, clips to 1.0, then restores order.
    """
    p = list(pvalues)
    m = len(p)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running_max = max(running_max, val)
        adjusted[idx] = min(1.0, running_max)
    return adjusted

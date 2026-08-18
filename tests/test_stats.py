import numpy as np

from hybridrag.stats import bootstrap_ci, holm_bonferroni, paired_test, rank_biserial


def test_paired_test_all_tied_skips():
    a = [0.5, 0.5, 0.5]
    res = paired_test(a, a)
    assert res.method.startswith("skipped")
    assert res.pvalue == 1.0
    assert res.effect_label == "negligible"


def test_paired_test_clear_difference_significant():
    a = [0.9, 0.85, 0.95, 0.88, 0.92, 0.9, 0.87, 0.93]
    b = [0.4, 0.35, 0.45, 0.38, 0.42, 0.4, 0.37, 0.43]
    res = paired_test(a, b)
    assert res.method in ("wilcoxon", "mann-whitney-u")
    assert res.pvalue < 0.05
    assert res.effect_label in ("medium", "large")


def test_paired_test_single_sample_edge_case():
    # A single pair with a nonzero difference must not crash and returns valid p.
    res = paired_test([1.0], [0.0])
    assert 0.0 <= res.pvalue <= 1.0
    assert res.n == 1


def test_rank_biserial_sign_and_range():
    a = [1.0, 1.0, 1.0, 1.0]
    b = [0.0, 0.0, 0.0, 0.0]
    r = rank_biserial(a, b)
    assert abs(r - 1.0) < 1e-9
    assert rank_biserial(b, a) == -r


def test_rank_biserial_all_tied_is_zero():
    assert rank_biserial([2.0, 2.0], [2.0, 2.0]) == 0.0


def test_bootstrap_ci_reproducible_with_seed():
    rng = np.random.default_rng(1)
    a = rng.normal(0.6, 0.1, size=40)
    b = rng.normal(0.5, 0.1, size=40)
    ci1 = bootstrap_ci(a, b, seed=123)
    ci2 = bootstrap_ci(a, b, seed=123)
    assert ci1 == ci2
    # Point estimate equals the observed mean difference.
    assert abs(ci1["point"] - float(np.mean(a - b))) < 1e-12
    # Interval brackets the point estimate.
    assert ci1["low"] <= ci1["point"] <= ci1["high"]


def test_bootstrap_ci_different_seeds_differ():
    a = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]
    b = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert bootstrap_ci(a, b, seed=1) != bootstrap_ci(a, b, seed=2)


def test_holm_bonferroni_hand_computed():
    # p = [0.01, 0.04, 0.03] -> adjusted [0.03, 0.06, 0.06] with monotone enforcement.
    adj = holm_bonferroni([0.01, 0.04, 0.03])
    assert abs(adj[0] - 0.03) < 1e-12
    assert abs(adj[1] - 0.06) < 1e-12
    assert abs(adj[2] - 0.06) < 1e-12


def test_holm_bonferroni_monotone_and_clipped():
    adj = holm_bonferroni([0.5, 0.5, 0.5])
    # Sorted adjusted values must be nondecreasing and clipped at 1.0.
    s = sorted(adj)
    assert all(s[i] <= s[i + 1] + 1e-12 for i in range(len(s) - 1))
    assert all(v <= 1.0 for v in adj)


def test_holm_bonferroni_empty():
    assert holm_bonferroni([]) == []

import math

from hybridrag import metrics as M


def test_recall_and_hit_single_gold():
    retrieved = ["d1", "d2", "d3", "d4", "d5"]
    gold = ["d3"]
    assert M.recall_at_k(retrieved, gold, 1) == 0.0
    assert M.recall_at_k(retrieved, gold, 3) == 1.0
    assert M.hit_at_k(retrieved, gold, 3) == 1.0
    assert M.hit_at_k(retrieved, gold, 2) == 0.0


def test_precision_at_k():
    retrieved = ["d1", "d2", "d3"]
    gold = ["d3"]
    assert abs(M.precision_at_k(retrieved, gold, 3) - 1 / 3) < 1e-12


def test_reciprocal_rank():
    assert M.reciprocal_rank(["d1", "d2", "d3"], ["d3"]) == 1 / 3
    assert M.reciprocal_rank(["d3", "d2"], ["d3"]) == 1.0
    assert M.reciprocal_rank(["d1", "d2"], ["d9"]) == 0.0


def test_recall_multi_gold():
    retrieved = ["d1", "d2", "d3"]
    gold = ["d1", "d3"]
    assert M.recall_at_k(retrieved, gold, 1) == 0.5
    assert M.recall_at_k(retrieved, gold, 3) == 1.0


def test_ndcg_single_gold_at_rank_three():
    retrieved = ["d1", "d2", "d3"]
    gold = ["d3"]
    # DCG = 1/log2(4) = 0.5 ; IDCG = 1/log2(2) = 1 ; nDCG = 0.5.
    assert abs(M.ndcg_at_k(retrieved, gold, 3) - 0.5) < 1e-12


def test_ndcg_two_gold_hand_computed():
    retrieved = ["d1", "d2", "d3"]
    gold = ["d1", "d3"]
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert abs(M.ndcg_at_k(retrieved, gold, 3) - dcg / idcg) < 1e-12


def test_ndcg_perfect_is_one():
    assert abs(M.ndcg_at_k(["d1"], ["d1"], 1) - 1.0) < 1e-12


def test_evaluate_query_keys_present():
    m = M.evaluate_query(["d3", "d1"], ["d3"], ks=(1, 3, 5, 10))
    for key in ("recall@1", "recall@5", "ndcg@10", "precision@3", "hit@1", "mrr"):
        assert key in m


def test_aggregate_means():
    rows = [{"recall@5": 1.0}, {"recall@5": 0.0}]
    assert M.aggregate(rows)["recall@5"] == 0.5
    assert M.aggregate([]) == {}

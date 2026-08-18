from hybridrag.fusion import normalized_score_fusion, reciprocal_rank_fusion


def test_rrf_matches_hand_computed_values():
    # list1 ranks A,B,C,D ; list2 ranks B,C,D,A ; k = 60, ranks are 1 based.
    list1 = ["A", "B", "C", "D"]
    list2 = ["B", "C", "D", "A"]
    fused = dict(reciprocal_rank_fusion([list1, list2], k=60))
    expected = {
        "A": 1 / 61 + 1 / 64,
        "B": 1 / 62 + 1 / 61,
        "C": 1 / 63 + 1 / 62,
        "D": 1 / 64 + 1 / 63,
    }
    for key, val in expected.items():
        assert abs(fused[key] - val) < 1e-12
    # B is the unique winner under these lists.
    assert reciprocal_rank_fusion([list1, list2], k=60)[0][0] == "B"


def test_rrf_weights_shift_ranking():
    list1 = ["A", "B"]
    list2 = ["B", "A"]
    # Heavier weight on list1 should push A above B.
    fused = reciprocal_rank_fusion([list1, list2], weights=[3.0, 1.0], k=60)
    assert fused[0][0] == "A"


def test_rrf_default_k_is_sixty():
    fused_default = reciprocal_rank_fusion([["A"]])
    fused_explicit = reciprocal_rank_fusion([["A"]], k=60)
    assert abs(fused_default[0][1] - fused_explicit[0][1]) < 1e-15


def test_normalized_fusion_min_max():
    # One map with a clear spread, another empty; normalization to [0,1].
    m1 = {"A": 10.0, "B": 0.0, "C": 5.0}
    fused = dict(normalized_score_fusion([m1]))
    assert abs(fused["A"] - 1.0) < 1e-12
    assert abs(fused["B"] - 0.0) < 1e-12
    assert abs(fused["C"] - 0.5) < 1e-12


def test_normalized_fusion_tie_handling_all_equal():
    # A degenerate map where every score is equal maps everything to 1.0.
    m = {"A": 4.0, "B": 4.0, "C": 4.0}
    fused = dict(normalized_score_fusion([m]))
    assert all(abs(v - 1.0) < 1e-12 for v in fused.values())


def test_normalized_fusion_weighted_sum():
    m1 = {"A": 1.0, "B": 0.0}
    m2 = {"A": 0.0, "B": 1.0}
    fused = dict(normalized_score_fusion([m1, m2], weights=[2.0, 1.0]))
    # A gets 2*1 + 1*0 = 2 ; B gets 2*0 + 1*1 = 1.
    assert abs(fused["A"] - 2.0) < 1e-12
    assert abs(fused["B"] - 1.0) < 1e-12

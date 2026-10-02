"""Saved G0 evidence includes each required numerical comparison."""

from lfu_tree_self_play.g0 import run_g0_verification


def test_report_covers_both_colors_and_all_mathematical_conditions() -> None:
    report = run_g0_verification(seed=5, pairs=500)

    assert report["passed"] is True
    assert report["seed"] == 5
    assert report["pairs_per_color"] == 500
    for color in ("black", "white"):
        comparison = report["conditions"][color]
        assert comparison["passed"] is True
        assert comparison["value_error"] <= comparison["value_tolerance"]
        assert comparison["gradient_passed"] is True
        assert comparison["oracle_agrees_with_exact"] is True
    assert report["conditions"]["importance_ratio"]["passed"] is True
    assert report["conditions"]["leave_one_out"]["passed"] is True
    assert report["conditions"]["shared_recomputed"]["passed"] is True
    for case in report["conditions"]["shared_recomputed"][
        "distinct_and_duplicate_siblings"
    ]:
        assert case["branch_returns"][0] != case["branch_returns"][1]
        assert case["clipped_edge_count"] > 0
        assert case["shared_edge_count"] + 1 == case["recomputed_edge_count"]

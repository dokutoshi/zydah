"""Policy comparison: pi_0 vs TD vs MC on separable synthetic items."""

from __future__ import annotations

from eval.bellman_eval import compare_policies, synthetic_items


def test_comparison_is_deterministic():
    items = synthetic_items(n=40, seed=0)
    assert compare_policies(items) == compare_policies(items)


def test_td_learns_to_escalate_confident_wrong_bucket():
    report = compare_policies(synthetic_items(n=40, seed=0))
    # pi_0 commits everything here (width/conflict under thresholds).
    assert report["pi_0"]["coverage"] == 1.0
    assert report["pi_0"]["accuracy_committed"] < 1.0
    # TD separates by width bucket: committed accuracy improves,
    # paid for with escalations (coverage drops, cost rises).
    assert report["td"]["accuracy_committed"] > report["pi_0"]["accuracy_committed"]
    assert report["td"]["escalate_rate"] > report["pi_0"]["escalate_rate"]
    assert report["mc"]["accuracy_committed"] >= report["pi_0"]["accuracy_committed"]
    assert report["q_states_td"] >= 2

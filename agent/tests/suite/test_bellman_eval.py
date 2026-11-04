"""Synthetic separable Bellman smoke check (arithmetic, not learnability)."""

from __future__ import annotations

from eval.bellman_eval import compare_policies, synthetic_items


def test_comparison_is_deterministic():
    items = synthetic_items(n=40, seed=0)
    assert compare_policies(items) == compare_policies(items)


def test_synthetic_separable_smoke_check_separates_by_construction():
    report = compare_policies(synthetic_items(n=40, seed=0))
    # pi_0 commits everything here (width/conflict under thresholds).
    assert report["pi_0"]["coverage"] == 1.0
    assert report["pi_0"]["accuracy_committed"] < 1.0
    # The fixture's label is a function of width bucket, so the TD/MC
    # arithmetic separates it by construction. This is not evidence
    # that learned routing beats the threshold policy on real data.
    assert report["td"]["accuracy_committed"] > report["pi_0"]["accuracy_committed"]
    assert report["td"]["escalate_rate"] > report["pi_0"]["escalate_rate"]
    assert report["mc"]["accuracy_committed"] >= report["pi_0"]["accuracy_committed"]
    assert report["q_states_td"] >= 2

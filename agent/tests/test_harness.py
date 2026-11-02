"""Harness baseline tests (synthetic masses only, no API calls)."""

import pytest

from eval.harness import (
    accuracy,
    average_scores,
    compare,
    decision_flip_rate,
    dempster_verdict,
    majority_vote,
    zydah_verdict,
)
from zydah.fusion import make_mass
from zydah.judges import SeededMockPanel

FRAME = frozenset({"pass", "partial", "fail"})
OMEGA = ("pass", "partial", "fail")


def test_majority_vote_with_deterministic_tiebreak():
    assert majority_vote(["pass", "fail", "pass"]) == "pass"
    assert majority_vote(["pass", "fail"]) == "fail"  # tie -> sorted-first


def test_average_scores_argmax_mean():
    scores = [
        {"pass": 0.9, "partial": 0.1, "fail": 0.0},
        {"pass": 0.4, "partial": 0.5, "fail": 0.1},
    ]
    assert average_scores(scores) == "pass"


def test_dempster_verdict_follows_agreement():
    masses = [
        make_mass(FRAME, {"pass": 0.8, OMEGA: 0.2}),
        make_mass(FRAME, {"pass": 0.7, OMEGA: 0.3}),
    ]
    assert dempster_verdict(masses, FRAME) == "pass"


def test_zydah_commits_on_agreement_escalates_on_conflict():
    agree = [
        make_mass(FRAME, {"pass": 0.85, OMEGA: 0.15}),
        make_mass(FRAME, {"pass": 0.85, OMEGA: 0.15}),
    ]
    out = zydah_verdict(agree, FRAME, tau=0.80)
    assert out["verdict"] == "pass" and out["decision"] == "commit"

    zadeh = [
        make_mass(FRAME, {"pass": 0.99, "partial": 0.01}),
        make_mass(FRAME, {"partial": 0.01, "fail": 0.99}),
    ]
    out = zydah_verdict(zadeh, FRAME, tau=0.80)
    assert out["decision"] == "escalate"


def test_majority_vote_ships_what_zydah_escalates():
    """Demo in miniature: 2-of-3 judges say pass, but fused belief is too
    conflicted to commit at tau = 0.80."""
    labels = ["pass", "pass", "fail"]
    assert majority_vote(labels) == "pass"
    masses = [
        make_mass(FRAME, {"pass": 0.55, OMEGA: 0.45}),
        make_mass(FRAME, {"pass": 0.55, OMEGA: 0.45}),
        make_mass(FRAME, {"fail": 0.95, OMEGA: 0.05}),
    ]
    out = zydah_verdict(masses, FRAME, tau=0.80)
    assert out["decision"] == "escalate"


def test_decision_flip_rate():
    assert decision_flip_rate(["a", "b", "a"], ["a", "a", "a"]) == pytest.approx(1 / 3)
    assert decision_flip_rate(["a", "a"], ["a", "a"]) == 0.0


def test_accuracy_ignores_abstentions_but_reports_coverage():
    out = accuracy(["pass", None, "fail"], ["pass", "pass", "fail"])
    assert out["accuracy_committed"] == pytest.approx(1.0)
    assert out["coverage"] == pytest.approx(2 / 3)
    assert out["n"] == 3


def test_compare_is_deterministic_and_covers_all_methods():
    panel = SeededMockPanel(FRAME, seed=42)
    items = [
        {
            "id": f"item{i}",
            "reference": "pass",
            "runs": [panel.elicit(f"item{i}", run=r) for r in (0, 1)],
        }
        for i in range(5)
    ]
    out1 = compare(items, FRAME, tau=0.80, seed=42)
    out2 = compare(items, FRAME, tau=0.80, seed=42)
    assert out1 == out2
    assert set(out1["metrics"]) == {"majority_vote", "average", "dempster", "zydah"}
    assert out1["manifest"] == {
        "seed": 42,
        "tau": 0.80,
        "frame": ["fail", "partial", "pass"],
        "rule": "yager",
        "methods": ["majority_vote", "average", "dempster", "zydah"],
    }
    for metrics in out1["metrics"].values():
        assert 0.0 <= metrics["flip_rate"] <= 1.0
        assert 0.0 <= metrics["coverage"] <= 1.0

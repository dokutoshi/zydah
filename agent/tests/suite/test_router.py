"""Router tests: commit if Bel >= tau, reject if Pl < tau, else escalate."""

import pytest

from zydah.fusion import make_mass
from zydah.router import (
    TAU_GENERAL,
    TAU_HIGH_STAKES,
    decide_best,
    decide_hypothesis,
    decide_interval,
)


def test_decide_interval_trichotomy():
    assert decide_interval(0.85, 1.0, 0.80) == "commit"
    assert decide_interval(0.10, 0.50, 0.80) == "reject"
    assert decide_interval(0.60, 0.90, 0.80) == "escalate"


def test_decide_interval_boundaries():
    assert decide_interval(0.80, 0.80, 0.80) == "commit"  # Bel >= tau wins
    assert decide_interval(0.50, 0.80, 0.80) == "escalate"  # Pl == tau straddles


def test_threshold_defaults():
    assert TAU_GENERAL == 0.80
    assert TAU_HIGH_STAKES == 0.95


def test_decide_hypothesis_reports_interval():
    frame = frozenset({"pass", "fail"})
    m = make_mass(frame, {"pass": 0.6, ("pass", "fail"): 0.4})
    out = decide_hypothesis(m, frozenset({"pass"}), TAU_GENERAL)
    assert out["bel"] == pytest.approx(0.6)
    assert out["pl"] == pytest.approx(1.0)
    assert out["width"] == pytest.approx(0.4)
    assert out["decision"] == "escalate"


def test_high_stakes_escalates_where_general_commits():
    frame = frozenset({"pass", "fail"})
    m = make_mass(frame, {"pass": 0.9, ("pass", "fail"): 0.1})
    assert decide_hypothesis(m, frozenset({"pass"}), TAU_GENERAL)["decision"] == "commit"
    assert decide_hypothesis(m, frozenset({"pass"}), TAU_HIGH_STAKES)["decision"] == "escalate"


def test_decide_best_escalates_on_zadeh_like_conflict():
    """Yager-fused Zadeh masses: best singleton is 'b' but [Bel, Pl] =
    [0.0001, 1.0] straddles tau -> escalate, where Dempster would commit."""
    frame = frozenset({"a", "b", "c"})
    m = make_mass(frame, {"b": 0.0001, ("a", "b", "c"): 0.9999})
    out = decide_best(m, frame, TAU_GENERAL)
    assert out["best"] == "b"
    assert out["decision"] == "escalate"


def test_decide_best_commits_on_agreement():
    frame = frozenset({"pass", "partial", "fail"})
    m = make_mass(frame, {"pass": 0.9, ("pass", "partial", "fail"): 0.1})
    out = decide_best(m, frame, TAU_GENERAL)
    assert out["best"] == "pass"
    assert out["decision"] == "commit"

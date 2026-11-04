"""Interface tests: [muse] direct vs [zydah] fused pipe-through (mock only)."""

import pytest

from zydah.fusion import make_mass
from zydah.interface import (
    direct_verdict,
    format_tagged,
    fuse_response,
    fused_verdict,
)
from zydah.judges import SeededMockPanel

FRAME = frozenset({"pass", "partial", "fail"})
OMEGA = ("pass", "partial", "fail")


def mock_panel() -> SeededMockPanel:
    """The deterministic offline panel, injected explicitly."""
    return SeededMockPanel(FRAME, seed=0)


class StubPanel:
    """Panel returning fixed masses regardless of evidence."""

    def __init__(self, masses):
        self._masses = masses

    def elicit(self, item_id="", run=0):
        return list(self._masses)


def test_fuse_response_requires_explicit_panel():
    with pytest.raises(TypeError, match="panel"):
        fuse_response("some agent output", FRAME)  # type: ignore[call-arg]


def test_fuse_response_has_both_tagged_arms():
    out = fuse_response("some agent output", FRAME, mock_panel())
    assert set(out) == {"muse", "zydah"}
    assert out["muse"]["verdict"] in FRAME
    assert out["zydah"]["decision"] in {"commit", "reject", "escalate"}


def test_fuse_response_is_deterministic_per_evidence():
    assert fuse_response("ev-1", FRAME, mock_panel()) == fuse_response(
        "ev-1", FRAME, mock_panel()
    )


def test_format_tagged_renders_both_tags():
    text = format_tagged(fuse_response("ev-1", FRAME, mock_panel()))
    assert "[muse]" in text
    assert "[zydah]" in text
    # Attribution by cycle: [muse] is the first response.
    assert "first response" in text.split("[zydah]")[0]


def test_format_tagged_includes_verifier_outcome():
    result = fuse_response("ev-1", FRAME, mock_panel())
    verification = {"verdict": "good", "followups": 2, "model_id": "m1"}
    text = format_tagged(result, verification)
    assert "verifier: good followups=2 model=m1" in text.split("[zydah]")[1]


def test_direct_arm_uses_lead_judge_only():
    masses = [
        make_mass(FRAME, {"pass": 0.9, OMEGA: 0.1}),
        make_mass(FRAME, {"fail": 0.9, OMEGA: 0.1}),
        make_mass(FRAME, {"fail": 0.9, OMEGA: 0.1}),
    ]
    assert direct_verdict(masses, FRAME)["verdict"] == "pass"


def test_fused_arm_escalates_on_zadeh_conflict_where_direct_commits():
    panel = StubPanel([
        make_mass(FRAME, {"pass": 0.99, "partial": 0.01}),
        make_mass(FRAME, {"partial": 0.01, "fail": 0.99}),
    ])
    out = fuse_response("conflicting case", FRAME, panel=panel, tau=0.80)
    assert out["muse"]["verdict"] == "pass"
    assert out["zydah"]["decision"] == "escalate"
    assert out["zydah"]["conflict"] == pytest.approx(0.9999)


def test_fused_arm_tbm_total_conflict_abstains_escalates():
    masses = [
        make_mass(FRAME, {"pass": 1.0}),
        make_mass(FRAME, {"fail": 1.0}),
    ]
    out = fused_verdict(masses, FRAME, rule="tbm")
    assert out["verdict"] is None
    assert out["decision"] == "escalate"
    assert out["conflict"] == pytest.approx(1.0)


def test_unknown_rule_rejected():
    with pytest.raises(ValueError, match="unknown fusion rule"):
        fuse_response("x", FRAME, mock_panel(), rule="bogus")

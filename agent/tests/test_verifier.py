"""Verifier tests: width/conflict validation + System 2 loop (mock only)."""

import pytest

from zydah.fusion import make_mass
from zydah.verifier import ZydahVerifier, format_transcript, load_verifier_config

FRAME = frozenset({"pass", "partial", "fail"})
OMEGA = ("pass", "partial", "fail")


class SequencePanel:
    """Returns the next mass set on each elicit call."""

    def __init__(self, mass_sets):
        self._mass_sets = list(mass_sets)
        self.calls = 0

    def elicit(self, item_id="", run=0):
        idx = min(self.calls, len(self._mass_sets) - 1)
        self.calls += 1
        return list(self._mass_sets[idx])


def wide_masses():
    return [
        make_mass(FRAME, {"pass": 0.6, OMEGA: 0.4}),
        make_mass(FRAME, {"partial": 0.6, OMEGA: 0.4}),
    ]


def narrow_masses():
    return [
        make_mass(FRAME, {"pass": 0.9, OMEGA: 0.1}),
        make_mass(FRAME, {"pass": 0.9, OMEGA: 0.1}),
    ]


def test_default_config_thresholds_come_from_yaml():
    cfg = load_verifier_config()
    assert cfg.width_threshold == pytest.approx(0.35)
    assert cfg.conflict_threshold == pytest.approx(0.35)


def test_validator_flags_wide_interval_as_speculative():
    verifier = ZydahVerifier(FRAME, panel=SequencePanel([wide_masses()]))
    out = verifier.validate({"verdict": "pass", "width": 0.85, "conflict": 0.10})
    assert out["verdict"] is False
    assert "question" in out


def test_validator_accepts_narrow_low_conflict_verdict():
    verifier = ZydahVerifier(FRAME, panel=SequencePanel([narrow_masses()]))
    out = verifier.validate({"verdict": "pass", "width": 0.05, "conflict": 0.02})
    assert out["verdict"] == "good"
    assert "question" not in out


def test_no_responder_returns_prompt_and_question():
    verifier = ZydahVerifier(FRAME, panel=SequencePanel([wide_masses()]))
    out = verifier.verify("Is it cloudy today in Ohio?")
    assert out["verdict"] is False
    assert out["question"]
    assert out["question"] in out["prompt"]
    assert "Ohio" in out["prompt"]


def test_iteration_with_responder_reduces_uncertainty_to_good():
    seen = {}

    def responder(prompt, model_id):
        seen["prompt"] = prompt
        seen["model_id"] = model_id
        return "Follow-up evidence strongly supports pass."

    panel = SequencePanel([wide_masses(), narrow_masses()])
    verifier = ZydahVerifier(FRAME, panel=panel, responder=responder, model_id="test-model")
    out = verifier.verify("Is it cloudy today in Ohio?")
    assert out["verdict"] == "good"
    assert out["followups"] == 1
    assert seen["model_id"] == "test-model"
    assert out["history"][0]["validation"]["verdict"] is False
    assert panel.calls == 2


def test_model_id_from_env(monkeypatch):
    cfg = load_verifier_config()
    monkeypatch.setenv(cfg.model.env_var, "env-model")
    verifier = ZydahVerifier(FRAME, panel=SequencePanel([narrow_masses()]), config=cfg)
    assert verifier.model_id == "env-model"


class SequenceElicitor:
    """Returns the next single mass per call (direct path, no panel)."""

    def __init__(self, masses):
        self._masses = list(masses)
        self.calls = 0

    def __call__(self, evidence, frame, model_id):
        idx = min(self.calls, len(self._masses) - 1)
        self.calls += 1
        return self._masses[idx]


def wide_mass():
    return make_mass(FRAME, {"pass": 0.5, OMEGA: 0.5})


def narrow_mass():
    return make_mass(FRAME, {"pass": 0.95, OMEGA: 0.05})


def test_verify_direct_bypasses_panel_and_accepts_narrow():
    class ExplodingPanel:
        def elicit(self, item_id="", run=0):
            raise AssertionError("panel must not be called on the direct path")

    verifier = ZydahVerifier(
        FRAME,
        panel=ExplodingPanel(),
        mass_elicitor=SequenceElicitor([narrow_mass()]),
    )
    out = verifier.verify_direct("claim")
    assert out["verdict"] == "good"
    assert out["mode"] == "direct"
    assert out["followups"] == 0


def test_verify_direct_iterates_with_responder_to_good():
    seen = {}

    def responder(prompt, model_id):
        seen["prompt"] = prompt
        return "Extra evidence for pass."

    elicitor = SequenceElicitor([wide_mass(), narrow_mass()])
    verifier = ZydahVerifier(
        FRAME, responder=responder, mass_elicitor=elicitor, model_id="m-direct"
    )
    out = verifier.verify_direct("Is it cloudy today in Ohio?")
    assert out["verdict"] == "good"
    assert out["followups"] == 1
    assert elicitor.calls == 2
    assert "Ohio" in seen["prompt"]


def test_transcript_renders_prompt_response_pairs():
    def responder(prompt, model_id):
        return "Extra evidence for pass."

    elicitor = SequenceElicitor([wide_mass(), wide_mass(), narrow_mass()])
    verifier = ZydahVerifier(FRAME, responder=responder, mass_elicitor=elicitor)
    out = verifier.verify_direct("claim")
    assert out["verdict"] == "good"
    transcript = format_transcript(out["history"])
    assert "[Prompt 1]" in transcript and "[Response 1]" in transcript
    assert "[Prompt 2]" in transcript and "[Response 2]" in transcript
    assert "[Prompt 3]" not in transcript  # final pass was good


def test_transcript_empty_without_followups():
    verifier = ZydahVerifier(FRAME, mass_elicitor=SequenceElicitor([narrow_mass()]))
    out = verifier.verify_direct("claim")
    assert format_transcript(out["history"]) == ""


def test_verify_direct_no_responder_returns_prompt():
    verifier = ZydahVerifier(FRAME, mass_elicitor=SequenceElicitor([wide_mass()]))
    out = verifier.verify_direct("claim")
    assert out["verdict"] is False
    assert out["question"] in out["prompt"]
    assert out["mode"] == "direct"


def test_iteration_budget_exhausted_returns_false():
    def responder(prompt, model_id):
        return "Still ambiguous."

    panel = SequencePanel([wide_masses()])
    verifier = ZydahVerifier(FRAME, panel=panel, responder=responder)
    out = verifier.verify("Is it cloudy today in Ohio?")
    assert out["verdict"] is False
    assert out["followups"] == verifier.cfg.max_iterations

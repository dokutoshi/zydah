"""Verifier component tests: YAML config, validator, panel-path loop.

The panel-bypass (direct) path has its own home in tests/direct/;
user-question scenarios live in tests/levels/.
"""

import json

import pytest

from zydah.fusion import make_mass
from zydah.verifier import ZydahVerifier, load_verifier_config

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
    # The returned prompt IS the followup JSON that a responder would get.
    assert out["prompt"] == json.dumps(out["followup"], indent=2)
    assert out["followup"]["question"] == out["question"]
    assert "status" not in out["followup"]


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


def test_iteration_budget_exhausted_returns_false():
    def responder(prompt, model_id):
        return "Still ambiguous."

    panel = SequencePanel([wide_masses()])
    verifier = ZydahVerifier(FRAME, panel=panel, responder=responder)
    out = verifier.verify("Is it cloudy today in Ohio?")
    assert out["verdict"] is False
    assert out["followups"] == verifier.cfg.max_iterations

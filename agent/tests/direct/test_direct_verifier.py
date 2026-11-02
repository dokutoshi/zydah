"""Direct-path tests: verify_direct bypasses the judge panel.

One elicitor mass per call (transport + verifier only); the panel
contract is proven by an exploding panel that must never be called.
Component/panel verifier tests live in tests/suite/test_verifier.py;
user-question scenarios live in tests/levels/.
"""

import json

from zydah.fusion import make_mass
from zydah.verifier import ZydahVerifier, format_transcript

FRAME = frozenset({"pass", "partial", "fail"})
OMEGA = ("pass", "partial", "fail")


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


def test_followup_is_system1_response_to_llm_without_status():
    # Flagged => System 1 returns a structured follow-up for the LLM to
    # answer with a better response. No delivery status is attached; the
    # caller/responder loop is what answers it.
    verifier = ZydahVerifier(FRAME, mass_elicitor=SequenceElicitor([wide_mass()]))
    out = verifier.verify_direct(
        "claim", assumptions=["claim has no date qualifier."]
    )
    assert out["verdict"] is False
    assert out["followup"] == {
        "question": out["question"],
        "summary": "Clarify the blocking assumption before relying on this response.",
        "assumptions": ["claim has no date qualifier."],
    }
    assert "status" not in out["followup"]
    assert out["history"][0]["followup"] == out["followup"]


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
    # The transmitted message IS the followup JSON, not a verbose prompt.
    assert json.loads(seen["prompt"]) == out["history"][0]["followup"]
    assert seen["prompt"] == json.dumps(out["history"][0]["followup"], indent=2)


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
    assert out["prompt"] == json.dumps(out["followup"], indent=2)
    assert out["mode"] == "direct"

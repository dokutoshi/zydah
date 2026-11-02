"""Evidence boundary tests: provenance views + anti-eavesdropping rule."""

import json

import pytest

from zydah.evidence import (
    MODEL_FOLLOWUP,
    ORIGINAL,
    EvidenceChunk,
    EvidenceLedger,
    classify_independence,
)
from zydah.fusion import make_mass
from zydah.interface import capture_first_response, format_attributed_cycle, seal_cycle
from zydah.verifier import ZydahVerifier

FRAME = frozenset({"pass", "partial", "fail"})
OMEGA = ("pass", "partial", "fail")


def test_original_only_view_hides_followups():
    ledger = EvidenceLedger()
    ledger.add(EvidenceChunk(ORIGINAL, "claim text", session_id="s-a", playground="A"))
    ledger.add(EvidenceChunk(MODEL_FOLLOWUP, "follow-up secret", session_id="s-b", playground="B"))
    assert ledger.original_only() == "claim text"
    assert "follow-up secret" not in ledger.original_only()
    assert "follow-up secret" in ledger.assisted()


def test_session_rule_joint_when_same_session():
    out = classify_independence("s-1", "s-1", followups=0)
    assert out["mode"] == "JOINT_CONTEXT"


def test_session_rule_independent_separate_sessions_no_followups():
    out = classify_independence("s-a", "s-b", followups=0)
    assert out["mode"] == "CONTEXT_INDEPENDENT"


def test_followups_downgrade_to_assisted():
    out = classify_independence("s-a", "s-b", followups=2)
    assert out["mode"] == "ASSISTED"


class SeqElicitor:
    def __init__(self, masses):
        self._masses = list(masses)
        self.seen = []

    def __call__(self, evidence, frame, model_id):
        self.seen.append(evidence)
        idx = min(len(self.seen) - 1, len(self._masses) - 1)
        return self._masses[idx]


def wide():
    return make_mass(FRAME, {"pass": 0.5, OMEGA: 0.5})


def narrow():
    return make_mass(FRAME, {"pass": 0.95, OMEGA: 0.05})


def test_iteration0_elicitor_sees_no_zydah_text():
    elicitor = SeqElicitor([narrow()])
    verifier = ZydahVerifier(FRAME, mass_elicitor=elicitor, session_id="s-b")
    out = verifier.verify_direct("plain claim")
    assert elicitor.seen[0] == "plain claim"
    assert "[zydah]" not in elicitor.seen[0]
    assert out["independence"]["mode"] in {"ASSISTED", "CONTEXT_INDEPENDENT", "JOINT_CONTEXT"}


def test_same_session_marks_joint_and_different_marks_independent():
    first = capture_first_response(
        text="model said pass", model_id="m", session_id="s-b", playground="A"
    )
    verifier = ZydahVerifier(
        FRAME, mass_elicitor=SeqElicitor([narrow()]), session_id="s-b"
    )
    out = verifier.verify_direct("claim", first_response=first)
    assert out["independence"]["mode"] == "JOINT_CONTEXT"

    first2 = capture_first_response(
        text="model said pass", model_id="m", session_id="s-a", playground="A"
    )
    verifier2 = ZydahVerifier(
        FRAME, mass_elicitor=SeqElicitor([narrow()]), session_id="s-b"
    )
    out2 = verifier2.verify_direct("claim", first_response=first2)
    assert out2["independence"]["mode"] == "CONTEXT_INDEPENDENT"


def test_followup_cycle_is_assisted_and_logged(tmp_path):
    log_file = tmp_path / "cycles.jsonl"

    def responder(prompt, model_id):
        return "more info"

    elicitor = SeqElicitor([wide(), narrow()])
    first = capture_first_response(
        text="first answer", model_id="m", session_id="s-a", playground="A"
    )
    verifier = ZydahVerifier(
        FRAME,
        responder=responder,
        mass_elicitor=elicitor,
        session_id="s-b",
        log_path=log_file,
    )
    out = verifier.verify_direct("claim", first_response=first)
    assert out["followups"] == 1
    assert out["independence"]["mode"] == "ASSISTED"
    lines = log_file.read_text().strip().splitlines()
    assert lines
    record = json.loads(lines[-1])
    assert record["verifier_session"] == "s-b"
    assert "s-a" in record["sessions"]
    assert "key" not in json.dumps(record).lower()


def test_sealed_output_shows_playground_and_session():
    first = capture_first_response(
        text="endpoint text", model_id="m1", session_id="s-a", playground="A"
    )
    verification = {
        "verdict": "good",
        "followups": 0,
        "model_id": "m1",
        "result": {
            "zydah": {
                "verdict": "pass",
                "decision": "commit",
                "bel": 0.9,
                "pl": 1.0,
                "width": 0.1,
                "conflict": 0.0,
                "rule": "yager",
            }
        },
    }
    text = format_attributed_cycle(seal_cycle(first, verification))
    assert "playground=A" in text.split("[zydah]")[0]
    assert "session=s-a" in text.split("[zydah]")[0]

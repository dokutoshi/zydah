"""Level tests: Simple questions vs Context-Independence comparisons.

- Simple: everyday questions (weather in a state; ambiguous city with no
  state). Expected: direct path either answers cleanly or asks for the
  missing detail instead of fabricating certainty.
- Context-Independence: comparative claims (car A vs car B) verified
  through the evidence boundary. Expected: separate sessions earn
  CONTEXT_INDEPENDENT; same session is JOINT_CONTEXT.
"""

import json
import os
import random

import pytest

from zydah.fusion import belief_frame, make_mass
from zydah.graph import build_place_graph
from zydah.graph_resolver import NumpyResolver
from zydah.interface import capture_first_response, direct_verdict
from zydah.transport import load_transport_config
from zydah.verifier import (
    ZydahVerifier,
    make_mass_elicitor,
    make_model_responder,
    serialize_followup,
)

FRAME = frozenset({"pass", "partial", "fail"})
OMEGA = ("pass", "partial", "fail")

STATES = ["Ohio", "Texas", "Oregon", "Maine", "Nevada", "Georgia", "Alaska", "Kansas"]


def _live_enabled() -> bool:
    """Live tests run only on explicit opt-in with endpoint env present."""
    if not os.getenv("ZYDAH_LIVE_TESTS"):
        return False
    cfg = load_transport_config()
    env_names = (str(cfg.base_url_env), str(cfg.api_key_env), str(cfg.model_env))
    return all(os.getenv(name) for name in env_names)


class FixedElicitor:
    """One fixed mass; records the evidence it was shown."""

    def __init__(self, mass):
        self._mass = mass
        self.seen = []

    def __call__(self, evidence, frame, model_id):
        self.seen.append(evidence)
        return self._mass


def confident_mass():
    return make_mass(FRAME, {"pass": 0.92, OMEGA: 0.08})


def unsure_mass():
    return make_mass(FRAME, {OMEGA: 1.0})


# --- Level: Simple ---

class SequenceElicitor:
    """Returns the next single mass per call; records evidence seen."""

    def __init__(self, masses):
        self._masses = list(masses)
        self.calls = 0
        self.seen = []

    def __call__(self, evidence, frame, model_id):
        self.seen.append(evidence)
        idx = min(self.calls, len(self._masses) - 1)
        self.calls += 1
        return self._masses[idx]


def cycle_record(out, *, question, response_text):
    """Artifact schema: identify the exchange, not just the judgment.

    ``response_text`` is the textual candidate answer being judged.
    Evaluator outputs (direct/fused judgments) live under each turn's
    ``evaluation`` block; they are not the textual response. A turn's
    ``followup`` is System 1's structured response to the LLM (question,
    summary, blocking assumptions from the verifier, not test-local
    fabrication), paired with ``followup_answer``: the LLM's better
    response to that follow-up (also the next turn's ``response``), or
    ``None`` when no responder answered it. No delivery ``status`` is
    recorded. The followup JSON itself is the transmitted message; no
    separate verbose prompt is dumped.
    """
    turns = []
    history = out["history"]
    for idx, entry in enumerate(history, start=1):
        prev = history[idx - 2] if idx > 1 else None
        prev_validation = prev.get("validation", {}) if prev else {}
        validation = entry["validation"]
        zydah = entry["zydah"]
        turn = {
            "turn": idx,
            "question": question if idx == 1 else prev_validation.get("question"),
            "response": response_text if idx == 1 else (prev.get("response") if prev else None),
            "belief_frame": belief_frame(entry["mass"], FRAME),
            "evaluation": {
                "direct": direct_verdict([entry["mass"]], FRAME),
                "zydah": zydah,
                "validator": validation.get("verdict"),
            },
        }
        if entry.get("followup"):
            turn["followup"] = {
                **entry["followup"],
                "followup_answer": entry.get("response"),
            }
        turns.append(turn)
    return {
        "question": question,
        "response": response_text,
        "turns": turns,
        "validator": out["verdict"],
        "followups": out["followups"],
        "independence": out.get("independence", {}).get("mode"),
    }


def test_simple_weather_random_states_answered_directly(output_dir):
    rng = random.Random(11)
    cases = []
    for state in rng.sample(STATES, k=4):
        elicitor = FixedElicitor(confident_mass())
        verifier = ZydahVerifier(FRAME, mass_elicitor=elicitor, session_id="s-b")
        question = f"What is the weather in {state} today?"
        response_text = f"It is cloudy in {state} today."
        first = capture_first_response(
            text=response_text, model_id="m-a", session_id="session-a", playground="A"
        )
        out = verifier.verify_direct(
            f"Claim to evaluate: {response_text}", first_response=first
        )
        assert out["verdict"] == "good"
        assert out["followups"] == 0
        assert elicitor.seen and state in question  # fixture sanity
        record = cycle_record(out, question=question, response_text=response_text)
        record["state"] = state
        cases.append(record)
    (output_dir / "weather_states.json").write_text(
        json.dumps(cases, indent=2) + "\n"
    )


def test_simple_springfield_without_state_asks_for_detail(output_dir):
    # Ambiguous city, no state: System 1's response to the LLM is the
    # structured follow-up (which Springfield?), not a confident verdict.
    elicitor = FixedElicitor(unsure_mass())
    verifier = ZydahVerifier(FRAME, mass_elicitor=elicitor, session_id="s-b")
    question = "What is the weather in Springfield today?"
    response_text = "It is cloudy in Springfield today."
    first = capture_first_response(
        text=response_text, model_id="m-a", session_id="session-a", playground="A"
    )
    out = verifier.verify_direct(
        f"Claim to evaluate: {response_text}",
        first_response=first,
        assumptions=["Springfield has no state qualifier."],
    )
    assert out["verdict"] is False
    assert out["followup"] == {
        "question": out["question"],
        "summary": "Clarify the blocking assumption before relying on this response.",
        "assumptions": ["Springfield has no state qualifier."],
    }
    assert "status" not in out["followup"]
    record = cycle_record(out, question=question, response_text=response_text)
    assert record["turns"][0]["followup"] == {
        **out["followup"],
        "followup_answer": None,  # no responder wired in this test
    }
    assert "next_prompt" not in json.dumps(record)
    record["followup_question"] = out["question"]
    (output_dir / "springfield.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )


def test_simple_springfield_followup_returns_better_response(output_dir):
    # The LLM receives System 1's follow-up JSON itself and answers it
    # with a better (disambiguated) response; System 1 re-judges it.
    def responder(prompt, model_id):
        payload = json.loads(prompt)  # transmitted message is the JSON
        assert payload["assumptions"] == ["Springfield has no state qualifier."]
        return "Springfield, Illinois: it is cloudy today."

    elicitor = SequenceElicitor([unsure_mass(), confident_mass()])
    verifier = ZydahVerifier(
        FRAME, mass_elicitor=elicitor, responder=responder, session_id="s-b"
    )
    question = "What is the weather in Springfield today?"
    response_text = "It is cloudy in Springfield today."
    first = capture_first_response(
        text=response_text, model_id="m-a", session_id="session-a", playground="A"
    )
    out = verifier.verify_direct(
        f"Claim to evaluate: {response_text}",
        first_response=first,
        assumptions=["Springfield has no state qualifier."],
    )
    assert out["verdict"] == "good"
    assert out["followups"] == 1
    assert out["independence"]["mode"] == "ASSISTED"  # follow-up was used
    record = cycle_record(out, question=question, response_text=response_text)
    assert record["turns"][0]["followup"]["assumptions"] == [
        "Springfield has no state qualifier."
    ]
    assert "status" not in record["turns"][0]["followup"]
    assert record["turns"][0]["followup"]["followup_answer"] == (
        "Springfield, Illinois: it is cloudy today."
    )
    assert record["turns"][1]["response"] == "Springfield, Illinois: it is cloudy today."
    (output_dir / "springfield_clarified.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )


def live_cycle_record(out, *, question, response_text, cfg):
    """Artifact for the live run: sent JSON and answers, never merged.

    A flagged turn records the exact follow-up payload under ``sent``
    plus ``answer_turn`` pointing at the turn that answered it; an answer
    turn carries ``in_response_to_turn`` and the model's verbatim text.
    No turn invents a question nobody asked.
    """
    turns = []
    history = out["history"]
    for idx, entry in enumerate(history, start=1):
        prev = history[idx - 2] if idx > 1 else None
        validation = entry["validation"]
        zydah = entry["zydah"]
        turn = {
            "turn": idx,
            "belief_frame": belief_frame(entry["mass"], FRAME),
            "width": zydah.get("width"),
            "evaluation": {
                "direct": direct_verdict([entry["mass"]], FRAME),
                "zydah": zydah,
                "validator": validation.get("verdict"),
            },
        }
        if idx == 1:
            turn["question"] = question
            turn["response"] = response_text
        else:
            turn["in_response_to_turn"] = idx - 1
            turn["response"] = prev.get("response") if prev else None
        if entry.get("followup"):
            turn["followup"] = {
                "sent": entry["followup"],
                "answer_turn": idx + 1 if entry.get("response") else None,
            }
        turns.append(turn)
    return {
        "question": question,
        "response": response_text,
        "turns": turns,
        "validator": out["verdict"],
        "followups": out["followups"],
        "model_id": out["model_id"],
        "independence": out.get("independence", {}).get("mode"),
        "config": {
            "max_iterations": int(cfg.max_iterations),
            "width_threshold": float(cfg.width_threshold),
            "conflict_threshold": float(cfg.conflict_threshold),
        },
    }


@pytest.mark.skipif(
    not _live_enabled(),
    reason="live follow-up disabled: set ZYDAH_LIVE_TESTS=1 and MODEL_API_* env",
)
def test_simple_springfield_followup_live_model_answer(output_dir):
    # Agent-default live loop: live elicitor + live responder, Hydra
    # config untouched. The recording wrapper observes the wire only.
    sent = []
    live_responder = make_model_responder()

    def recording_responder(prompt, model_id):
        sent.append(prompt)
        return live_responder(prompt, model_id)

    verifier = ZydahVerifier(FRAME, responder=recording_responder, session_id="s-b")
    cfg = verifier.cfg
    question = "What is the weather in Springfield today?"
    response_text = "It is cloudy in Springfield today."
    first = capture_first_response(
        text=response_text, model_id="m-a", session_id="session-a", playground="A"
    )
    out = verifier.verify_direct(
        f"Claim to evaluate: {response_text}",
        first_response=first,
        assumptions=["Springfield has no state qualifier."],
    )
    # Wire fidelity: each live call carried that turn's follow-up JSON.
    # (A budget-exhausted final flag is recorded but never sent.)
    answered = [
        entry
        for entry in out["history"]
        if entry.get("followup") and entry.get("response")
    ]
    assert len(sent) == out["followups"] <= int(cfg.max_iterations)
    assert len(sent) == len(answered)
    for entry, prompt in zip(answered, sent):
        assert json.loads(prompt) == entry["followup"]
        assert prompt == serialize_followup(entry["followup"])
    # Termination consistency with the Hydra-defined values.
    widths = [entry["zydah"]["width"] for entry in out["history"]]
    if out["verdict"] == "good":
        assert widths[-1] <= float(cfg.width_threshold)
    else:
        assert out["verdict"] is False
        assert out["followups"] == int(cfg.max_iterations)
        assert widths[-1] > float(cfg.width_threshold)
    record = live_cycle_record(
        out, question=question, response_text=response_text, cfg=cfg
    )
    (output_dir / "springfield_live.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )


@pytest.mark.skipif(
    not _live_enabled(),
    reason="live RLHF disabled: set ZYDAH_LIVE_TESTS=1 and MODEL_API_* env",
)
def test_live_human_choice_followup_to_model(output_dir):
    # RLHF path, live after the pause: the pause itself is produced
    # offline (unresolved Springfield graph), then the human choice
    # B -> North is sent to the LIVE model as a HumanFollowup JSON,
    # the model's revised response is appended, and re-elicitation is
    # the LIVE default elicitor. The human keypress is the only
    # programmatic input; everything downstream is a real model call.
    sent = []
    live_responder = make_model_responder()

    def setup_responder(prompt, model_id):
        return "No additional context is available."

    verifier = ZydahVerifier(
        FRAME,
        responder=setup_responder,
        mass_elicitor=SequenceElicitor([unsure_mass()]),
        graph_resolver=NumpyResolver(),
        graph_factory=lambda evidence: build_place_graph(),
        session_id="s-b",
    )
    question = "What is the weather in Springfield today?"
    response_text = "It is cloudy in Springfield today."
    first = capture_first_response(
        text=response_text, model_id="m-a", session_id="session-a", playground="A"
    )
    paused = verifier.verify_direct(
        f"Claim to evaluate: {response_text}",
        first_response=first,
        assumptions=["Springfield has no state qualifier."],
    )
    assert paused["needs_human"] is True
    assert {"key": "B", "value": "North"} in paused["choices"]

    def recording_live_responder(prompt, model_id):
        sent.append(prompt)
        return live_responder(prompt, model_id)

    verifier.responder = recording_live_responder
    verifier.mass_elicitor = make_mass_elicitor(
        str(verifier.cfg.direct_mass_prompt_template)
    )
    resumed = verifier.resume_direct(paused, "B")

    # The transmitted follow-up is the human-choice JSON + graph context.
    assert sent, "live responder was never called"
    sent_payload = json.loads(sent[0])
    assert sent_payload["human_choice"] == {"key": "B", "value": "North"}
    assert sent_payload["context"]["partition"] == "North"
    assert sent_payload["context"]["candidates"] == [
        "Springfield, IL",
        "Springfield, MA",
        "Springfield, OH",
        "Springfield, OR",
    ]
    assert "Springfield" in sent_payload["context"]["question"]
    assert resumed["human_followup"] == sent_payload
    assert resumed["history"][0]["human_feedback"] == resumed["human_followup"]
    model_response = resumed["history"][0].get("response")
    assert model_response and str(model_response).strip()
    model_answers = [
        chunk.text
        for chunk in resumed["ledger"].chunks
        if chunk.kind == "model_followup"
    ]
    if resumed.get("needs_human"):
        # Faithful alternative: the model used the North context and
        # asked the next narrowing question instead of answering.
        assert "Springfield" in str(model_response)
    else:
        assert model_answers and all(str(t).strip() for t in model_answers)
        assert resumed["followups"] >= 1
    assert resumed["followups"] <= int(verifier.cfg.max_iterations)
    if resumed["verdict"] == "good":
        assert resumed["history"][-1]["zydah"]["width"] <= float(
            verifier.cfg.width_threshold
        )
    record = {
        "question": question,
        "response": response_text,
        "choices": paused["choices"],
        "human_followup_sent": sent[0],
        "model_response": model_response,
        "model_answers": model_answers,
        "human_feedback": resumed["human_followup"],
        "verdict": resumed["verdict"],
        "needs_human": resumed.get("needs_human", False),
        "followups": resumed["followups"],
        "independence": resumed.get("independence", {}).get("mode"),
        "turns": live_cycle_record(
            resumed, question=question, response_text=response_text, cfg=verifier.cfg
        )["turns"],
    }
    (output_dir / "springfield_rlhf.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )


@pytest.mark.skipif(
    not _live_enabled(),
    reason="live clue test disabled: set ZYDAH_LIVE_TESTS=1 and MODEL_API_* env",
)
def test_live_other_clue_through_zydah(output_dir):
    # The Celtics clue, live: pause is produced offline, then the
    # free-text Other answer goes through the LIVE model (anchored by
    # the directional frame) and re-elicitation is the LIVE elicitor.
    # See docs/Panel Log - Why Zydah vs MC Graph.md (2026-11-02).
    sent = []
    live_responder = make_model_responder()

    def setup_responder(prompt, model_id):
        return "No additional context is available."

    verifier = ZydahVerifier(
        FRAME,
        responder=setup_responder,
        mass_elicitor=SequenceElicitor([unsure_mass()]),
        graph_resolver=NumpyResolver(),
        graph_factory=lambda evidence: build_place_graph(),
        session_id="s-b",
    )
    question = "What is the weather in Springfield today?"
    response_text = "It is cloudy in Springfield today."
    first = capture_first_response(
        text=response_text, model_id="m-a", session_id="session-a", playground="A"
    )
    paused = verifier.verify_direct(
        f"Claim to evaluate: {response_text}",
        first_response=first,
        assumptions=["Springfield has no state qualifier."],
    )
    assert paused["needs_human"] is True

    def recording_live_responder(prompt, model_id):
        sent.append(prompt)
        return live_responder(prompt, model_id)

    verifier.responder = recording_live_responder
    verifier.mass_elicitor = make_mass_elicitor(
        str(verifier.cfg.direct_mass_prompt_template)
    )
    from zydah.similarity import SentenceTransformerScorer

    verifier.similarity_scorer = SentenceTransformerScorer(
        str(verifier.cfg.similarity_model)
    )
    resumed = verifier.resume_direct(
        paused, "E", other_text="the celtics are a good team"
    )
    assert sent, "live responder was never called"
    payload = json.loads(sent[0])
    assert payload["human_choice"] == {
        "key": "E",
        "value": "the celtics are a good team",
    }
    assert payload["context"]["offered_choices"][:4] == [
        "East", "North", "South", "West",
    ]
    model_response = resumed["history"][0].get("response")
    assert model_response and str(model_response).strip()
    record = {
        "question": question,
        "response": response_text,
        "free_text": "the celtics are a good team",
        "human_followup_sent": payload,
        "model_response": model_response,
        "graph_conditioning": resumed.get("graph_conditioning"),
        "verdict": resumed["verdict"],
        "needs_human": resumed.get("needs_human", False),
        "followups": resumed["followups"],
        "independence": resumed.get("independence", {}).get("mode"),
        "turns": live_cycle_record(
            resumed, question=question, response_text=response_text, cfg=verifier.cfg
        )["turns"],
    }
    (output_dir / "celtics_clue.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )


# --- Level: Context-Independence ---

@pytest.mark.parametrize(
    "car_a,car_b",
    [("BMW M4", "Nissan GT-R"), ("Ford Mustang", "Dodge Challenger"), ("Tesla Model 3", "Toyota Camry")],
)
def test_comparison_separate_sessions_context_independent(car_a, car_b, output_dir):
    question = f"Which is faster, a {car_a} or a {car_b}?"
    claim = f"Claim to evaluate: A {car_a} is faster than a {car_b}."
    response_text = f"The {car_a} is faster."
    first = capture_first_response(
        text=response_text, model_id="m-a", session_id="session-a", playground="A"
    )
    verifier = ZydahVerifier(
        FRAME, mass_elicitor=FixedElicitor(confident_mass()), session_id="session-b"
    )
    out = verifier.verify_direct(claim, first_response=first)
    assert out["verdict"] == "good"
    assert out["independence"]["mode"] == "CONTEXT_INDEPENDENT"
    (output_dir / "comparison.json").write_text(
        json.dumps(
            cycle_record(out, question=question, response_text=response_text), indent=2
        )
        + "\n"
    )


def test_comparison_same_session_is_joint_context(output_dir):
    question = "Which is faster, a BMW M4 or a Nissan GT-R?"
    claim = "Claim to evaluate: A BMW M4 is faster than a Nissan GT-R."
    response_text = "The BMW M4 is faster."
    first = capture_first_response(
        text=response_text, model_id="m-a", session_id="shared", playground="A"
    )
    verifier = ZydahVerifier(
        FRAME, mass_elicitor=FixedElicitor(confident_mass()), session_id="shared"
    )
    out = verifier.verify_direct(claim, first_response=first)
    assert out["independence"]["mode"] == "JOINT_CONTEXT"
    (output_dir / "comparison.json").write_text(
        json.dumps(
            cycle_record(out, question=question, response_text=response_text), indent=2
        )
        + "\n"
    )

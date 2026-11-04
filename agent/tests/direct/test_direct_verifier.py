"""Direct-path tests: verify_direct bypasses the judge panel.

One elicitor mass per call (transport + verifier only); the panel
contract is proven by an exploding panel that must never be called.
Component/panel verifier tests live in tests/suite/test_verifier.py;
user-question scenarios live in tests/levels/.
"""

import json

import pytest

from zydah.evidence import GRAPH_RESOLUTION, MODEL_FOLLOWUP, ORIGINAL
from zydah.fusion import make_mass
from zydah.graph import build_place_graph
from zydah.graph_resolver import NumpyResolver
from zydah.interface import capture_first_response
from zydah.verifier import ZydahVerifier, format_needs_human, format_transcript

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


def test_human_question_pauses_loop_without_entering_ledger():
    def responder(prompt, model_id):
        return "Which Springfield do you mean — could you share the state?"

    verifier = ZydahVerifier(
        FRAME,
        responder=responder,
        mass_elicitor=SequenceElicitor([wide_mass()]),
        session_id="s-b",
    )
    first = capture_first_response(
        text="It is cloudy in Springfield today.",
        model_id="m-a",
        session_id="session-a",
        playground="A",
    )
    out = verifier.verify_direct("claim", first_response=first)
    assert out["verdict"] is False
    assert out["needs_human"] is True
    assert out["human_question"].startswith("Which Springfield")
    assert out["followups"] == 0  # question was not consumed as evidence
    assert [chunk.kind for chunk in out["ledger"].chunks] == [ORIGINAL]
    assert out["independence"]["mode"] == "CONTEXT_INDEPENDENT"
    rendered = format_needs_human(out)
    assert rendered.startswith("[human]\nWhich Springfield")
    assert "[zydah]" in rendered


def test_agent_answer_routes_through_graph_resolver():
    def responder(prompt, model_id):
        return "Timezone context is available for the candidates."

    verifier = ZydahVerifier(
        FRAME,
        responder=responder,
        mass_elicitor=SequenceElicitor([wide_mass(), narrow_mass()]),
        graph_resolver=NumpyResolver(),
        graph_factory=lambda evidence: build_place_graph(
            context={"timezone": "America/Chicago"}
        ),
    )
    out = verifier.verify_direct("claim")
    assert out["verdict"] == "good"
    assert out["followups"] == 1
    kinds = [chunk.kind for chunk in out["ledger"].chunks]
    assert MODEL_FOLLOWUP in kinds and GRAPH_RESOLUTION in kinds
    assert "Graph resolution" in out["ledger"].assisted()


def test_unresolved_graph_generates_partition_question_for_human():
    def responder(prompt, model_id):
        return "No additional context is available."

    verifier = ZydahVerifier(
        FRAME,
        responder=responder,
        mass_elicitor=SequenceElicitor([wide_mass()]),
        graph_resolver=NumpyResolver(),
        graph_factory=lambda evidence: build_place_graph(),
    )
    out = verifier.verify_direct("claim")
    assert out["needs_human"] is True
    for choice in ("A. East", "B. North", "C. South", "D. West", "E. Other"):
        assert choice in out["human_question"]
    assert MODEL_FOLLOWUP not in [c.kind for c in out["ledger"].chunks]


class FakeSimilarityScorer:
    """Deterministic encoder stand-in: one fixed score per call."""

    def __init__(self, value):
        self.value = value
        self.calls = []

    def similarities(self, text, candidates):
        self.calls.append((text, list(candidates)))
        return [self.value for _ in candidates]


def _paused_springfield(sent, human_answer=None, similarity_scorer=None):
    """Drive verify_direct to an agent_unresolved pause; record prompts."""

    def responder(prompt, model_id):
        sent.append(prompt)
        if '"human_choice"' in prompt:
            return human_answer or "Springfield, Illinois: it is cloudy today."
        return "No additional context is available."

    verifier = ZydahVerifier(
        FRAME,
        responder=responder,
        mass_elicitor=SequenceElicitor([wide_mass(), narrow_mass()]),
        graph_resolver=NumpyResolver(),
        graph_factory=lambda evidence: build_place_graph(),
        session_id="s-b",
        similarity_scorer=similarity_scorer,
    )
    first = capture_first_response(
        text="It is cloudy in Springfield today.",
        model_id="m-a",
        session_id="session-a",
        playground="A",
    )
    out = verifier.verify_direct("claim", first_response=first)
    assert out["needs_human"] is True
    return verifier, out


def test_choices_are_structured_enum_backed():
    _verifier, out = _paused_springfield([])
    assert out["choices"] == [
        {"key": "A", "value": "East"},
        {"key": "B", "value": "North"},
        {"key": "C", "value": "South"},
        {"key": "D", "value": "West"},
        {"key": "E", "value": "Other"},
    ]


def test_choice_b_sends_north_to_model_and_reelicits():
    sent = []
    verifier, out = _paused_springfield(sent)
    resumed = verifier.resume_direct(out, "B")
    human_prompt = sent[-1]
    # The follow-up carries the expanded graph context, not just "North".
    expected = {
        "human_choice": {"key": "B", "value": "North"},
        "context": {
            "partition": "North",
            "candidates": [
                "Springfield, IL",
                "Springfield, MA",
                "Springfield, OH",
                "Springfield, OR",
            ],
            # (A, ~A, Omega) per candidate: mass sits on the 4-member
            # focal set, so no singleton is committed either way.
            "candidate_beliefs": {
                "Springfield, IL": {"A": 0.0, "not_A": 0.0, "omega": 1.0},
                "Springfield, MA": {"A": 0.0, "not_A": 0.0, "omega": 1.0},
                "Springfield, OH": {"A": 0.0, "not_A": 0.0, "omega": 1.0},
                "Springfield, OR": {"A": 0.0, "not_A": 0.0, "omega": 1.0},
            },
            "offered_choices": [],
            "relationship_questions": {},
            "question": "claim",
        },
    }
    assert json.loads(human_prompt) == expected
    assert resumed["verdict"] == "good"
    assert resumed["needs_human"] is False
    assert resumed["followups"] == 1
    assert resumed["human_followup"] == expected
    # RLHF signal is recorded on the paused turn.
    assert resumed["history"][0]["human_feedback"] == resumed["human_followup"]
    answers = [
        chunk.text
        for chunk in resumed["ledger"].chunks
        if chunk.kind == MODEL_FOLLOWUP
    ]
    assert answers == ["Springfield, Illinois: it is cloudy today."]


def test_model_question_continues_to_candidate_choices(output_dir):
    sent = []
    answers = []

    def responder(prompt, model_id):
        sent.append(prompt)
        if '"human_choice"' in prompt:
            payload = json.loads(prompt)
            if payload["human_choice"]["value"] == "North":
                answer = "Which Springfield did you mean?"
            else:
                answer = "Springfield, Massachusetts: it is cloudy today."
        else:
            answer = "No additional context is available."
        answers.append(answer)
        return answer

    verifier = ZydahVerifier(
        FRAME,
        responder=responder,
        mass_elicitor=SequenceElicitor([wide_mass(), narrow_mass()]),
        graph_resolver=NumpyResolver(),
        graph_factory=lambda evidence: build_place_graph(),
        session_id="s-b",
    )
    first = capture_first_response(
        text="It is cloudy in Springfield today.",
        model_id="m-a",
        session_id="session-a",
        playground="A",
    )
    paused = verifier.verify_direct("claim", first_response=first)
    round1 = verifier.resume_direct(paused, "B")
    assert round1["needs_human"] is True
    assert round1["choices"] == [
        {"key": "A", "value": "Springfield, IL"},
        {"key": "B", "value": "Springfield, MA"},
        {"key": "C", "value": "Springfield, OH"},
        {"key": "D", "value": "Springfield, OR"},
        {"key": "E", "value": "Other"},
    ]
    assert "B. Springfield, MA" in round1["human_question"]
    round2 = verifier.resume_direct(round1, "B")
    payload = json.loads(sent[-1])
    assert payload["human_choice"] == {"key": "B", "value": "Springfield, MA"}
    assert payload["context"]["candidates"] == ["Springfield, MA"]
    assert payload["context"]["candidate_beliefs"] == {
        "Springfield, MA": {"A": 1.0, "not_A": 0.0, "omega": 0.0}
    }
    assert round2["verdict"] == "good"
    artifact = {
        "evidence": "claim",
        "first_response": "It is cloudy in Springfield today.",
        "rounds": [
            {
                "round": 1,
                "sent": sent[0],
                "model_response": answers[0],
                "route": "agent_unresolved",
                "choices": paused["choices"],
                "human_question": paused["human_question"],
            },
            {
                "round": 2,
                "human_choice": {"key": "B", "value": "North"},
                "sent": sent[1],
                "model_response": answers[1],
                "route": "human",
                "next_choices": round1["choices"],
                "human_question": round1["human_question"],
            },
            {
                "round": 3,
                "human_choice": {"key": "B", "value": "Springfield, MA"},
                "sent": sent[2],
                "model_response": answers[2],
                "route": "agent",
                "verdict": round2["verdict"],
                "followups": round2["followups"],
            },
        ],
        "verdict": round2["verdict"],
        "state_path": ["8 candidates", "North subset (4)", "Springfield, MA (1)"],
    }
    (output_dir / "candidate_choices.json").write_text(
        json.dumps(artifact, indent=2) + "\n"
    )


def test_other_choice_sends_free_text_to_model():
    sent = []
    verifier, out = _paused_springfield(sent)
    resumed = verifier.resume_direct(out, "E", other_text="Central")
    payload = json.loads(sent[-1])
    assert payload["human_choice"] == {"key": "E", "value": "Central"}
    # Other answers carry the directional frame they must map onto.
    assert payload["context"]["partition"] == "Other"
    assert payload["context"]["offered_choices"] == [
        "East", "North", "South", "West", "Other",
    ]
    assert "Springfield, MA" in payload["context"]["candidates"]
    assert resumed["verdict"] == "good"


def test_other_clue_gets_directional_frame(output_dir):
    sent = []
    verifier, out = _paused_springfield(sent)
    resumed = verifier.resume_direct(
        out, "E", other_text="the celtics are a good team"
    )
    payload = json.loads(sent[-1])
    assert payload["human_choice"] == {
        "key": "E",
        "value": "the celtics are a good team",
    }
    # The model can map the clue onto the offered directions/candidates
    # (Celtics -> Boston, MA -> East / Springfield, MA).
    assert payload["context"]["offered_choices"][:4] == [
        "East", "North", "South", "West",
    ]
    assert "Springfield, MA" in payload["context"]["candidates"]
    # The model is asked how the clue relates to each candidate; the
    # LLM (not an entity table) supplies Celtics -> Boston, MA.
    relationship = payload["context"]["relationship_questions"][
        "Springfield, MA"
    ]
    assert "celtics" in relationship
    assert "Springfield, MA" in relationship
    artifact = {
        "free_text": "the celtics are a good team",
        "choices_offered": out["choices"],
        "human_followup_sent": payload,
        "model_answers": [
            chunk.text
            for chunk in resumed["ledger"].chunks
            if chunk.kind == MODEL_FOLLOWUP
        ],
        "verdict": resumed["verdict"],
        "followups": resumed["followups"],
    }
    (output_dir / "other_clue.json").write_text(
        json.dumps(artifact, indent=2) + "\n"
    )


def test_relationship_answer_conditions_massachusetts(output_dir):
    # The LLM answers the relationship question with world knowledge;
    # System 1 matches Massachusetts in that answer and Yager-conditions
    # the credal state, moving attribution weight to Springfield, MA.
    sent = []
    verifier, out = _paused_springfield(
        sent,
        human_answer=(
            "The Celtics are Boston's NBA team; Boston and Springfield "
            "are both in Massachusetts."
        ),
    )
    resumed = verifier.resume_direct(
        out, "E", other_text="the celtics are a good team"
    )
    assert resumed["verdict"] == "good"
    conditioning = resumed["graph_conditioning"]
    assert conditioning["support"] == ["Springfield, MA"]
    # The stored value is the belief frame itself: discounted mass on
    # the MA focal element, residual on Omega.
    frame_value = conditioning["belief_frame"]
    assert "Springfield, MA" in frame_value["frame"]
    assert frame_value["mass"] == pytest.approx(
        {"Springfield, MA": 0.75, "Omega": 0.25}
    )
    beliefs = conditioning["candidate_beliefs"]
    assert beliefs["Springfield, MA"] == {
        "A": 0.75,
        "not_A": 0.0,
        "omega": 0.25,
    }
    assert beliefs["Springfield, IL"] == {
        "A": 0.0,
        "not_A": 0.75,
        "omega": 0.25,
    }
    graph_chunks = [
        chunk.text
        for chunk in resumed["ledger"].chunks
        if chunk.kind == GRAPH_RESOLUTION
    ]
    assert graph_chunks and "springfield:MA" in graph_chunks[-1]
    artifact = {
        "free_text": "the celtics are a good team",
        "human_followup_sent": json.loads(sent[-1]),
        "model_response": resumed["history"][0]["response"],
        "graph_conditioning": conditioning,
        "verdict": resumed["verdict"],
    }
    (output_dir / "celtics_relationship.json").write_text(
        json.dumps(artifact, indent=2) + "\n"
    )


def test_live_style_relationship_answer_conditions_only_massachusetts():
    # Live-shaped answer: every candidate gets a section, most are
    # negated, MA is affirmed, and the answer still ends with a
    # question. Extraction must run per candidate, before any
    # human-question routing, and support must be MA alone.
    sent = []
    verifier, out = _paused_springfield(
        sent,
        human_answer=(
            "Springfield, IL: No city, state, or regional relationship. "
            "Illinois is Bulls country.\n"
            "Springfield, MO: No relationship. No NBA Celtics "
            "connection to Missouri.\n"
            "Springfield, MA: Yes - direct state/regional relationship. "
            "The Boston Celtics play in Boston, MA. Springfield, MA is "
            "in the same state as Boston.\n"
            "Springfield, OH: No direct relationship. Ohio's team is "
            "the Cleveland Cavaliers, a rival.\n"
            "Which Springfield did you mean?"
        ),
    )
    resumed = verifier.resume_direct(
        out, "E", other_text="the celtics are a good team"
    )
    assert resumed["verdict"] == "good"
    assert resumed["needs_human"] is False
    conditioning = resumed["graph_conditioning"]
    assert conditioning["support"] == ["Springfield, MA"]
    assert (
        conditioning["candidate_beliefs"]["Springfield, MA"]["A"]
        > conditioning["candidate_beliefs"]["Springfield, IL"]["A"]
    )


def test_numbered_question_sections_condition_only_massachusetts():
    # Second live shape: an opening enumeration, then one numbered
    # question section per candidate; only MA is affirmed.
    sent = []
    verifier, out = _paused_springfield(
        sent,
        human_answer=(
            "Candidates: Springfield, IL, Springfield, MO, "
            "Springfield, MA, Springfield, OH.\n"
            "**1. How does the clue relate to Springfield, IL?**\n"
            "No city, state, or regional relationship. None.\n"
            "**2. How does the clue relate to Springfield, MA?**\n"
            "Yes - direct state relationship. The Boston Celtics "
            "play in Boston, Massachusetts. Springfield, MA is also "
            "in Massachusetts.\n"
            "**3. How does the clue relate to Springfield, OH?**\n"
            "No city, state, or regional relationship. None.\n"
            "Which Springfield did you mean?"
        ),
    )
    resumed = verifier.resume_direct(
        out, "E", other_text="the celtics are a good team"
    )
    assert resumed["needs_human"] is False
    assert resumed["graph_conditioning"]["support"] == [
        "Springfield, MA"
    ]


def test_enumeration_of_candidates_is_not_relationship_support():
    sent = []
    verifier, out = _paused_springfield(
        sent,
        human_answer=(
            "There are 8 candidates, all with equal weight: "
            "Springfield, IL, MO, MA, OH, OR, TN, GA, VA, each 0.125. "
            "Which Springfield did you mean?"
        ),
    )
    resumed = verifier.resume_direct(
        out, "E", other_text="the celtics are a good team"
    )
    assert "graph_conditioning" not in resumed
    assert resumed["needs_human"] is True


def test_question_answer_with_unresolved_graph_offers_choices():
    # A question-shaped model answer must face the graph first: the
    # unresolved graph generates the [human] MC question (with
    # choices), and the pause is resumable via resume_direct.
    def responder(prompt, model_id):
        if '"human_choice"' in prompt:
            return "Springfield, Illinois: it is cloudy today."
        return "Which Springfield did you mean?"

    verifier = ZydahVerifier(
        FRAME,
        responder=responder,
        mass_elicitor=SequenceElicitor([wide_mass(), narrow_mass()]),
        graph_resolver=NumpyResolver(),
        graph_factory=lambda evidence: build_place_graph(),
        session_id="s-b",
    )
    first = capture_first_response(
        text="It is cloudy in Springfield today.",
        model_id="m-a",
        session_id="session-a",
        playground="A",
    )
    out = verifier.verify_direct("claim", first_response=first)
    assert out["needs_human"] is True
    assert out["history"][0]["route"] == "agent_unresolved"
    assert [c["value"] for c in out["choices"]] == [
        "East", "North", "South", "West", "Other",
    ]
    resumed = verifier.resume_direct(out, "B")
    assert resumed["verdict"] == "good"


def test_balanced_markers_affirmative_counterparts_condition():
    # New affirmative family: existence/connection phrasing, not just
    # the location verbs the original list had.
    sent = []
    verifier, out = _paused_springfield(
        sent,
        human_answer=(
            "Springfield, MA: It has a direct relationship - the "
            "Celtics are connected to Massachusetts, the home-state "
            "team for Springfield."
        ),
    )
    resumed = verifier.resume_direct(
        out, "E", other_text="the celtics are a good team"
    )
    assert resumed["graph_conditioning"]["support"] == [
        "Springfield, MA"
    ]


def test_balanced_markers_negative_counterparts_block():
    # New negative family: negated location verbs must block support
    # even when the state name is mentioned.
    sent = []
    verifier, out = _paused_springfield(
        sent,
        human_answer=(
            "Springfield, MA: The Celtics are not based in "
            "Massachusetts and have no tie to Springfield."
        ),
    )
    resumed = verifier.resume_direct(
        out, "E", other_text="the celtics are a good team"
    )
    assert "graph_conditioning" not in resumed


def test_encoder_cross_check_keeps_support_above_threshold():
    scorer = FakeSimilarityScorer(0.9)
    sent = []
    verifier, out = _paused_springfield(
        sent,
        human_answer=(
            "The Celtics are Boston's NBA team; Boston and Springfield "
            "are both in Massachusetts."
        ),
        similarity_scorer=scorer,
    )
    resumed = verifier.resume_direct(
        out, "E", other_text="the celtics are a good team"
    )
    conditioning = resumed["graph_conditioning"]
    assert conditioning["support"] == ["Springfield, MA"]
    assert conditioning["similarity_scores"] == {"Springfield, MA": 0.9}
    assert scorer.calls and "Massachusetts" in scorer.calls[0][1][0]


def test_encoder_cross_check_drops_support_below_threshold():
    scorer = FakeSimilarityScorer(0.1)
    sent = []
    verifier, out = _paused_springfield(
        sent,
        human_answer=(
            "The Celtics are Boston's NBA team; Boston and Springfield "
            "are both in Massachusetts."
        ),
        similarity_scorer=scorer,
    )
    resumed = verifier.resume_direct(
        out, "E", other_text="the celtics are a good team"
    )
    assert "graph_conditioning" not in resumed


def test_candidate_choices_cover_eight_candidates_plus_other():
    candidates = [f"Springfield, {s}" for s in
                  ("IL", "MO", "MA", "OH", "OR", "TN", "GA", "VA")]
    choices = ZydahVerifier._candidate_choices(candidates)
    assert [c.value for c in choices] == candidates + ["Other"]
    assert choices[-1].key.value == "I"


def test_negative_relationship_answer_conditions_nothing():
    sent = []
    verifier, out = _paused_springfield(
        sent, human_answer="That clue is unrelated to any Springfield."
    )
    resumed = verifier.resume_direct(
        out, "E", other_text="the celtics are a good team"
    )
    assert "graph_conditioning" not in resumed


def test_invalid_and_textless_choices_fail_closed():
    import pytest as _pytest

    verifier, out = _paused_springfield([])
    with _pytest.raises(ValueError):
        verifier.resume_direct(out, "Z")
    with _pytest.raises(ValueError):
        verifier.resume_direct(out, "E")  # Other without free text


def test_verify_direct_no_responder_returns_prompt():
    verifier = ZydahVerifier(FRAME, mass_elicitor=SequenceElicitor([wide_mass()]))
    out = verifier.verify_direct("claim")
    assert out["verdict"] is False
    assert out["question"] in out["prompt"]
    assert out["prompt"] == json.dumps(out["followup"], indent=2)
    assert out["mode"] == "direct"


def _graph_verifier(sent, responder=None, similarity_scorer=None):
    def _responder(prompt, model_id):
        sent.append(prompt)
        return responder(prompt) if responder else ""

    return ZydahVerifier(
        FRAME,
        responder=_responder if responder is not None else None,
        mass_elicitor=SequenceElicitor([wide_mass()]),
        graph_resolver=NumpyResolver(),
        graph_factory=lambda evidence: build_place_graph(),
        similarity_scorer=similarity_scorer,
    )


def test_human_text_state_mention_conditions_without_model():
    sent = []
    verifier = _graph_verifier(sent)
    out = verifier.condition_on_human_text(
        "it is the Massachusetts one",
        question="What is the weather in Springfield today?",
    )
    assert out is not None
    assert out["support"] == ["Springfield, MA"]
    beliefs = out["candidate_beliefs"]
    assert beliefs["Springfield, MA"]["A"] > beliefs["Springfield, IL"]["A"]
    assert sent == []  # deterministic: no model call was needed


def test_human_text_entity_clue_uses_model_relationship_answer():
    sent = []

    def responder(prompt):
        return (
            "Springfield, IL: no relationship to the Celtics.\n"
            "Springfield, MA: the Celtics are Boston, Massachusetts; "
            "same state relationship.\n"
            "Springfield, MO: no relationship to the Celtics."
        )

    verifier = _graph_verifier(sent, responder=responder)
    out = verifier.condition_on_human_text(
        "the celtics are a good team",
        question="What is the weather in Springfield today?",
    )
    assert out is not None
    assert out["support"] == ["Springfield, MA"]
    # The clue entered as an Other HumanFollowup JSON with questions.
    followup = json.loads(sent[-1])
    assert followup["human_choice"]["value"] == "the celtics are a good team"
    assert followup["context"]["relationship_questions"]["Springfield, MA"]


def test_human_text_negation_veto_beats_similarity():
    sent = []
    verifier = _graph_verifier(
        sent, similarity_scorer=FakeSimilarityScorer(1.0)
    )
    out = verifier.condition_on_human_text(
        "It is not the Massachusetts one. It is the Illinois one.",
        question="What is the weather in Springfield today?",
    )
    assert out is not None
    assert out["support"] == ["Springfield, IL"]


def test_human_text_without_support_conditions_nothing():
    sent = []
    verifier = _graph_verifier(sent, responder=lambda prompt: "No relationship.")
    out = verifier.condition_on_human_text(
        "i am watching on tv",
        question="What is the weather in Springfield today?",
    )
    assert out is None


def test_human_text_without_resolver_conditions_nothing():
    verifier = ZydahVerifier(
        FRAME, mass_elicitor=SequenceElicitor([wide_mass()])
    )
    assert (
        verifier.condition_on_human_text("it is the Massachusetts one")
        is None
    )


def test_verify_direct_aggregates_transport_usage(monkeypatch):
    from zydah.transport import chat_complete, load_transport_config

    cfg = load_transport_config()
    monkeypatch.setenv(cfg.base_url_env, "https://api.example.test/v1")
    monkeypatch.setenv(cfg.api_key_env, "test-key")
    monkeypatch.setenv(cfg.model_env, "test-model")

    def opener(url, data, headers, timeout):
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        }

    def elicitor(evidence, frame, model_id):
        chat_complete("score this", opener=opener)
        return narrow_mass()

    verifier = ZydahVerifier(FRAME, mass_elicitor=elicitor)
    out = verifier.verify_direct("claim")
    assert out["usage"]["calls"] == 1
    assert out["usage"]["tokens_in"] == 11
    assert out["usage"]["tokens_out"] == 7
    assert out["usage"]["elapsed_s"] >= 0


def test_verify_direct_without_transport_calls_reports_zero_usage():
    verifier = ZydahVerifier(
        FRAME, mass_elicitor=SequenceElicitor([narrow_mass()])
    )
    out = verifier.verify_direct("claim")
    assert out["usage"]["calls"] == 0
    assert out["usage"]["tokens_in"] in (None, 0)

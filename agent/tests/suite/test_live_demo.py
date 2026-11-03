"""Interactive demo loop tests (fake verifier, no API calls)."""

import importlib.util
import json
import sys

SPEC = importlib.util.spec_from_file_location("live_demo", "tools/live_demo.py")
live_demo = importlib.util.module_from_spec(SPEC)
sys.modules["live_demo"] = live_demo
SPEC.loader.exec_module(live_demo)


class FakeVerifier:
    def __init__(self):
        self.calls = []

    def resume_direct(self, previous, key, other_text=None):
        self.calls.append((previous, key, other_text))
        return {
            "verdict": "good",
            "needs_human": False,
            "followups": 1,
            "history": [],
        }


def test_interact_prompts_for_choice_and_resumes():
    paused = {
        "verdict": False,
        "needs_human": True,
        "human_question": "Which of these applies?\nA. East\nB. North\nE. Other",
        "choices": [
            {"key": "A", "value": "East"},
            {"key": "B", "value": "North"},
            {"key": "E", "value": "Other"},
        ],
        "followups": 0,
        "history": [],
    }
    verifier = FakeVerifier()
    answers = iter(["b"])
    emitted = []

    final = live_demo.interact(
        paused,
        verifier,
        ask=lambda prompt: next(answers),
        emit=emitted.append,
    )

    assert verifier.calls == [(paused, "B", None)]
    assert final["verdict"] == "good"
    assert any("[human]" in line for line in emitted)


def test_render_result_leads_with_final_response_then_verdict():
    emitted = []
    result = {
        "verdict": "good",
        "needs_human": False,
        "followups": 1,
        "history": [],
        "graph_conditioning": {"support": ["Springfield, MA"]},
    }

    live_demo.render_result(result, emitted.append)

    assert emitted[0] == "[zydah] you are asking about Springfield, MA."
    assert emitted[1].startswith("verdict=good")


def test_save_tee_writes_partial_and_final_json(tmp_path):
    path = tmp_path / "latest.json"
    tee = live_demo.SaveTee(path)
    tee.begin("What is the weather in Springfield today?")
    tee.emit("Springfield, MA is cloudy today.")

    partial = json.loads(path.read_text())
    assert partial["question"] == "What is the weather in Springfield today?"
    assert partial["transcript"][-1] == "Springfield, MA is cloudy today."

    tee.finish(
        "What is the weather in Springfield today?",
        "Springfield, MA is cloudy today.",
        {"verdict": "good", "needs_human": False, "followups": 0, "history": []},
    )

    final = json.loads(path.read_text())
    assert final["response"] == "Springfield, MA is cloudy today."
    assert final["verdict"] == "good"
    assert final["transcript"][-1] == "Springfield, MA is cloudy today."


def test_model_question_prompts_human_then_continues(monkeypatch):
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "model": {"env_var": "LIVE_DEMO_TEST_MODEL", "id": "m-test"},
            "human_question_markers": ["which"],
        }
    )
    model_answers = iter(
        [
            "Which Springfield were you thinking of?",
            "Springfield, MA is cloudy today.",
        ]
    )
    prompts = []

    def fake_chat(prompt, model_id=None):
        prompts.append(prompt)
        return next(model_answers)

    class ClarifyingVerifier(FakeVerifier):
        def verify_direct(self, evidence, first_response=None, assumptions=None):
            self.evidence = evidence
            return {
                "verdict": "good",
                "needs_human": False,
                "followups": 0,
                "history": [],
            }

    verifier = ClarifyingVerifier()
    monkeypatch.setattr(live_demo, "chat_complete", fake_chat)
    monkeypatch.setattr(
        live_demo, "build_verifier", lambda q, r, m: verifier
    )
    emitted = []
    human_answers = iter(["the celtics are a great team"])

    final, _verifier, response = live_demo.run_question(
        "What is the weather in Springfield today?",
        cfg=cfg,
        ask=lambda prompt: next(human_answers),
        emit=emitted.append,
    )

    assert response == "Springfield, MA is cloudy today."
    assert "Human answer: the celtics are a great team" in verifier.evidence
    assert any("Springfield, MA is cloudy today." in line for line in emitted)
    assert final["verdict"] == "good"
    assert len(prompts) == 2


def test_interact_eof_leaves_loop_paused():
    paused = {
        "verdict": False,
        "needs_human": True,
        "human_question": "Which of these applies?\nB. North",
        "choices": [{"key": "B", "value": "North"}],
        "followups": 0,
        "history": [],
    }
    verifier = FakeVerifier()

    def ask(prompt):
        raise EOFError

    final = live_demo.interact(
        paused,
        verifier,
        ask=ask,
        emit=lambda line: None,
    )

    assert final is paused
    assert verifier.calls == []


def test_interact_other_requires_and_sends_free_text():
    paused = {
        "verdict": False,
        "needs_human": True,
        "human_question": "Which of these applies?\nE. Other",
        "choices": [{"key": "E", "value": "Other"}],
        "followups": 0,
        "history": [],
    }
    verifier = FakeVerifier()
    answers = iter(["E", "the celtics are a good team"])

    final = live_demo.interact(
        paused,
        verifier,
        ask=lambda prompt: next(answers),
        emit=lambda line: None,
    )

    assert verifier.calls == [
        (paused, "E", "the celtics are a good team")
    ]
    assert final["verdict"] == "good"

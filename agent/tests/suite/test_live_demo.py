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


def test_pause_without_choices_returns_without_resuming():
    paused = {
        "verdict": False,
        "needs_human": True,
        "human_question": "Which Springfield did you mean?",
        "choices": [],
        "followups": 0,
        "history": [],
    }
    verifier = FakeVerifier()
    emitted = []

    final = live_demo.interact(
        paused, verifier, ask=lambda prompt: "B", emit=emitted.append
    )

    assert final is paused
    assert verifier.calls == []
    assert any("[human]" in line for line in emitted)
    assert not any("no structured choices" in line for line in emitted)


def test_buffered_direction_is_sent_before_evaluation(monkeypatch):
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "model": {"env_var": "LIVE_DEMO_TEST_MODEL", "id": "m-test"},
            "human_question_markers": ["which"],
        }
    )

    def fake_chat(prompt, model_id=None):
        return iter(["It is cloudy in Springfield today."])

    class CapturingVerifier(FakeVerifier):
        def verify_direct(self, evidence, first_response=None, assumptions=None):
            self.evidence = evidence
            return {
                "verdict": "good",
                "needs_human": False,
                "followups": 0,
                "history": [],
            }

    verifier = CapturingVerifier()
    monkeypatch.setattr(live_demo, "stream_chat_complete", fake_chat)
    monkeypatch.setattr(live_demo, "_terminal_fd", lambda: None)
    monkeypatch.setattr(
        live_demo, "build_verifier", lambda q, r, m, **kw: verifier
    )
    directions = iter(["use Springfield, MA"])

    final, _verifier, _response = live_demo.run_question(
        "What is the weather in Springfield today?",
        cfg=cfg,
        ask=lambda prompt: next(directions),
        emit=lambda line: None,
    )

    assert "Human direction: use Springfield, MA" in verifier.evidence
    assert final["verdict"] == "good"


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


def test_model_question_prompts_human_then_continues(monkeypatch, capsys):
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
        return iter([next(model_answers)])

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
    monkeypatch.setattr(live_demo, "stream_chat_complete", fake_chat)
    monkeypatch.setattr(live_demo, "_terminal_fd", lambda: None)
    monkeypatch.setattr(
        live_demo, "build_verifier", lambda q, r, m, **kw: verifier
    )
    emitted = []
    # First the human's answer to the model's question, then an empty
    # buffer Enter that releases Zydah to evaluate.
    human_answers = iter(["the celtics are a great team", ""])

    final, _verifier, response = live_demo.run_question(
        "What is the weather in Springfield today?",
        cfg=cfg,
        ask=lambda prompt: next(human_answers),
        emit=emitted.append,
    )

    assert response == "Springfield, MA is cloudy today."
    assert "...Thinking" in emitted
    # Call-tree order: [zydah] prints after [muse], before [human].
    zydah_idx = next(
        i for i, line in enumerate(emitted) if line.startswith("[zydah] the model needs")
    )
    human_idx = next(
        i for i, line in enumerate(emitted) if line.startswith("[human]")
    )
    assert zydah_idx < human_idx
    assert "Human answer: the celtics are a great team" in verifier.evidence
    # Streamed lines go through the same emit channel as the JSON.
    assert any(
        "Springfield, MA is cloudy today." in line for line in emitted
    )
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


def test_after_verdict_saves_then_invites_followup(tmp_path):
    path = tmp_path / "latest.json"
    tee = live_demo.SaveTee(path)
    emitted = []
    result = {
        "verdict": "good",
        "needs_human": False,
        "followups": 0,
        "history": [],
    }

    live_demo.after_verdict(
        result,
        tee=tee,
        question="q",
        response="r",
        save_path=path,
        emit=emitted.append,
    )

    assert emitted[-2] == f"saved: {path}"
    assert emitted[-1] == live_demo.RESOLVED_INVITE
    assert "followup" in emitted[-1]
    assert json.loads(path.read_text())["verdict"] == "good"


def test_after_verdict_invites_without_save_when_no_tee():
    emitted = []
    result = {
        "verdict": "good",
        "needs_human": False,
        "followups": 0,
        "history": [],
    }

    live_demo.after_verdict(
        result,
        tee=None,
        question="q",
        response="r",
        save_path=None,
        emit=emitted.append,
    )

    assert emitted == [live_demo.RESOLVED_INVITE]


def test_after_verdict_pauses_without_save_or_invite(tmp_path):
    path = tmp_path / "latest.json"
    tee = live_demo.SaveTee(path)
    emitted = []
    paused = {
        "verdict": False,
        "needs_human": True,
        "human_question": "Which Springfield did you mean?",
        "followups": 0,
        "history": [],
    }

    live_demo.after_verdict(
        paused,
        tee=tee,
        question="q",
        response="r",
        save_path=path,
        emit=emitted.append,
    )

    assert emitted == [
        "[zydah] paused for human input; not saved as a finished run."
    ]
    assert not any("followup" in line for line in emitted)
    assert not path.exists()


def test_feed_output_translates_newlines_to_crlf():
    import io

    out = io.StringIO()
    box = live_demo.TerminalBox(0, out=out)
    box.feed_output("ab\ncd")
    assert out.getvalue() == "ab\r\ncd"


def test_feed_output_keeps_existing_crlf_intact():
    import io

    out = io.StringIO()
    box = live_demo.TerminalBox(0, out=out)
    box.feed_output("ab\r\ncd")
    assert out.getvalue() == "ab\r\ncd"


def test_feed_output_with_buffer_redraws_input_line():
    import io

    out = io.StringIO()
    box = live_demo.TerminalBox(0, out=out)
    box.buffer = "hi"
    box.feed_output("ab\ncd")
    assert out.getvalue() == "\r\x1b[Kab\r\ncd\n\r\x1b[K[you] hi"


def test_terminal_box_buffer_edit_and_send():
    import io
    import os

    read_fd, write_fd = os.pipe()
    out = io.StringIO()
    box = live_demo.TerminalBox(read_fd, out=out)
    os.write(write_fd, b"abc\x7fd\r")
    sent = None
    for _ in range(10):
        sent = box.poll_submit(0.5)
        if sent is not None:
            break
    os.close(read_fd)
    os.close(write_fd)
    assert sent == "abd"


def test_terminal_box_does_not_prefix_streamed_output():
    import io
    import os

    read_fd, write_fd = os.pipe()
    out = io.StringIO()
    box = live_demo.TerminalBox(read_fd, out=out)
    box.feed_output("Yes - so you nailed it!")
    os.close(read_fd)
    os.close(write_fd)
    rendered = out.getvalue()
    assert rendered.startswith("Yes - so you nailed it!")
    assert "[you]" not in rendered


def test_terminal_box_handles_iterm_bracketed_paste():
    import io
    import os

    read_fd, write_fd = os.pipe()
    out = io.StringIO()
    box = live_demo.TerminalBox(read_fd, out=out)
    os.write(write_fd, b"\x1b[200~hello world\x1b[201~\r")
    sent = None
    for _ in range(30):
        sent = box.poll_submit(0.5)
        if sent is not None:
            break
    os.close(read_fd)
    os.close(write_fd)
    assert sent == "hello world"


def test_error_log_appends_jsonl_record(tmp_path):
    path = tmp_path / "errors.log"
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        live_demo._log_error("live_call", exc, path=path)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["context"] == "live_call"
    assert record["error"] == "boom"
    assert "RuntimeError" in record["traceback"]


def test_final_line_follows_dominant_mass_not_validator_label():
    emitted = []
    result = {
        "verdict": "good",
        "needs_human": False,
        "followups": 0,
        "history": [
            {"zydah": {"pignistic": {"fail": 0.97, "partial": 0.01, "pass": 0.01}}}
        ],
    }
    live_demo.render_result(result, emitted.append)
    assert emitted[0] == "[zydah] the response does not check out."
    assert emitted[1].startswith("verdict=good")


def test_empty_stream_falls_back_to_plain_call(monkeypatch):
    monkeypatch.setattr(
        live_demo, "stream_chat_complete", lambda prompt, model_id=None: iter([])
    )
    monkeypatch.setattr(
        live_demo,
        "chat_complete",
        lambda prompt, model_id=None: "A cat is a feline.",
    )
    monkeypatch.setattr(live_demo, "_terminal_fd", lambda: None)
    monkeypatch.setattr(live_demo, "_log_error", lambda *a, **k: None)
    emitted = []

    text, pending = live_demo._stream_model_text(
        "a dog is a canine but a cat is a what?", "m-test", emitted.append
    )

    assert text == "A cat is a feline."
    assert pending is None
    assert emitted[-1] == "A cat is a feline."


def test_thinking_elicitor_uses_zydah_semaphore():
    emitted = []
    elicitor = live_demo._thinking_elicitor(
        lambda evidence, frame, model_id: {"mass": True}, emitted.append
    )
    out = elicitor("evidence", frozenset({"pass"}), "m-test")
    assert out == {"mass": True}
    assert emitted == ["[zydah] ...Thinking"]


def test_render_result_zydah_view_shows_internal_frames():
    from zydah.fusion import make_mass

    frame = frozenset({"pass", "partial", "fail"})
    result = {
        "verdict": "good",
        "needs_human": False,
        "followups": 0,
        "history": [
            {
                "zydah": {"pignistic": {"pass": 0.9, "fail": 0.05, "partial": 0.05}},
                "mass": make_mass(frame, {"pass": 0.9, frame: 0.1}),
            }
        ],
        "graph_conditioning": {"support": ["Springfield, MA"]},
    }
    default_lines = []
    live_demo.render_result(result, default_lines.append)
    assert len(default_lines) == 2
    assert not any("belief_frame" in line for line in default_lines)

    zydah_lines = []
    live_demo.render_result(result, zydah_lines.append, show_zydah=True)
    assert zydah_lines[0] == "[zydah] you are asking about Springfield, MA."
    assert any("turn 1 belief_frame:" in line for line in zydah_lines)
    assert any("graph_conditioning:" in line for line in zydah_lines)


def test_render_result_verbose_capture_logs_internal_view(caplog):
    import logging

    from zydah.fusion import make_mass

    frame = frozenset({"pass", "partial", "fail"})
    result = {
        "verdict": "good",
        "needs_human": False,
        "followups": 1,
        "history": [
            {
                "prompt": '{"question": "Need more info?"}',
                "response": "Springfield, MA is cloudy today.",
                "mass": make_mass(frame, {"pass": 0.9, frame: 0.1}),
            }
        ],
        "graph_conditioning": {"support": ["Springfield, MA"]},
    }
    emitted = []

    with caplog.at_level(logging.INFO, logger="zydah.live_demo"):
        live_demo.render_result(
            result,
            emitted.append,
            show_zydah=False,
            capture_zydah_view=True,
        )

    assert len(emitted) == 2
    assert not any("belief_frame" in line for line in emitted)
    assert "[Prompt 1]" in caplog.text
    assert "turn 1 belief_frame:" in caplog.text
    assert '"pass": 0.9' in caplog.text
    assert "graph_conditioning:" in caplog.text


def test_thinking_elicitor_verbose_capture_logs_semaphore(caplog):
    import logging

    elicitor = live_demo._thinking_elicitor(
        lambda evidence, frame, model_id: {"mass": True},
        lambda line: None,
        capture_zydah_view=True,
    )

    with caplog.at_level(logging.INFO, logger="zydah.live_demo"):
        out = elicitor("evidence", frozenset({"pass"}), "m-test")

    assert out == {"mass": True}
    assert "[zydah] ...Thinking" in caplog.text


def test_main_verbose_enables_capture_without_showing_zydah(monkeypatch):
    seen = []

    def fake_run_question(question, **kwargs):
        seen.append(kwargs)
        return (
            {"verdict": "good", "needs_human": False, "followups": 0, "history": []},
            object(),
            "resp",
        )

    def no_terminal(prompt):
        raise EOFError

    monkeypatch.setattr(live_demo, "run_question", fake_run_question)
    monkeypatch.setattr(live_demo, "terminal_input", no_terminal)

    assert live_demo.main(["--verbose", "--question", "hi"]) == 0
    assert seen[-1]["show_zydah"] is False
    assert seen[-1]["capture_zydah_view"] is True


def test_help_lists_zydah_flag(capsys):
    import pytest

    with pytest.raises(SystemExit) as excinfo:
        live_demo.main(["--help"])
    assert excinfo.value.code == 0
    assert "--zydah" in capsys.readouterr().out


def test_empty_clarification_answer_reaches_zydah(monkeypatch):
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "model": {"env_var": "LIVE_DEMO_TEST_MODEL", "id": "m-test"},
            "human_question_markers": ["which"],
        }
    )
    calls = []

    def fake_chat(prompt, model_id=None):
        calls.append(prompt)
        return iter(["Which event do you mean?"])

    class QuestionVerifier(FakeVerifier):
        def verify_direct(self, evidence, first_response=None, assumptions=None):
            self.evidence = evidence
            return {
                "verdict": "good",
                "needs_human": False,
                "followups": 0,
                "history": [],
            }

    verifier = QuestionVerifier()
    monkeypatch.setattr(live_demo, "stream_chat_complete", fake_chat)
    monkeypatch.setattr(live_demo, "_terminal_fd", lambda: None)
    monkeypatch.setattr(
        live_demo, "build_verifier", lambda q, r, m, **kw: verifier
    )
    # Empty clarification answer, then empty buffer at the gate.
    answers = iter(["", ""])

    final, _verifier, _response = live_demo.run_question(
        "who is winning tonight?",
        cfg=cfg,
        ask=lambda prompt: next(answers),
        emit=lambda line: None,
    )

    assert len(calls) == 1
    assert final["verdict"] == "good"


def test_verbose_logs_trace_question_routing(monkeypatch, caplog):
    """--verbose traces text: asks_human decision, answer, scored claim."""
    import logging

    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "model": {"env_var": "LIVE_DEMO_TEST_MODEL", "id": "m-test"},
            "human_question_markers": ["which"],
            "max_iterations": 5,
        }
    )
    intro = "Hi, I'm Muse Spark!\n\nWhat did you want to know about today?"
    answers = iter([intro, "A cat is a **feline**."])

    def fake_chat(prompt, model_id=None):
        return iter([next(answers)])

    class CapturingVerifier(FakeVerifier):
        def verify_direct(self, evidence, first_response=None, assumptions=None):
            self.evidence = evidence
            return {
                "verdict": "good",
                "needs_human": False,
                "followups": 0,
                "history": [],
            }

    verifier = CapturingVerifier()
    monkeypatch.setattr(live_demo, "stream_chat_complete", fake_chat)
    monkeypatch.setattr(live_demo, "_terminal_fd", lambda: None)
    monkeypatch.setattr(
        live_demo, "build_verifier", lambda q, r, m, **kw: verifier
    )
    typed = iter(["well a dog is a canine and a cat is a what?", ""])

    with caplog.at_level(logging.INFO, logger="zydah.live_demo"):
        live_demo.run_question(
            "what do you know today?",
            cfg=cfg,
            ask=lambda prompt: next(typed),
            emit=lambda line: None,
        )

    assert "asks_human=True" in caplog.text
    assert "human answer: 'well a dog is a canine" in caplog.text
    assert "gate direction: ''" in caplog.text
    assert "evidence scored:" in caplog.text
    assert "Claim to evaluate: A cat is a **feline**." in caplog.text


def test_verbose_flag_sets_info_level(monkeypatch):
    import logging

    monkeypatch.setattr(
        live_demo,
        "run_question",
        lambda q, **kw: (
            {"verdict": "good", "needs_human": False, "followups": 0, "history": []},
            object(),
            "resp",
        ),
    )

    def no_terminal(prompt):
        raise EOFError

    monkeypatch.setattr(live_demo, "terminal_input", no_terminal)
    assert live_demo.main(["--verbose", "--question", "hi"]) == 0
    assert logging.getLogger().level == logging.INFO


def test_terminal_input_prompt_goes_to_stdout(monkeypatch, capsys):
    """The [you] prompt is UI: stdout, never lost in a stderr redirect."""

    class TtyStdin:
        def isatty(self):
            return True

        def fileno(self):
            raise OSError("no fd in test")

        def readline(self):
            return "hello\n"

    monkeypatch.setattr(live_demo.sys, "stdin", TtyStdin())
    assert live_demo.terminal_input("[you] ") == "hello"
    assert capsys.readouterr().out == "[you] "


def test_log_file_captures_info_without_stderr_redirect(monkeypatch, tmp_path):
    log_path = tmp_path / "trace.log"
    monkeypatch.setattr(
        live_demo,
        "run_question",
        lambda q, **kw: (
            {"verdict": "good", "needs_human": False, "followups": 0, "history": []},
            object(),
            "resp",
        ),
    )

    def no_terminal(prompt):
        raise EOFError

    monkeypatch.setattr(live_demo, "terminal_input", no_terminal)
    assert live_demo.main(["--log-file", str(log_path), "--question", "hi"]) == 0
    text = log_path.read_text()
    assert "verbose logging enabled" in text
    assert "input question: 'hi'" in text


def test_excepthook_logs_and_echoes_one_line(caplog, capfd):
    import logging

    with caplog.at_level(logging.ERROR, logger="zydah.live_demo"):
        live_demo._excepthook(RuntimeError, RuntimeError("boom"), None)
    assert "uncaught" in caplog.text
    assert "[zydah] fatal: RuntimeError: boom (see log)" in capfd.readouterr().out


def test_question_ending_intro_routes_typed_line_to_model(monkeypatch):
    """Regression: an intro ending in '?' with no YAML marker match is
    still a question to the human. The typed follow-up must be sent to
    the model as a human answer, and the continuation is what Zydah
    scores -- not the intro."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "model": {"env_var": "LIVE_DEMO_TEST_MODEL", "id": "m-test"},
            "human_question_markers": ["which"],
            "max_iterations": 5,
        }
    )
    intro = (
        "Hi, I'm Muse Spark!\n\n"
        "What did you want to know about today?"
    )
    model_answers = iter([intro, "A cat is a **feline**."])
    prompts = []

    def fake_chat(prompt, model_id=None):
        prompts.append(prompt)
        return iter([next(model_answers)])

    class CapturingVerifier(FakeVerifier):
        def verify_direct(self, evidence, first_response=None, assumptions=None):
            self.evidence = evidence
            return {
                "verdict": "good",
                "needs_human": False,
                "followups": 0,
                "history": [],
            }

    verifier = CapturingVerifier()
    monkeypatch.setattr(live_demo, "stream_chat_complete", fake_chat)
    monkeypatch.setattr(live_demo, "_terminal_fd", lambda: None)
    monkeypatch.setattr(
        live_demo, "build_verifier", lambda q, r, m, **kw: verifier
    )
    # Typed follow-up answers the model's question; empty Enter at the
    # gate then releases Zydah to evaluate the continuation.
    human_answers = iter(["well a dog is a canine and a cat is a what?", ""])

    final, _verifier, response = live_demo.run_question(
        "what do you know today?",
        cfg=cfg,
        ask=lambda prompt: next(human_answers),
        emit=lambda line: None,
    )

    assert len(prompts) == 2  # second model call happened
    assert "well a dog is a canine and a cat is a what?" in prompts[1]
    assert response == "A cat is a **feline**."
    assert "Human answer: well a dog is a canine" in verifier.evidence
    assert "Claim to evaluate: A cat is a **feline**." in verifier.evidence
    assert final["verdict"] == "good"


def test_ends_with_question_marker_rules():
    ends = live_demo._ends_with_question
    assert ends("What did you want to know about today?")
    assert ends('He asked "why?"')
    assert ends("She asked 'why?'")
    assert ends("In code: `why?`")
    assert ends("(really?)")
    assert ends("First line.\n\nWhat now?")
    assert not ends("It is cloudy today.")
    assert not ends("What?!")
    assert not ends("A question? Then a statement.")
    assert not ends("")


def test_zydah_json_is_logged_and_parseable(monkeypatch, caplog):
    """Each verifier update logs one `zydah_json` INFO record that a
    consumer (e.g. Streamlit) can parse to redraw the belief frame."""
    import logging

    from omegaconf import OmegaConf

    from zydah.fusion import make_mass

    cfg = OmegaConf.create(
        {
            "model": {"env_var": "LIVE_DEMO_TEST_MODEL", "id": "m-test"},
            "human_question_markers": ["which"],
            "max_iterations": 5,
        }
    )
    frame = frozenset({"pass", "partial", "fail"})

    class JsonVerifier(FakeVerifier):
        def verify_direct(self, evidence, first_response=None, assumptions=None):
            return {
                "verdict": "good",
                "needs_human": False,
                "followups": 0,
                "history": [
                    {"mass": make_mass(frame, {"pass": 0.9, frame: 0.1})}
                ],
            }

    verifier = JsonVerifier()
    monkeypatch.setattr(
        live_demo, "stream_chat_complete", lambda p, model_id=None: iter(["Done."])
    )
    monkeypatch.setattr(live_demo, "_terminal_fd", lambda: None)
    monkeypatch.setattr(
        live_demo, "build_verifier", lambda q, r, m, **kw: verifier
    )

    with caplog.at_level(logging.INFO, logger="zydah.live_demo"):
        live_demo.run_question(
            "a dog is a canine and a cat is a what?",
            cfg=cfg,
            ask=lambda prompt: "",
            emit=lambda line: None,
        )

    lines = [r for r in caplog.text.splitlines() if "zydah_json " in r]
    assert len(lines) == 1
    payload = json.loads(lines[0].split("zydah_json ", 1)[1])
    assert payload["verdict"] == "good"
    assert payload["question"] == "a dog is a canine and a cat is a what?"
    assert payload["turns"][0]["belief_frame"]["mass"]["pass"] == 0.9
    assert payload["belief_frame"]["mass"]["pass"] == 0.9


def test_gate_question_is_sent_to_model_not_scored_as_direction(monkeypatch):
    """Regression: at the gate, buffered text ending in '?' is a new
    question for the model -- not a Human direction folded into the
    old claim. The follow-up's answer is what Zydah scores."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "model": {"env_var": "LIVE_DEMO_TEST_MODEL", "id": "m-test"},
            "human_question_markers": ["which"],
            "max_iterations": 5,
        }
    )
    first = (
        "You're thinking of 2 different guys!\n\n"
        "The frog prince is Prince Naveen. "
        "The movie star that sang in the rain is Gene Kelly."
    )
    answers = iter([first, "Yes: Prince, the musician, sang Purple Rain."])
    prompts = []

    def fake_chat(prompt, model_id=None):
        prompts.append(prompt)
        return iter([next(answers)])

    class CapturingVerifier(FakeVerifier):
        def verify_direct(self, evidence, first_response=None, assumptions=None):
            self.evidence = evidence
            return {
                "verdict": "good",
                "needs_human": False,
                "followups": 0,
                "history": [],
            }

    verifier = CapturingVerifier()
    monkeypatch.setattr(live_demo, "stream_chat_complete", fake_chat)
    monkeypatch.setattr(live_demo, "_terminal_fd", lambda: None)
    monkeypatch.setattr(
        live_demo, "build_verifier", lambda q, r, m, **kw: verifier
    )
    # Gate: the new question, then empty Enter releases evaluation.
    typed = iter(["and is there one person that was both?", ""])

    final, _verifier, response = live_demo.run_question(
        "there is a frog that became a prince in a movie and movie star "
        "that sung in the rain - who is this?",
        cfg=cfg,
        ask=lambda prompt: next(typed),
        emit=lambda line: None,
    )

    assert len(prompts) == 2
    assert "and is there one person that was both?" in prompts[1]
    assert response == "Yes: Prince, the musician, sang Purple Rain."
    assert "Human question: and is there one person that was both?" in verifier.evidence
    assert "Human direction:" not in verifier.evidence
    assert "Claim to evaluate: Yes: Prince" in verifier.evidence
    assert final["verdict"] == "good"


def test_gate_q_marks_quit_so_main_can_say_bye(monkeypatch):
    """'q' at the gate is a quit, not a silent pause: it is flagged
    so main exits with [zydah] bye, keeping the checkpoint that was
    scored when the response landed."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "model": {"env_var": "LIVE_DEMO_TEST_MODEL", "id": "m-test"},
            "human_question_markers": ["which"],
            "max_iterations": 5,
        }
    )

    class CapturingVerifier(FakeVerifier):
        def __init__(self):
            self.calls = 0

        def verify_direct(self, evidence, first_response=None, assumptions=None):
            self.calls += 1
            return {
                "verdict": "good",
                "needs_human": False,
                "followups": 0,
                "history": [],
            }

    verifier = CapturingVerifier()
    monkeypatch.setattr(
        live_demo,
        "stream_chat_complete",
        lambda p, model_id=None: iter(["Prince sang in the rain."]),
    )
    monkeypatch.setattr(live_demo, "_terminal_fd", lambda: None)
    monkeypatch.setattr(
        live_demo, "build_verifier", lambda q, r, m, **kw: verifier
    )

    result, _verifier, _response = live_demo.run_question(
        "who sang in the rain?",
        cfg=cfg,
        ask=lambda prompt: "q",
        emit=lambda line: None,
    )

    assert verifier.calls == 1
    assert result["verdict"] == "good"
    assert result["quit"] is True


def test_main_says_bye_and_exits_on_gate_quit(monkeypatch, capsys):
    monkeypatch.setattr(
        live_demo,
        "run_question",
        lambda q, **kw: (
            {
                "verdict": False,
                "needs_human": True,
                "quit": True,
                "followups": 0,
                "history": [],
            },
            object(),
            "resp",
        ),
    )
    assert live_demo.main(["--question", "who sang in the rain?"]) == 0
    out = capsys.readouterr().out
    assert "[zydah] bye" in out
    assert "paused for human input" not in out


def test_human_clue_conditions_graph_before_scoring(monkeypatch):
    """A free-text clarification answer runs the relationship
    resolver (HumanFollowup machinery) before Zydah scores: the
    conditioning lands in the evidence and on the result."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "model": {"env_var": "LIVE_DEMO_TEST_MODEL", "id": "m-test"},
            "human_question_markers": ["which"],
            "max_iterations": 5,
        }
    )
    answers = iter(
        [
            "Which Springfield do you mean?",
            "It is cloudy in Springfield, MA today.",
        ]
    )

    def fake_chat(prompt, model_id=None):
        return iter([next(answers)])

    class ConditioningVerifier(FakeVerifier):
        def __init__(self):
            super().__init__()
            self.clues = []

        def condition_on_human_text(self, text, question=""):
            self.clues.append(text)
            return {
                "support": ["Springfield, MA"],
                "text": "Graph resolution (conditioned): Springfield, MA",
                "candidate_beliefs": {
                    "Springfield, MA": {"A": 0.75, "not_A": 0.0, "omega": 0.25}
                },
            }

        def verify_direct(self, evidence, first_response=None, assumptions=None):
            self.evidence = evidence
            return {
                "verdict": "good",
                "needs_human": False,
                "followups": 0,
                "history": [],
            }

    verifier = ConditioningVerifier()
    monkeypatch.setattr(live_demo, "stream_chat_complete", fake_chat)
    monkeypatch.setattr(live_demo, "_terminal_fd", lambda: None)
    monkeypatch.setattr(
        live_demo, "build_verifier", lambda q, r, m, **kw: verifier
    )
    typed = iter(["the celtics are a great team", ""])

    final, _verifier, _response = live_demo.run_question(
        "What is the weather in Springfield today?",
        cfg=cfg,
        ask=lambda prompt: next(typed),
        emit=lambda line: None,
    )

    assert verifier.clues == ["the celtics are a great team"]
    assert "Human clue conditioning:" in verifier.evidence
    assert final["graph_conditioning"]["support"] == ["Springfield, MA"]


def _checkpoint_cfg():
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "model": {"env_var": "LIVE_DEMO_TEST_MODEL", "id": "m-test"},
            "human_question_markers": ["which"],
            "max_iterations": 7,
        }
    )


class _CheckpointVerifier(FakeVerifier):
    def __init__(self):
        super().__init__()
        self.evidence_calls = []

    def verify_direct(self, evidence, first_response=None, assumptions=None):
        self.evidence_calls.append(evidence)
        return {
            "verdict": "good",
            "needs_human": False,
            "followups": 0,
            "history": [],
        }


def _wire_checkpoint_demo(monkeypatch, verifier, model_answers):
    prompts = []
    answers = iter(model_answers)

    def fake_chat(prompt, model_id=None):
        prompts.append(prompt)
        return iter([next(answers)])

    monkeypatch.setattr(live_demo, "stream_chat_complete", fake_chat)
    monkeypatch.setattr(live_demo, "_terminal_fd", lambda: None)
    monkeypatch.setattr(
        live_demo, "build_verifier", lambda q, r, m, **kw: verifier
    )
    return prompts


def test_every_mode_scores_each_model_response(monkeypatch, caplog):
    import logging

    verifier = _CheckpointVerifier()
    _wire_checkpoint_demo(
        monkeypatch,
        verifier,
        [
            "Hi, I'm Muse Spark!\n\nWhat did you want to know about today?",
            "A cat is a **feline**.",
        ],
    )
    typed = iter(["well a dog is a canine and a cat is a what?", ""])

    with caplog.at_level(logging.INFO, logger="zydah.live_demo"):
        final, _verifier, response = live_demo.run_question(
            "what do you know today?",
            cfg=_checkpoint_cfg(),
            ask=lambda prompt: next(typed),
            emit=lambda line: None,
        )

    assert response == "A cat is a **feline**."
    assert len(verifier.evidence_calls) == 2
    assert "Claim to evaluate: Hi, I'm Muse Spark!" in verifier.evidence_calls[0]
    assert "Claim to evaluate: A cat is a **feline**." in verifier.evidence_calls[1]
    assert caplog.text.count("zydah_json ") == 2
    assert final["verdict"] == "good"


def test_substantive_mode_skips_pure_question_but_scores_answer(
    monkeypatch, caplog
):
    import logging

    verifier = _CheckpointVerifier()
    _wire_checkpoint_demo(
        monkeypatch,
        verifier,
        [
            "Hi, I'm Muse Spark!\n\nWhat did you want to know about today?",
            "A cat is a **feline**.",
        ],
    )
    typed = iter(["well a dog is a canine and a cat is a what?", ""])

    with caplog.at_level(logging.INFO, logger="zydah.live_demo"):
        live_demo.run_question(
            "what do you know today?",
            cfg=_checkpoint_cfg(),
            ask=lambda prompt: next(typed),
            emit=lambda line: None,
            score_mode="substantive",
        )

    assert len(verifier.evidence_calls) == 1
    assert "Claim to evaluate: A cat is a **feline**." in verifier.evidence_calls[0]
    assert "zydah_unscored reason=pure_question mode=substantive" in caplog.text
    assert caplog.text.count("zydah_json ") == 1


def test_substantive_response_ending_in_question_is_substantive():
    assert live_demo._is_substantive_response(
        "It is cloudy in Springfield, MA today. Which Springfield did you mean?"
    )
    assert live_demo._is_substantive_response(
        "Happy Wednesday!\n\n"
        "* A dog is a **canine** (family Canidae)\n"
        "* And a cat is a **feline** (family Felidae)!\n\n"
        "What else do you want to explore?"
    )
    assert not live_demo._is_substantive_response(
        "What did you want to know about today?"
    )
    assert not live_demo._is_substantive_response("")


def test_gate_quit_keeps_prior_checkpoint(monkeypatch):
    verifier = _CheckpointVerifier()
    _wire_checkpoint_demo(
        monkeypatch, verifier, ["Prince sang in the rain."]
    )

    result, _verifier, _response = live_demo.run_question(
        "who sang in the rain?",
        cfg=_checkpoint_cfg(),
        ask=lambda prompt: "q",
        emit=lambda line: None,
    )

    assert len(verifier.evidence_calls) == 1
    assert result["verdict"] == "good"
    assert result["quit"] is True


def test_main_passes_score_mode(monkeypatch):
    seen = []

    def fake_run_question(question, **kwargs):
        seen.append(kwargs["score_mode"])
        return (
            {"verdict": "good", "needs_human": False, "followups": 0, "history": []},
            object(),
            "resp",
        )

    def no_terminal(prompt):
        raise EOFError

    monkeypatch.setattr(live_demo, "run_question", fake_run_question)
    monkeypatch.setattr(live_demo, "terminal_input", no_terminal)

    assert live_demo.main(["--question", "hi"]) == 0
    assert live_demo.main(["--score-mode", "substantive", "--question", "hi"]) == 0
    assert seen == ["every", "substantive"]


def test_artifact_turns_carry_interval_decision_fields():
    from zydah.fusion import make_mass

    frame = frozenset({"pass", "partial", "fail"})
    result = {
        "verdict": "good",
        "needs_human": False,
        "followups": 0,
        "history": [
            {
                "mass": make_mass(
                    frame,
                    {"pass": 0.65, "partial": 0.10, "fail": 0.05, frame: 0.2},
                ),
                "zydah": {
                    "bel": 0.65,
                    "pl": 0.85,
                    "width": 0.2,
                    "conflict": 0.0,
                    "decision": "escalate",
                },
            }
        ],
    }
    artifact = live_demo.artifact_for("q?", "a.", result)
    turn = artifact["turns"][0]
    assert turn["bel"] == 0.65
    assert turn["pl"] == 0.85
    assert turn["width"] == 0.2
    assert turn["conflict"] == 0.0
    assert turn["decision"] == "escalate"


def test_build_verifier_enables_jsonl_cycle_log(monkeypatch):
    captured = {}

    class FakeVerifier:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(live_demo, "ZydahVerifier", FakeVerifier)
    live_demo.build_verifier("hello?", "hi.", "m-test", emit=lambda line: None)
    assert captured["log_path"] == live_demo.VERIFIER_LOG

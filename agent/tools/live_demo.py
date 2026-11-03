"""Interactive terminal demo for Zydah's live direct loop.

Unlike the pytest level tests, this tool does not inject scripted model
answers. It asks the configured model for a first response, sends that
response through Zydah, displays the follow-up transcript and belief
frame, and pauses at the terminal whenever Zydah needs a human choice.

Usage:
    uv run python tools/live_demo.py
    uv run python tools/live_demo.py --question "What is the weather in Springfield today?"
    uv run python tools/live_demo.py --save outputs/live_demo/latest.json

Environment values are read from the repository-root ``.env`` by
default. No secret values are printed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid

try:
    import termios
except ImportError:  # non-POSIX fallback; input() still waits for Enter
    termios = None  # type: ignore[assignment]
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional

from dotenv import load_dotenv

from zydah.choices import OTHER_VALUE
from zydah.fusion import belief_frame
from zydah.graph import build_place_graph
from zydah.graph_resolver import NumpyResolver
from zydah.interface import capture_first_response
from zydah.similarity import SentenceTransformerScorer
from zydah.transport import chat_complete
from zydah.verifier import (
    ZydahVerifier,
    format_needs_human,
    format_transcript,
    load_verifier_config,
    make_model_responder,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FRAME = frozenset({"pass", "partial", "fail"})
QUIT_WORDS = {"q", "quit", "exit"}
Ask = Callable[[str], str]
Emit = Callable[[str], None]


def _is_quit(text: str) -> bool:
    return text.strip().lower() in QUIT_WORDS


def _flush_pending_input(stream) -> None:
    """Discard type-ahead typed before the prompt appeared."""
    if termios is None:
        return
    try:
        termios.tcflush(stream.fileno(), termios.TCIFLUSH)
    except Exception:
        return


def terminal_input(prompt: str) -> str:
    """Read one line from the controlling terminal.

    A demo launched with stdin piped or closed must still wait for the
    human at the keyboard. When stdin is already a TTY, use it; when it
    is not, fall back to ``/dev/tty`` instead of consuming EOF and
    racing through prompts. Pending type-ahead is discarded first, so
    a line pasted or typed before the prompt cannot submit itself.
    """
    if sys.stdin is not None and sys.stdin.isatty():
        _flush_pending_input(sys.stdin)
        return input(prompt)
    try:
        with open("/dev/tty", "r+", encoding="utf-8") as tty:
            _flush_pending_input(tty)
            tty.write(prompt)
            tty.flush()
            line = tty.readline()
    except OSError as exc:
        raise EOFError("no interactive terminal is available") from exc
    if line == "":
        raise EOFError("no interactive terminal is available")
    return line.rstrip("\n")


def _model_id(cfg) -> str:
    import os

    return os.environ.get(str(cfg.model.env_var)) or str(cfg.model.id)


def _asks_human(text: str, cfg) -> bool:
    """True when a model response is itself a question for the human."""
    if "?" not in text:
        return False
    markers = []
    if cfg is not None:
        try:
            markers = list(cfg.human_question_markers)
        except Exception:
            markers = []
    lowered = text.lower()
    return any(str(marker).lower() in lowered for marker in markers)


def _assumptions_for(question: str, response: str) -> List[str]:
    combined = f"{question}\n{response}"
    if re.search(r"\bSpringfield\b(?!\s*,\s*[A-Z]{2}\b)", combined):
        return ["Springfield has no state qualifier."]
    return []


def build_verifier(question: str, response: str, model_id: str):
    """Build the live verifier, adding the Springfield graph when relevant."""
    cfg = load_verifier_config()
    combined = f"{question}\n{response}".lower()
    graph_resolver = None
    graph_factory = None
    similarity_scorer = None
    if "springfield" in combined:
        graph_resolver = NumpyResolver()
        graph_factory = lambda evidence: build_place_graph()
        similarity_scorer = SentenceTransformerScorer(
            str(cfg.similarity_model)
        )
    return ZydahVerifier(
        FRAME,
        responder=make_model_responder(),
        config=cfg,
        model_id=model_id,
        graph_resolver=graph_resolver,
        graph_factory=graph_factory,
        similarity_scorer=similarity_scorer,
        session_id=f"demo-verifier-{uuid.uuid4().hex[:8]}",
    )


def _latest_belief_frame(result: Mapping[str, object]):
    history = result.get("history") or []
    for entry in reversed(history):  # type: ignore[union-attr]
        if isinstance(entry, Mapping) and "mass" in entry:
            return belief_frame(entry["mass"], FRAME)  # type: ignore[arg-type]
    return None


def _final_response_line(result: Mapping[str, object]) -> str:
    """One-sentence final response shown before the verdict metrics."""
    conditioning = result.get("graph_conditioning")
    if isinstance(conditioning, Mapping):
        support = conditioning.get("support") or []
        if support:
            return f"you are asking about {', '.join(map(str, support))}."
    if result.get("needs_human"):
        return "I still need one detail before I can conclude."
    if result.get("verdict") == "good":
        return "the response checks out."
    return "the response is still not dependable."


def render_result(result: Mapping[str, object], emit: Emit) -> None:
    """Render the verifier's current state without hiding follow-ups."""
    emit(f"[zydah] {_final_response_line(result)}")
    emit(
        "verdict={verdict} needs_human={needs_human} followups={followups}".format(
            verdict=result.get("verdict"),
            needs_human=result.get("needs_human", False),
            followups=result.get("followups", 0),
        )
    )
    history = result.get("history") or []
    transcript = format_transcript(history)  # type: ignore[arg-type]
    if transcript:
        emit(transcript)
    frame_value = _latest_belief_frame(result)
    if frame_value is not None:
        emit("latest belief_frame:")
        emit(json.dumps(frame_value, indent=2))
    conditioning = result.get("graph_conditioning")
    if conditioning:
        emit("graph_conditioning:")
        emit(json.dumps(conditioning, indent=2))


def _choice_for_key(result: Mapping[str, object], key: str):
    choices = result.get("choices") or []
    for choice in choices:  # type: ignore[union-attr]
        if str(choice.get("key", "")).upper() == key:
            return choice
    return None


def interact(result, verifier, ask: Ask = terminal_input, emit: Emit = print):
    """Pause for terminal choices and resume the direct loop."""
    current = result
    while current.get("needs_human"):
        choices = current.get("choices") or []
        if not choices:
            rendered = format_needs_human(current)
            if rendered:
                emit(rendered)
            emit("This pause has no structured choices to resume from.")
            return current
        emit(format_needs_human(current))
        try:
            key = ask("[you: choice key] ").strip().upper()
        except EOFError:
            emit("No terminal input available; leaving the loop paused.")
            return current
        if _is_quit(key):
            return current
        selected = _choice_for_key(current, key)
        if selected is None:
            emit(f"Choice {key!r} was not offered; try again.")
            continue
        other_text = None
        if str(selected.get("value")) == OTHER_VALUE:
            try:
                other_text = ask("[you: free text] ")
            except EOFError:
                emit("No terminal input available; leaving the loop paused.")
                return current
            if _is_quit(other_text):
                return current
            if not other_text.strip():
                emit("Other requires free text; try again.")
                continue
        try:
            current = verifier.resume_direct(
                current, key, other_text=other_text
            )
        except Exception as exc:  # keep the demo terminal usable
            emit(f"[zydah] resume failed: {exc}")
            return current
        render_result(current, emit)
    return current


def artifact_for(question: str, response: str, result: Mapping[str, object]) -> Dict:
    """JSON artifact for one interactive run (ledger objects excluded)."""
    turns = []
    for idx, entry in enumerate(result.get("history") or [], start=1):  # type: ignore[union-attr]
        if not isinstance(entry, Mapping):
            continue
        turn = {"turn": idx}
        if "mass" in entry:
            turn["belief_frame"] = belief_frame(entry["mass"], FRAME)  # type: ignore[arg-type]
        if entry.get("response"):
            turn["response"] = entry["response"]
        if entry.get("followup"):
            turn["followup"] = entry["followup"]
        turns.append(turn)
    artifact = {
        "question": question,
        "response": response,
        "verdict": result.get("verdict"),
        "needs_human": result.get("needs_human", False),
        "followups": result.get("followups", 0),
        "turns": turns,
    }
    if result.get("graph_conditioning"):
        artifact["graph_conditioning"] = result["graph_conditioning"]
    return artifact


class SaveTee:
    """Terminal emit mirrored into a JSON artifact, tee-style.

    Every emitted line is printed and immediately appended to the
    saved JSON under ``transcript``. When a question finishes, the
    same file is rewritten with the structured artifact (question,
    response, verdict, turns, belief frames) plus that transcript.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.transcript: List[str] = []
        self.question: Optional[str] = None
        self.response: Optional[str] = None
        self.result: Optional[Mapping[str, object]] = None

    def begin(self, question: str) -> None:
        self.question = question
        self.response = None
        self.result = None
        self.flush()

    def emit(self, line: object = "") -> None:
        text = str(line)
        print(text)
        self.transcript.append(text)
        self.flush()

    def finish(
        self, question: str, response: str, result: Mapping[str, object]
    ) -> None:
        self.question = question
        self.response = response
        self.result = result
        self.flush()

    def flush(self) -> None:
        if self.result is not None and self.question is not None:
            payload = artifact_for(
                self.question, self.response or "", self.result
            )
        else:
            payload = {}
            if self.question is not None:
                payload["question"] = self.question
        payload["transcript"] = list(self.transcript)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2) + "\n")


def run_question(
    question: str,
    *,
    cfg=None,
    ask: Ask = terminal_input,
    emit: Emit = print,
):
    """Ask the live model once, verify its answer, and pause for input."""
    cfg = cfg or load_verifier_config()
    model_id = _model_id(cfg)
    emit("[muse] calling the configured model...")
    response = chat_complete(question, model_id=model_id)
    emit("[muse]")
    emit(response)
    verifier = build_verifier(question, response, model_id)
    context_lines = [f"Question: {question}", f"Response: {response}"]
    while _asks_human(response, cfg):
        # The model's response is a route to the human, not an answer
        # to score. Collect the human's answer and ask the model to
        # continue before Zydah evaluates anything.
        emit("[human] the model asked you a question; answer it here.")
        try:
            human_answer = ask("[you: answer] ")
        except EOFError:
            human_answer = "q"
        if _is_quit(human_answer):
            paused = {
                "verdict": False,
                "needs_human": True,
                "human_question": response,
                "followups": 0,
                "history": [],
            }
            return paused, verifier, response
        if not human_answer.strip():
            continue
        context_lines.append(f"Human answer: {human_answer}")
        followup_prompt = (
            f"Original question: {question}\n"
            f"You asked for clarification: {response}\n"
            f"Human answer: {human_answer}\n"
            "Please now answer the original question using that answer."
        )
        response = chat_complete(followup_prompt, model_id=model_id)
        emit("[muse]")
        emit(response)
        context_lines.append(f"Response: {response}")
    first = capture_first_response(
        text=response,
        model_id=model_id,
        session_id=f"demo-model-{uuid.uuid4().hex[:8]}",
        playground="A",
    )
    evidence = "\n".join(
        [*context_lines, f"Claim to evaluate: {response}"]
    )
    result = verifier.verify_direct(
        evidence,
        first_response=first,
        assumptions=_assumptions_for(question, response),
    )
    render_result(result, emit)
    final = interact(result, verifier, ask=ask, emit=emit)
    return final, verifier, response


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive Zydah live demo")
    parser.add_argument("--question", help="ask one question, then still pause for any human choices")
    parser.add_argument("--env", default=str(REPO_ROOT / ".env"), help="env file to load")
    parser.add_argument("--save", type=Path, help="write the latest run artifact JSON here")
    args = parser.parse_args(argv)

    env_path = Path(args.env)
    if env_path.exists():
        load_dotenv(env_path)
    cfg = load_verifier_config()
    tee = SaveTee(args.save) if args.save else None
    emit = tee.emit if tee is not None else print
    emit("Zydah live demo. Type q at a prompt to quit.")
    questions = [args.question] if args.question else []
    while True:
        if questions:
            question = questions.pop(0)
        else:
            try:
                question = terminal_input("[you] ")
            except EOFError:
                emit("No interactive terminal is available for prompts.")
                return 0
            if _is_quit(question):
                return 0
            if not question.strip():
                continue
        if tee is not None:
            tee.begin(question)
        try:
            result, _verifier, response = run_question(
                question, cfg=cfg, emit=emit
            )
        except Exception as exc:
            emit(f"[zydah] live call failed: {exc}")
            if args.question:
                return 1
            continue
        if tee is not None:
            tee.finish(question, response, result)
            emit(f"saved: {args.save}")
        if args.question:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())



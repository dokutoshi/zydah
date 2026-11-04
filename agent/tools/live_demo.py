"""Interactive terminal demo for Zydah's live direct loop.

Unlike the pytest level tests, this tool does not inject scripted model
answers. It asks the configured model for a first response, sends that
response through Zydah, displays the follow-up transcript and belief
frame, and pauses at the terminal whenever Zydah needs a human choice.

Usage:
    uv run python tools/live_demo.py
    uv run python tools/live_demo.py --question "What is the weather in Springfield today?"
    uv run python tools/live_demo.py --save outputs/live_demo/latest.json
    uv run python tools/live_demo.py --verbose   # INFO logs to stderr
    uv run python tools/live_demo.py --log-file trace.log

Environment values are read from the repository-root ``.env`` by
default. No secret values are printed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import re
import select
import sys
import threading
import time
import traceback
import tty
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
from zydah.transport import chat_complete, stream_chat_complete
from zydah.verifier import (
    ZydahVerifier,
    format_needs_human,
    format_transcript,
    load_verifier_config,
    make_mass_elicitor,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FRAME = frozenset({"pass", "partial", "fail"})
QUIT_WORDS = {"q", "quit", "exit"}
RESOLVED_INVITE = "[zydah] any time you have a followup ping me - Zydah will assist."
Ask = Callable[[str], str]
Emit = Callable[[str], None]

logger = logging.getLogger("zydah.live_demo")


def _is_quit(text: str) -> bool:
    return text.strip().lower() in QUIT_WORDS


ERROR_LOG = REPO_ROOT / "outputs" / "live_demo" / "errors.log"


def _log_error(context: str, exc: Exception, path: Path = ERROR_LOG) -> None:
    """Append one JSONL error record (best effort; never raises)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.time(),
            "context": context,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except Exception:
        pass


def _excepthook(exc_type, exc, tb):
    logging.getLogger("zydah.live_demo").error(
        "uncaught", exc_info=(exc_type, exc, tb)
    )
    sys.__stdout__.write(f"\n[zydah] fatal: {exc_type.__name__}: {exc} (see log)\n")
    sys.__stdout__.flush()


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
        # input(prompt) sent the prompt to stderr on this runtime
        # (it landed in trace.log under 2>trace.log, off-screen).
        # The prompt is UI: write it to stdout ourselves, and read
        # with readline() so no readline backend is involved.
        sys.stdout.write(prompt)
        sys.stdout.flush()
        line = sys.stdin.readline()
        if line == "":
            raise EOFError("no interactive terminal is available")
        return line.rstrip("\n")
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


class TerminalBox:
    """Bidirectional terminal box: stream above, live input below.

    Raw-mode async loop (Muse style): output text is written
    as it arrives while the human's input buffer stays editable at
    the bottom. Enter sends the buffer; Backspace edits; Ctrl-C
    interrupts; Ctrl-D on an empty buffer is EOF. When the human has
    typed nothing, streamed output flows untouched.
    """

    def __init__(self, fd: int, out=None) -> None:
        self.fd = fd
        self.out = out if out is not None else sys.stdout
        self.buffer = ""
        self._saved_attrs = None

    def __enter__(self) -> "TerminalBox":
        if termios is not None and os.isatty(self.fd):
            self._saved_attrs = termios.tcgetattr(self.fd)
            tty.setraw(self.fd)
            # Do NOT render the input prompt yet: streamed output must
            # not appear after a [you] prefix. The box becomes visible
            # only when the human actually types.
        return self

    def __exit__(self, *exc) -> bool:
        if self._saved_attrs is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved_attrs)
            self._saved_attrs = None
            self.out.write("\r\n")
            self.out.flush()
        return False

    def render(self) -> None:
        self.out.write(f"\r\x1b[K[you] {self.buffer}")
        self.out.flush()

    def feed_output(self, text: str) -> None:
        """Write streamed output, then restore the input line.

        Newlines are translated to CR LF: the box runs the terminal
        in raw mode, where a bare line feed moves down without
        returning to column zero (staircase output).
        """
        text = text.replace("\r\n", "\n").replace("\n", "\r\n")
        if self.buffer:
            self.out.write("\r\x1b[K")
            self.out.write(text)
            if text and not text.endswith("\n"):
                self.out.write("\n")
            self.render()
        else:
            self.out.write(text)
        self.out.flush()

    def poll_submit(self, timeout: float = 0.0) -> Optional[str]:
        """Process pending keys; return the sent buffer on Enter."""
        ready, _, _ = select.select([self.fd], [], [], timeout)
        if not ready:
            return None
        data = os.read(self.fd, 1)
        if not data:
            raise EOFError
        byte = data[0]
        if byte == 3:  # Ctrl-C
            raise KeyboardInterrupt
        if byte == 4:  # Ctrl-D
            if not self.buffer:
                raise EOFError
            return None
        if byte in (10, 13):  # Enter sends the buffer
            line = self.buffer
            self.buffer = ""
            self.out.write("\r\n")
            self.out.flush()
            return line
        if byte in (8, 127):  # Backspace
            if self.buffer:
                self.buffer = self.buffer[:-1]
                self.render()
            return None
        if byte == 27:  # escape sequence: arrows drain; iTerm2 paste
            seq = ""
            for _ in range(8):
                ready, _, _ = select.select([self.fd], [], [], 0.01)
                if not ready:
                    break
                seq += os.read(self.fd, 1).decode("utf-8", "ignore")
                if seq.endswith("~"):
                    break
            if seq == "[200~":  # bracketed paste start (iTerm2)
                end = "\x1b[201~"
                raw_paste = ""
                while not raw_paste.endswith(end):
                    ready, _, _ = select.select([self.fd], [], [], 0.5)
                    if not ready:
                        break
                    raw_paste += os.read(self.fd, 1).decode(
                        "utf-8", "ignore"
                    )
                pasted = (
                    raw_paste[: -len(end)]
                    if raw_paste.endswith(end)
                    else raw_paste
                )
                text = pasted.replace("\r", " ").replace("\n", " ")
                if text:
                    if not self.buffer:
                        self.out.write("\n")
                    self.buffer += text
                    self.render()
            return None
        if byte >= 32:
            if not self.buffer:
                # First keypress: move off the streamed-text line
                # before showing the input buffer.
                self.out.write("\n")
            self.buffer += data.decode("utf-8", "ignore")
            self.render()
        return None


def _terminal_fd() -> Optional[int]:
    """Readable terminal fd for the box, or None when there is none."""
    try:
        if sys.stdin is not None and sys.stdin.isatty():
            return sys.stdin.fileno()
    except (OSError, ValueError):
        pass
    try:
        return os.open("/dev/tty", os.O_RDWR)
    except OSError:
        return None


def _model_id(cfg) -> str:
    return os.environ.get(str(cfg.model.env_var)) or str(cfg.model.id)


def _ends_with_question(text: str) -> bool:
    """True when the last meaningful line of a response is a question.

    The logical marker is '?'. After it, no punctuation may follow
    except closers: end-quote (single or double), tick mark, or ')'.
    Anything else after the '?' means the question is not the ending.
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return False
    tail = lines[-1].rstrip("\"'`)")
    return tail.endswith("?")


def _asks_human(text: str, cfg) -> bool:
    """True when a model response is itself a question for the human.

    Two signals, either suffices: a YAML human-question marker anywhere
    in the text, or the response structurally ending in a question
    (last meaningful line ends with '?'). The structural rule is
    demo-local: it does not widen the YAML markers, and the human
    still directs -- an empty answer ends the clarification round and
    Zydah evaluates what it has.
    """
    if "?" not in text:
        return False
    markers = []
    if cfg is not None:
        try:
            markers = list(cfg.human_question_markers)
        except Exception:
            markers = []
    lowered = text.lower()
    if any(str(marker).lower() in lowered for marker in markers):
        return True
    return _ends_with_question(text)


def _assumptions_for(question: str, response: str) -> List[str]:
    combined = f"{question}\n{response}"
    if re.search(r"\bSpringfield\b(?!\s*,\s*[A-Z]{2}\b)", combined):
        return ["Springfield has no state qualifier."]
    return []


def _streaming_responder(emit: Emit):
    """Follow-up responder that streams its answer as it is generated."""

    def responder(prompt: str, model_id: str) -> str:
        emit("...Thinking")
        emit("[muse]")
        parts: List[str] = []
        for chunk in stream_chat_complete(prompt, model_id=model_id):
            sys.stdout.write(chunk)
            sys.stdout.flush()
            parts.append(chunk)
        sys.stdout.write("\n")
        sys.stdout.flush()
        text = "".join(parts)
        recorder = getattr(emit, "__self__", None)
        if hasattr(recorder, "streamed"):
            recorder.streamed(text)  # type: ignore[attr-defined]
        return text

    return responder


def _thinking_elicitor(base_elicitor, emit: Emit):
    """Wrap a mass elicitor with Zydah's own thinking semaphore."""

    def elicitor(evidence, frame, model_id):
        emit("[zydah] ...Thinking")
        return base_elicitor(evidence, frame, model_id)

    return elicitor


def build_verifier(
    question: str,
    response: str,
    model_id: str,
    emit: Emit = print,
    show_zydah: bool = False,
):
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
    base_elicitor = make_mass_elicitor(
        str(cfg.direct_mass_prompt_template)
    )

    return ZydahVerifier(
        FRAME,
        responder=_streaming_responder(emit),
        mass_elicitor=_thinking_elicitor(
            base_elicitor, emit if show_zydah else (lambda line: None)
        ),
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
    """One-sentence final response shown before the verdict metrics.

    The validator's ``good`` only means *confident* (narrow interval);
    the sentence follows the dominant mass, so a confident fail reads
    as a fail, not as an endorsement.
    """
    conditioning = result.get("graph_conditioning")
    if isinstance(conditioning, Mapping):
        support = conditioning.get("support") or []
        if support:
            return f"you are asking about {', '.join(map(str, support))}."
    if result.get("needs_human"):
        return "I still need one detail before I can conclude."
    if result.get("verdict") == "good":
        history = result.get("history") or []
        for entry in reversed(history):  # type: ignore[union-attr]
            if not isinstance(entry, Mapping):
                continue
            betp = (entry.get("zydah") or {}).get("pignistic") or {}
            if betp:
                dominant = max(betp, key=lambda label: betp[label])
                if dominant == "fail":
                    return "the response does not check out."
                if dominant == "partial":
                    return "the response is only partly supported."
                break
        return "the response checks out."
    return "the response is still not dependable."


def render_result(
    result: Mapping[str, object], emit: Emit, show_zydah: bool = False
) -> None:
    """Render the verifier's current state.

    Default view is the concluded [zydah] sentence plus the verdict
    line. With ``show_zydah`` (--zydah), also show the internal view:
    the follow-up transcript and the belief frame as it adjusted on
    every turn, plus any graph conditioning.
    """
    emit(f"[zydah] {_final_response_line(result)}")
    emit(
        "verdict={verdict} needs_human={needs_human} followups={followups}".format(
            verdict=result.get("verdict"),
            needs_human=result.get("needs_human", False),
            followups=result.get("followups", 0),
        )
    )
    if not show_zydah:
        return
    history = result.get("history") or []
    transcript = format_transcript(history)  # type: ignore[arg-type]
    if transcript:
        emit(transcript)
    for idx, entry in enumerate(history, start=1):  # type: ignore[union-attr]
        if isinstance(entry, Mapping) and "mass" in entry:
            emit(f"turn {idx} belief_frame:")
            emit(
                json.dumps(
                    belief_frame(entry["mass"], FRAME), indent=2  # type: ignore[arg-type]
                )
            )
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


def interact(
    result,
    verifier,
    ask: Ask = terminal_input,
    emit: Emit = print,
    show_zydah: bool = False,
):
    """Pause for terminal choices and resume the direct loop."""
    current = result
    while current.get("needs_human"):
        choices = current.get("choices") or []
        if not choices:
            # The question already streamed above; do not dump it again.
            emit("[human] the model's question above is for you.")
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
            _log_error("resume_direct", exc)
            emit(f"[zydah] resume failed: {exc} (logged to errors.log)")
            return current
        render_result(current, emit, show_zydah=show_zydah)
        _log_zydah_json("", "", current)
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


def _log_zydah_json(
    question: str, response: str, result: Mapping[str, object]
) -> None:
    """Log the Zydah JSON artifact as one logger.info record.

    Single line prefixed ``zydah_json`` so a log consumer (e.g. the
    future Streamlit chat view) can parse each update and redraw the
    belief frame on every prompt from ``turns[].belief_frame``;
    ``belief_frame`` also carries the latest turn's frame directly.
    """
    payload = artifact_for(question, response, result)
    if not question:
        payload.pop("question", None)
    if not response:
        payload.pop("response", None)
    payload["belief_frame"] = _latest_belief_frame(result)
    logger.info("zydah_json %s", json.dumps(payload, default=str))


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

    def streamed(self, text: str) -> None:
        """Record a token-streamed response without re-printing it."""
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


def _stream_model_text(prompt: str, model_id: str, emit: Emit):
    """Stream one model response; let the human type while it lands.

    Async loop: a producer thread feeds response chunks into a queue
    while the main loop selects on the keyboard (TerminalBox). Text
    typed and sent during streaming is returned as the pending buffer
    (the human's next direction); without a terminal this degrades to
    plain streaming. Returns ``(response_text, pending_or_None)``.
    """
    emit("...Thinking")
    emit("[muse]")
    fd = _terminal_fd()
    box: Optional[TerminalBox] = None
    if fd is not None:
        candidate = TerminalBox(fd)
        try:
            candidate.__enter__()
            box = candidate
        except Exception:
            box = None  # raw mode unavailable: degrade to plain stream
    if box is None:
        # No raw terminal: stream through the same emit channel the
        # JSON transcript records, line by line, so what the terminal
        # shows and what the artifact holds cannot diverge.
        parts: List[str] = []
        buffer = ""
        for chunk in stream_chat_complete(prompt, model_id=model_id):
            parts.append(chunk)
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                emit(line)
        if buffer:
            emit(buffer)
        text = "".join(parts)
        if not text.strip():
            return _empty_stream_fallback(prompt, model_id, emit), None
        return text, None

    chunks: "queue.Queue" = queue.Queue()

    def _produce() -> None:
        try:
            for chunk in stream_chat_complete(prompt, model_id=model_id):
                chunks.put(chunk)
        except Exception as exc:  # surfaced on the main loop below
            chunks.put(exc)
        finally:
            chunks.put(None)

    threading.Thread(target=_produce, daemon=True).start()
    parts = []
    pending: Optional[str] = None
    try:
        while True:
            try:
                item = chunks.get(timeout=0.05)
            except queue.Empty:
                item = _STREAM_PENDING
            if isinstance(item, Exception):
                raise item
            if isinstance(item, str):
                parts.append(item)
                box.feed_output(item)
            submitted = box.poll_submit(0)
            if submitted is not None:
                pending = submitted
            if item is None:
                break
    finally:
        box.__exit__()
    text = "".join(parts)
    if text and not text.endswith("\n"):
        sys.stdout.write("\n")
        sys.stdout.flush()
    recorder = getattr(emit, "__self__", None)
    if hasattr(recorder, "streamed"):
        recorder.streamed(text)  # type: ignore[attr-defined]
    if not text.strip():
        return _empty_stream_fallback(prompt, model_id, emit), pending
    return text, pending


_STREAM_PENDING = object()


def _empty_stream_fallback(prompt: str, model_id: str, emit: Emit) -> str:
    """A stream that yields nothing must not become an empty claim.

    Logs the empty stream, retries once as a plain call, and emits
    the text through the normal channel so it is visible and saved.
    """
    _log_error(
        "empty_stream",
        RuntimeError("stream yielded no content; fell back to plain call"),
    )
    text = chat_complete(prompt, model_id=model_id)
    for line in text.splitlines() or [""]:
        emit(line)
    return text


def run_question(
    question: str,
    *,
    cfg=None,
    ask: Ask = terminal_input,
    emit: Emit = print,
    show_zydah: bool = False,
):
    """Ask the live model once, verify its answer, and pause for input."""
    cfg = cfg or load_verifier_config()
    model_id = _model_id(cfg)
    logger.info("question: %r", question)
    response, pending_direction = _stream_model_text(question, model_id, emit)
    logger.info("model response:\n%s", response)
    logger.info(
        "asks_human=%s pending_direction=%r",
        _asks_human(response, cfg),
        pending_direction,
    )
    verifier = build_verifier(
        question, response, model_id, emit=emit, show_zydah=show_zydah
    )
    context_lines = [f"Question: {question}", f"Response: {response}"]
    try:
        clarification_limit = int(cfg.max_iterations)
    except Exception:
        clarification_limit = 5  # config always defines this; test cfg may not
    clarifications = 0

    def _clarify(current_response: str):
        """Clarification rounds for a question-ending model response.

        The model's response is a route to the human, not an answer
        to score. Collect the human's answer and ask the model to
        continue before Zydah evaluates anything. An empty answer
        ends the round: evaluate what we have rather than re-asking
        the same question forever. Returns (response, paused|None).
        """
        nonlocal clarifications, pending_direction
        while (
            _asks_human(current_response, cfg)
            and clarifications < clarification_limit
        ):
            clarifications += 1
            # Call-tree order: after [muse], [zydah] prints first.
            emit("[zydah] the model needs one detail before I can conclude.")
            emit("[human] the model asked you a question; answer it here.")
            try:
                human_answer = ask("[you: answer] ")
            except EOFError:
                human_answer = "q"
            if _is_quit(human_answer):
                paused = {
                    "verdict": False,
                    "needs_human": True,
                    "quit": True,
                    "human_question": current_response,
                    "followups": 0,
                    "history": [],
                }
                return current_response, paused
            logger.info("human answer: %r", human_answer)
            if not human_answer.strip():
                break
            context_lines.append(f"Human answer: {human_answer}")
            followup_prompt = (
                f"Original question: {question}\n"
                f"You asked for clarification: {current_response}\n"
                f"Human answer: {human_answer}\n"
                "Please now answer the original question using that answer."
            )
            current_response, pending_next = _stream_model_text(
                followup_prompt, model_id, emit
            )
            logger.info("model response (continuation):\n%s", current_response)
            logger.info("asks_human=%s", _asks_human(current_response, cfg))
            if pending_next is not None:
                pending_direction = pending_next
            context_lines.append(f"Response: {current_response}")
        return current_response, None

    response, paused = _clarify(response)
    if paused is not None:
        return paused, verifier, response
    # The human directs from the input buffer: typed text is sent as
    # a direction on Enter; an empty buffer simply releases Zydah to
    # evaluate the response as-is. Nothing is scored before this.
    # Buffered text ending in '?' is not a direction, though: it is
    # a new question for the model. Route it (recorded as a Human
    # question), score its answer, and gate again on that answer.
    question_rounds = 0
    while True:
        try:
            direction = (
                pending_direction
                if pending_direction is not None
                else ask("[you] ")
            )
        except EOFError:
            direction = "q"
        pending_direction = None
        logger.info("gate direction: %r", direction)
        if _is_quit(direction):
            paused = {
                "verdict": False,
                "needs_human": True,
                "quit": True,
                "human_question": response,
                "followups": 0,
                "history": [],
            }
            return paused, verifier, response
        if not direction.strip():
            break
        if not _ends_with_question(direction.strip()):
            context_lines.append(f"Human direction: {direction.strip()}")
            break
        question_rounds += 1
        if question_rounds > clarification_limit:
            context_lines.append(f"Human direction: {direction.strip()}")
            break
        logger.info("gate question: %r", direction)
        context_lines.append(f"Human question: {direction.strip()}")
        followup_prompt = (
            f"Original question: {question}\n"
            f"Your response: {response}\n"
            f"Human follow-up question: {direction.strip()}\n"
            "Please answer the follow-up question."
        )
        response, pending_next = _stream_model_text(
            followup_prompt, model_id, emit
        )
        logger.info("model response (follow-up):\n%s", response)
        if pending_next is not None:
            pending_direction = pending_next
        context_lines.append(f"Response: {response}")
        response, paused = _clarify(response)
        if paused is not None:
            return paused, verifier, response
    first = capture_first_response(
        text=response,
        model_id=model_id,
        session_id=f"demo-model-{uuid.uuid4().hex[:8]}",
        playground="A",
    )
    evidence = "\n".join(
        [*context_lines, f"Claim to evaluate: {response}"]
    )
    logger.info("evidence scored:\n%s", evidence)
    # No extra Thinking here: the elicitor emits one per scoring call.
    result = verifier.verify_direct(
        evidence,
        first_response=first,
        assumptions=_assumptions_for(question, response),
    )
    _log_zydah_json(question, response, result)
    render_result(result, emit, show_zydah=show_zydah)
    final = interact(
        result, verifier, ask=ask, emit=emit, show_zydah=show_zydah
    )
    return final, verifier, response


def after_verdict(
    result: Mapping[str, object],
    *,
    tee=None,
    question: str = "",
    response: str = "",
    save_path=None,
    emit: Emit = print,
) -> None:
    """Close one demo question: save resolved runs, invite follow-ups.

    A pause is not a finished run: it is announced and left unsaved,
    with no follow-up invite. A resolved run is saved through the tee
    (when one is configured) and always ends with the invite instead
    of dropping silently back to the prompt.
    """
    if result.get("needs_human"):
        # A pause is not a finished run: do not present the
        # tee artifact as a saved final. The transcript has
        # already streamed to the file; the session stays open.
        emit("[zydah] paused for human input; not saved as a finished run.")
        return
    if tee is not None:
        tee.finish(question, response, result)
        emit(f"saved: {save_path}")
    emit(RESOLVED_INVITE)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive Zydah live demo")
    parser.add_argument("--question", help="ask one question, then still pause for any human choices")
    parser.add_argument("--env", default=str(REPO_ROOT / ".env"), help="env file to load")
    parser.add_argument("--save", type=Path, help="write the latest run artifact JSON here")
    parser.add_argument("--zydah", action="store_true", help="show Zydah's internal view: thinking semaphores, follow-up transcript, per-turn belief frames")
    parser.add_argument("--verbose", action="store_true", help="log input, model text, routing decisions, and scored evidence to stderr (logging INFO)")
    parser.add_argument("--log-file", type=Path, help="write INFO logs to this file via a FileHandler (no stderr redirect needed)")
    args = parser.parse_args(argv)

    sys.excepthook = _excepthook
    handlers: List[logging.Handler] = []
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    if args.verbose:
        handlers.append(logging.StreamHandler(sys.stderr))
    if handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s %(name)s: %(message)s",
            handlers=handlers,
            force=True,
        )
        logger.info("verbose logging enabled")

    env_path = Path(args.env)
    if env_path.exists():
        load_dotenv(env_path)
    cfg = load_verifier_config()
    tee = SaveTee(args.save) if args.save else None
    emit = tee.emit if tee is not None else print
    emit("Zydah live demo. Type q at a prompt to quit.")
    questions = [args.question] if args.question else []
    try:
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
            logger.info("input question: %r", question)
            if tee is not None:
                tee.begin(question)
            try:
                result, _verifier, response = run_question(
                    question, cfg=cfg, emit=emit, show_zydah=args.zydah
                )
            except Exception as exc:
                _log_error("live_call", exc)
                emit(f"[zydah] live call failed: {exc} (logged to errors.log)")
                if args.question:
                    return 1
                continue
            if result.get("quit"):
                emit("[zydah] bye")
                return 0
            after_verdict(
                result,
                tee=tee,
                question=question,
                response=response,
                save_path=args.save,
                emit=emit,
            )
            # Like Muse: one persistent [you] buffer. Enter sends
            # the buffered question; Enter on an empty buffer sends
            # nothing and simply redraws the prompt. The loop never
            # quits on Enter alone.
    except KeyboardInterrupt:
        emit("\n[zydah] bye")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())



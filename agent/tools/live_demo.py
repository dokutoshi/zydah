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
    uv run python tools/live_demo.py --score-mode substantive

Environment values are read from the repository-root ``.env`` by
default. No secret values are printed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import subprocess
from datetime import datetime, timezone
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
from rich.console import Console
from rich.json import JSON as RichJSON
from rich.panel import Panel as RichPanel
from rich.table import Table
from rich.text import Text

from zydah.choices import OTHER_VALUE
from zydah.fusion import belief_frame, pignistic
from zydah.graph import build_place_graph
from zydah.graph_resolver import NumpyResolver
from zydah.judges import (
    JUDGE_SPARK,
    JUDGE_SPARK_OPEN,
    JUDGE_WONDER,
    LivePanel,
    MetaApiJudge,
    PanelConfig,
    load_panel_config,
)
from zydah.interface import capture_first_response
from zydah.similarity import SentenceTransformerScorer
from zydah.transport import chat_complete, last_call_stats, stream_chat_complete
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
VERIFIER_LOG = REPO_ROOT / "outputs" / "live_demo" / "cycles.jsonl"


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


_TURN_TAG = re.compile(r"^\[(muse|zydah) turn (\d+)\]")


class TurnSequencer:
    """Emit wrapper that pairs each turn's [muse] and [zydah] blocks.

    Lines tagged ``[muse turn N]`` print immediately and release any
    queued ``[zydah turn N]`` lines; a zydah line for a turn whose muse
    header has not printed yet is held in the queue. Everything else
    passes through untouched. This keeps the terminal (and the saved
    transcript, which shares the emit channel) in muse-before-zydah
    order per turn even when a zydah line is produced first.
    """

    def __init__(self, emit: Emit) -> None:
        self._emit = emit
        self._muse_seen: set = set()
        self._pending: Dict[int, List[str]] = {}

    def __getattr__(self, name: str):
        # Delegate attributes (e.g. __self__ of a bound SaveTee.emit)
        # so streamed-text recording keeps working through the wrapper.
        return getattr(self._emit, name)

    def __call__(self, line: object = "") -> None:
        text = str(line)
        match = _TURN_TAG.match(text)
        if match is None:
            self._emit(text)
            return
        turn = int(match.group(2))
        if match.group(1) == "muse":
            self._muse_seen.add(turn)
            self._emit(text)
            for queued in self._pending.pop(turn, []):
                self._emit(queued)
            return
        if turn in self._muse_seen:
            self._emit(text)
        else:
            self._pending.setdefault(turn, []).append(text)

    def flush_pending(self) -> None:
        """Print any held zydah lines, in turn order, so none are lost."""
        for turn in sorted(self._pending):
            for queued in self._pending[turn]:
                self._emit(queued)
        self._pending.clear()


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


SCORE_MODE_EVERY = "every"
SCORE_MODE_SUBSTANTIVE = "substantive"
SCORE_MODES = (SCORE_MODE_EVERY, SCORE_MODE_SUBSTANTIVE)


def _is_substantive_response(text: str) -> bool:
    """True when a response contains claim-bearing, non-question text.

    Structural and conservative: a substantive answer that ends
    with a question is identified by a period-terminated declarative
    segment before that question. When the response does not end in
    a question, any meaningful text is enough. A greeting followed
    only by a question therefore stays unscored in substantive mode.
    """
    segments = re.split(r"(?<=[.!?])\s+", str(text).strip())
    meaningful_segments = []
    for segment in segments:
        meaningful = segment.strip().strip("*_`#> ")
        if any(char.isalnum() for char in meaningful):
            meaningful_segments.append(meaningful)
    if any(segment.rstrip("\"'`)").endswith(".") for segment in meaningful_segments):
        return True
    lines = [
        line.strip().strip("*_`#> ")
        for line in str(text).splitlines()
        if any(char.isalnum() for char in line)
    ]
    if any(
        re.match(r"^(?:[-*•]|\d+[.)])\s+\S", line.strip())
        and not _ends_with_question(line)
        for line in str(text).splitlines()
    ):
        return True
    if not _ends_with_question(text):
        return bool(meaningful_segments)
    non_question_lines = [
        line for line in lines if not _ends_with_question(line)
    ]
    return len(non_question_lines) > 1


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
    """Follow-up responder that streams its answer as it is generated.

    Internal follow-up calls are labelled with the responding model
    (``[muse responder <model>]``) so they cannot be mistaken for a
    turn response (``[muse turn N]``) or a judge.
    """

    def responder(prompt: str, model_id: str) -> str:
        emit("...Thinking")
        emit(f"[muse responder {model_id}]")
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


def _thinking_elicitor(
    base_elicitor, emit: Emit, capture_zydah_view: bool = False
):
    """Wrap a mass elicitor with Zydah's own thinking semaphore."""

    def elicitor(evidence, frame, model_id):
        if capture_zydah_view:
            logger.info("[zydah] ...Thinking")
        emit("[zydah] ...Thinking")
        return base_elicitor(evidence, frame, model_id)

    return elicitor


def _verifier_log_path() -> Path:
    """Verifier JSONL cycle log for the demo (created on demand)."""
    VERIFIER_LOG.parent.mkdir(parents=True, exist_ok=True)
    return VERIFIER_LOG


def build_verifier(
    question: str,
    response: str,
    model_id: str,
    emit: Emit = print,
    show_zydah: bool = False,
    capture_zydah_view: bool = False,
    panel: bool = False,
    panel_source: str = "yaml",
):
    """Build the live verifier, adding the Springfield graph when relevant.

    With ``panel=True`` (--zydah-panel), scoring runs the 3-judge panel
    path (verify) instead of the single-elicitor direct path: judges
    come from conf/panel.yaml (pydantic PanelConfig), one session id
    per judge expanding the demo's existing separate-session pattern.
    """
    cfg = load_verifier_config()
    panel_obj = None
    if panel:
        if panel_source == "yaml":
            panel_cfg = load_panel_config()
            judge_count = len(panel_cfg.judges)
            make_panel = lambda sessions: LivePanel.from_config(  # noqa: E731
                panel_cfg, FRAME, session_ids=sessions
            )
        elif panel_source == "base":
            # Hardcoded base panel: the three base models, no YAML read.
            effort = str(PanelConfig.model_fields["effort"].default)
            base_judges = [
                MetaApiJudge(JUDGE_WONDER, effort=effort, name="wonder"),
                MetaApiJudge(JUDGE_SPARK, effort=effort, name="spark"),
                MetaApiJudge(
                    JUDGE_SPARK_OPEN, effort=effort, name="spark-open"
                ),
            ]
            judge_count = len(base_judges)
            make_panel = lambda sessions: LivePanel(  # noqa: E731
                FRAME,
                judges=base_judges,
                quorum=int(PanelConfig.model_fields["quorum"].default),
                session_ids=sessions,
            )
        else:
            raise ValueError(
                f"panel_source must be 'yaml' or 'base', got {panel_source!r}"
            )
        judge_sessions = [
            f"demo-judge-{idx}-{uuid.uuid4().hex[:8]}"
            for idx in range(1, judge_count + 1)
        ]
        panel_obj = make_panel(judge_sessions)

        def _announce_judge(judge) -> None:
            line = (
                f"[zydah] judge {getattr(judge, 'name', judge)} "
                f"({getattr(judge, 'model', '?')}) ...Thinking"
            )
            if show_zydah:
                emit(line)
            if capture_zydah_view:
                logger.info("%s", line)

        panel_obj.on_judge_start = _announce_judge
        logger.info(
            "panel source=%s sessions: %s", panel_source, judge_sessions
        )
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
        panel=panel_obj,
        responder=_streaming_responder(emit),
        mass_elicitor=_thinking_elicitor(
            base_elicitor,
            emit if show_zydah else (lambda line: None),
            capture_zydah_view=capture_zydah_view,
        ),
        config=cfg,
        model_id=model_id,
        graph_resolver=graph_resolver,
        graph_factory=graph_factory,
        similarity_scorer=similarity_scorer,
        session_id=f"demo-verifier-{uuid.uuid4().hex[:8]}",
        log_path=_verifier_log_path(),
    )


def _panel_json_record(entry: Mapping[str, object]) -> Dict:
    """Compact per-judge record for one panel cycle (Streamlit-ready).

    Each judge: name, model, session id, its own mass (feedback) and
    pignistic (individual scoring); plus the fused interval summary.
    """
    judges = []
    masses = entry.get("judge_masses") or []
    metas = entry.get("judges") or []
    for idx, mass in enumerate(masses):  # type: ignore[union-attr]
        meta = metas[idx] if idx < len(metas) else {}  # type: ignore[union-attr]
        judges.append(
            {
                "name": meta.get("name") or f"judge-{idx + 1}",
                "model": meta.get("model"),
                "session_id": meta.get("session_id"),
                "mass": belief_frame(mass, FRAME)["mass"],  # type: ignore[arg-type]
                "pignistic": pignistic(mass, FRAME),  # type: ignore[arg-type]
            }
        )
    zydah_entry = entry.get("zydah") or {}
    return {
        "judges": judges,
        "fused": {
            key: zydah_entry.get(key)  # type: ignore[union-attr]
            for key in ("decision", "bel", "pl", "width", "conflict")
        },
    }


_MASS_KEYS = ("fail", "partial", "pass", "Omega")


def _panel_table_text(record: Mapping[str, object]) -> str:
    """Render one panel cycle as a Rich table + fused-belief panel.

    Rendered into a string (no color) so it flows through the same
    emit channel as the rest of the terminal view and lands in the
    saved transcript; the machine-readable panel_json stays in the
    INFO log at full precision.
    """
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False, no_color=True)
    table = Table(title="Judge panel", header_style="bold cyan")
    table.add_column("judge")
    table.add_column("model")
    for key in _MASS_KEYS:
        table.add_column(key, justify="right")
    for judge in record.get("judges") or []:  # type: ignore[union-attr]
        mass = judge.get("mass") or {}
        table.add_row(
            str(judge.get("name")),
            str(judge.get("model")),
            *(f"{float(mass.get(key, 0.0)):.2f}" for key in _MASS_KEYS),
        )
    console.print(table)
    fused = record.get("fused") or {}
    color = {"commit": "green", "escalate": "yellow"}.get(
        str(fused.get("decision")), "red"  # type: ignore[union-attr]
    )
    body = Text.assemble(
        ("decision: ", "bold"),
        (str(fused.get("decision")), f"bold {color}"),  # type: ignore[union-attr]
        f"\nBel {float(fused.get('bel') or 0):.3f}   "  # type: ignore[union-attr]
        f"Pl {float(fused.get('pl') or 0):.3f}   "  # type: ignore[union-attr]
        f"width {float(fused.get('width') or 0):.3f}   "  # type: ignore[union-attr]
        f"conflict {float(fused.get('conflict') or 0):.3f}",  # type: ignore[union-attr]
    )
    console.print(RichPanel(body, title="Fused belief", border_style=color))
    return buffer.getvalue().rstrip("\n")


def _log_panel_json(result: Mapping[str, object]) -> Optional[Dict]:
    """Log the latest panel cycle as one panel_json INFO record."""
    history = result.get("history") or []
    for entry in reversed(history):  # type: ignore[union-attr]
        if isinstance(entry, Mapping) and entry.get("judge_masses"):
            record = _panel_json_record(entry)
            logger.info("panel_json %s", json.dumps(record))
            return record
    return None


def _merge_usage(*parts) -> Dict[str, object]:
    """Sum usage dicts (calls/elapsed/tokens); missing parts are skipped."""
    merged: Dict[str, object] = {
        "calls": 0,
        "elapsed_s": 0.0,
        "tokens_in": None,
        "tokens_out": None,
    }
    for part in parts:
        if not isinstance(part, Mapping):
            continue
        merged["calls"] = int(merged["calls"]) + int(part.get("calls") or 0)  # type: ignore[arg-type]
        merged["elapsed_s"] = float(merged["elapsed_s"]) + float(part.get("elapsed_s") or 0)  # type: ignore[arg-type]
        for key in ("tokens_in", "tokens_out"):
            value = part.get(key)
            if isinstance(value, int):
                current = merged[key]
                merged[key] = (current if isinstance(current, int) else 0) + value
    return merged


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


def _frame_blocks(result: Mapping[str, object]):
    """(label, belief-frame dict) pairs for every mass in the history."""
    blocks = []
    history = result.get("history") or []
    for idx, entry in enumerate(history, start=1):  # type: ignore[union-attr]
        if isinstance(entry, Mapping) and "mass" in entry:
            score_turn = entry.get("score_turn")
            iteration = entry.get("iteration")
            if isinstance(score_turn, int) and isinstance(iteration, int):
                label = f"turn {score_turn} iteration {iteration} belief_frame"
            else:
                # Untagged entry (e.g. a resume cycle): never claim a
                # response turn we cannot prove.
                label = f"history {idx} belief_frame"
            blocks.append(
                (label, belief_frame(entry["mass"], FRAME))  # type: ignore[arg-type]
            )
    return blocks


def _belief_frame_table_text(label: str, frame: Mapping[str, object]) -> str:
    """Render one belief frame as a small Rich table (label printed separately)."""
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False, no_color=True)
    table = Table(header_style="bold cyan")
    table.add_column("label")
    table.add_column("mass", justify="right")
    mass = frame.get("mass") or {}
    for key in _MASS_KEYS:
        table.add_row(key, f"{float(mass.get(key, 0.0)):.3f}")  # type: ignore[union-attr]
    console.print(table)
    return buffer.getvalue().rstrip("\n")


def _rich_json_text(json_text: str) -> str:
    """Render one JSON blob with Rich (plain text when unparseable)."""
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False, no_color=True)
    try:
        console.print(RichJSON(json_text))
    except Exception:
        return json_text
    return buffer.getvalue().rstrip("\n")


def _rich_transcript_text(blob: str) -> str:
    """Format a follow-up transcript: prose as-is, JSON blocks via Rich.

    ``format_transcript`` interleaves ``[Prompt N]`` headers, the
    transmitted follow-up JSON, and ``[Response N]`` model text. The
    JSON blocks render with indentation/highlight structure; the
    surrounding prose stays untouched. Log records keep the raw blob.
    """
    decoder = json.JSONDecoder()
    pieces: List[str] = []
    index = 0
    while index < len(blob):
        start = blob.find("{", index)
        if start == -1:
            pieces.append(blob[index:])
            break
        pieces.append(blob[index:start])
        try:
            _obj, end = decoder.raw_decode(blob, start)
        except ValueError:
            pieces.append(blob[start : start + 1])
            index = start + 1
            continue
        pieces.append(_rich_json_text(blob[start:end]))
        index = end
    return "".join(pieces).rstrip("\n")


def _internal_view_lines(result: Mapping[str, object]) -> List[str]:
    """Zydah's internal view: transcript, per-turn frames, conditioning."""
    lines: List[str] = []
    history = result.get("history") or []
    transcript = format_transcript(history)  # type: ignore[arg-type]
    if transcript:
        lines.append(transcript)
    for label, frame in _frame_blocks(result):
        lines.append(f"{label}:")
        lines.append(json.dumps(frame, indent=2))
    conditioning = result.get("graph_conditioning")
    if conditioning:
        lines.append("graph_conditioning:")
        lines.append(json.dumps(conditioning, indent=2))
    return lines


def _log_internal_view(result: Mapping[str, object]) -> None:
    """Capture Zydah's internal view in the INFO log."""
    for line in _internal_view_lines(result):
        logger.info("%s", line)


def render_result(
    result: Mapping[str, object],
    emit: Emit,
    show_zydah: bool = False,
    capture_zydah_view: bool = False,
    turn: Optional[int] = None,
) -> None:
    """Render the verifier's current state.

    Default view is the concluded [zydah] sentence plus the verdict
    line. With ``show_zydah`` (--zydah), also show the internal view:
    the follow-up transcript and the belief frame as it adjusted on
    every turn, plus any graph conditioning. With
    ``capture_zydah_view`` (--verbose), that same internal view is
    captured through ``logger.info`` without being shown on the
    terminal unless ``show_zydah`` is also enabled.
    """
    tag = f"[zydah turn {turn}]" if turn is not None else "[zydah]"
    emit(f"{tag} {_final_response_line(result)}")
    verdict_line = (
        "verdict={verdict} needs_human={needs_human} followups={followups}".format(
            verdict=result.get("verdict"),
            needs_human=result.get("needs_human", False),
            followups=result.get("followups", 0),
        )
    )
    usage = result.get("usage")
    if isinstance(usage, Mapping) and usage.get("calls"):
        verdict_line += f" · {float(usage.get('elapsed_s') or 0):.2f}s"
        tokens_in, tokens_out = usage.get("tokens_in"), usage.get("tokens_out")
        if isinstance(tokens_in, int) and isinstance(tokens_out, int):
            verdict_line += f" · {tokens_in + tokens_out} tok"
    emit(verdict_line)
    if not show_zydah and not capture_zydah_view:
        return
    lines = _internal_view_lines(result)
    if capture_zydah_view:
        for line in lines:
            logger.info("%s", line)
    if not show_zydah:
        return
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("[Prompt "):
            emit(_rich_transcript_text(line))
            index += 1
            continue
        if line.lstrip().startswith("{"):
            emit(_rich_json_text(line))
            index += 1
            continue
        if line.endswith("belief_frame:") and index + 1 < len(lines):
            # Terminal: label line + Rich table. Log keeps plain JSON.
            emit(line)
            emit(
                _belief_frame_table_text(
                    line[:-1], json.loads(lines[index + 1])
                )
            )
            index += 2
            continue
        emit(line)
        index += 1


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
    capture_zydah_view: bool = False,
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
        render_result(
            current,
            emit,
            show_zydah=show_zydah,
            capture_zydah_view=capture_zydah_view,
        )
        _log_zydah_json("", "", current)
    return current


def artifact_for(question: str, response: str, result: Mapping[str, object]) -> Dict:
    """JSON artifact for one interactive run (ledger objects excluded)."""
    turns = []
    for idx, entry in enumerate(result.get("history") or [], start=1):  # type: ignore[union-attr]
        if not isinstance(entry, Mapping):
            continue
        turn = {"turn": idx}
        if "score_turn" in entry:
            turn["score_turn"] = entry["score_turn"]
        if "iteration" in entry:
            turn["iteration"] = entry["iteration"]
        if "mass" in entry:
            turn["belief_frame"] = belief_frame(entry["mass"], FRAME)  # type: ignore[arg-type]
        zydah_entry = entry.get("zydah")
        if isinstance(zydah_entry, Mapping):
            for key in ("bel", "pl", "width", "conflict", "decision"):
                if key in zydah_entry:
                    turn[key] = zydah_entry[key]
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
    if result.get("usage"):
        artifact["usage"] = result["usage"]
    if result.get("score_mode"):
        artifact["score_mode"] = result.get("score_mode")
    if result.get("unscored_reason"):
        artifact["unscored_reason"] = result.get("unscored_reason")
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


def _stream_model_text(
    prompt: str, model_id: str, emit: Emit, turn: Optional[int] = None
):
    """Stream one model response; let the human type while it lands.

    Async loop: a producer thread feeds response chunks into a queue
    while the main loop selects on the keyboard (TerminalBox). Text
    typed and sent during streaming is returned as the pending buffer
    (the human's next direction); without a terminal this degrades to
    plain streaming. Returns ``(response_text, pending_or_None)``.
    """
    emit("...Thinking")
    emit(f"[muse turn {turn}]" if turn is not None else "[muse]")
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
    score_mode: str = SCORE_MODE_EVERY,
    capture_zydah_view: bool = False,
    zydah_panel: bool = False,
    panel_source: str = "yaml",
):
    """Ask the live model, score responses as they land, and pause for input."""
    if score_mode not in SCORE_MODES:
        raise ValueError(
            f"score_mode must be one of {SCORE_MODES}, got {score_mode!r}"
        )
    cfg = cfg or load_verifier_config()
    model_id = _model_id(cfg)
    logger.info("score_mode=%s", score_mode)
    logger.info("question: %r", question)
    sequencer = TurnSequencer(emit)
    out: Emit = sequencer  # type: ignore[assignment]
    turn_counter = 0
    usage_by_turn: Dict[int, Dict[str, object]] = {}

    def _stream(prompt: str):
        """Stream one model response under the next turn number."""
        nonlocal turn_counter
        turn_counter += 1
        seq_before = int(last_call_stats().get("seq", 0))
        text, pending = _stream_model_text(
            prompt, model_id, out, turn=turn_counter
        )
        stats = last_call_stats()
        if int(stats.get("seq", 0)) > seq_before:
            usage_by_turn[turn_counter] = {
                "calls": int(stats["seq"]) - seq_before,
                "elapsed_s": float(stats.get("elapsed_s") or 0),
                "tokens_in": stats.get("tokens_in"),
                "tokens_out": stats.get("tokens_out"),
            }
        return text, pending, turn_counter

    response, pending_direction, response_turn = _stream(question)
    logger.info("model response:\n%s", response)
    logger.info(
        "asks_human=%s pending_direction=%r",
        _asks_human(response, cfg),
        pending_direction,
    )
    # Setup (panel judges, graph, scorer) takes time before the first
    # score lands; tell the user Zydah is launching rather than hanging.
    out("...launching zydah panel" if zydah_panel else "...launching zydah")
    verifier = build_verifier(
        question,
        response,
        model_id,
        emit=out,
        show_zydah=show_zydah,
        capture_zydah_view=capture_zydah_view,
        panel=zydah_panel,
        panel_source=panel_source,
    )
    context_lines = [f"Question: {question}", f"Response: {response}"]
    try:
        clarification_limit = int(cfg.max_iterations)
    except Exception:
        clarification_limit = 5  # config always defines this; test cfg may not
    clarifications = 0
    human_conditioning = None

    def _condition_on_text(text: str):
        """Run the relationship resolver on a human's free text.

        Human free text enters the same HumanFollowup JSON ->
        polarity gate (System 1 markers veto before MiniLM scores)
        -> candidate beliefs machinery as a structured Other
        choice. Returns the conditioning summary text, or None when
        nothing could be conditioned (no graph, no support).
        """
        nonlocal human_conditioning
        condition = getattr(verifier, "condition_on_human_text", None)
        if not callable(condition):
            return None
        try:
            found = condition(text, question=question)
        except Exception as exc:  # keep the demo terminal usable
            _log_error("human_conditioning", exc)
            return None
        if found is None:
            return None
        human_conditioning = found
        logger.info("human conditioning: %s", found.get("text"))
        context_lines.append(
            f"Human clue conditioning: {found.get('text')}"
        )
        return str(found.get("text") or "") or None

    checkpoint_history: List[object] = []
    checkpoint_followups = 0
    latest_result = None
    paused_result = None
    last_scored_signature = None
    last_scored_response = None
    rendered_result_id = None
    captured_result_id = None

    last_scored_turn: Optional[int] = None

    def _score_response(current_response: str, turn: Optional[int] = None):
        """Score one landed model response and log its belief frame."""
        nonlocal checkpoint_followups, latest_result, paused_result
        nonlocal last_scored_signature, last_scored_response, rendered_result_id
        nonlocal captured_result_id, last_scored_turn
        if (
            score_mode == SCORE_MODE_SUBSTANTIVE
            and not _is_substantive_response(current_response)
        ):
            logger.info(
                "zydah_unscored reason=pure_question mode=substantive"
            )
            return latest_result
        evidence = "\n".join(
            [*context_lines, f"Claim to evaluate: {current_response}"]
        )
        signature = (current_response, evidence)
        if signature == last_scored_signature:
            return latest_result
        logger.info("evidence scored:\n%s", evidence)
        first = capture_first_response(
            text=current_response,
            model_id=model_id,
            session_id=f"demo-model-{uuid.uuid4().hex[:8]}",
            playground="A",
        )
        if zydah_panel:
            # Panel path: 3 judges fuse per cycle (verify). No sealed
            # first-response independence bookkeeping in v1, and human
            # choices pause without resume (no resume_panel yet).
            scored = verifier.verify(
                evidence,
                assumptions=_assumptions_for(question, current_response),
            )
        else:
            scored = verifier.verify_direct(
                evidence,
                first_response=first,
                assumptions=_assumptions_for(question, current_response),
            )
        result = dict(scored)
        if turn is not None:
            # Tag each verifier iteration with its coordinates so the
            # rendering can pair frames with the [muse turn N] they
            # scored instead of numbering frames by history position.
            result["history"] = [
                (
                    {**entry, "score_turn": turn, "iteration": idx}
                    if isinstance(entry, Mapping)
                    else entry
                )
                for idx, entry in enumerate(
                    result.get("history") or [], start=1  # type: ignore[union-attr]
                )
            ]
        if human_conditioning is not None:
            result["graph_conditioning"] = human_conditioning
        # Per-turn budget: this turn's streamed response plus the
        # verifier's own calls (elicitor + follow-up responder).
        result["usage"] = _merge_usage(
            usage_by_turn.pop(turn, None) if turn is not None else None,
            result.get("usage"),
        )
        checkpoint_history.extend(result.get("history") or [])
        checkpoint_followups += int(result.get("followups", 0) or 0)
        result["history"] = list(checkpoint_history)
        result["followups"] = checkpoint_followups
        result["score_mode"] = score_mode
        latest_result = result
        last_scored_signature = signature
        last_scored_response = current_response
        last_scored_turn = turn
        _log_zydah_json(question, current_response, latest_result)
        panel_record = _log_panel_json(latest_result) if zydah_panel else None
        if show_zydah:
            render_result(
                latest_result,
                out,
                show_zydah=True,
                capture_zydah_view=capture_zydah_view,
                turn=turn,
            )
            rendered_result_id = id(latest_result)
            if panel_record is not None:
                out(_panel_table_text(panel_record))
            if capture_zydah_view:
                captured_result_id = id(latest_result)
        elif capture_zydah_view:
            _log_internal_view(latest_result)
            captured_result_id = id(latest_result)
        if latest_result.get("needs_human") and latest_result.get("choices"):
            paused_result = latest_result
        return latest_result

    def _quit_result(paused):
        """Keep an existing checkpoint when the human quits mid-question."""
        if latest_result is not None:
            quit_result = dict(latest_result)
            quit_result["quit"] = True
            return quit_result
        logger.info(
            "zydah_unscored reason=no_checkpoint mode=%s", score_mode
        )
        return paused

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
            paused_result is None
            and _asks_human(current_response, cfg)
            and clarifications < clarification_limit
        ):
            clarifications += 1
            # Tagged with the upcoming turn: the sequencer holds this
            # line until the [muse turn N] header has printed.
            out(
                f"[zydah turn {turn_counter + 1}] the model needs one "
                "detail before I can conclude."
            )
            out("[human] the model asked you a question; answer it here.")
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
            conditioning_text = _condition_on_text(human_answer)
            followup_prompt = (
                f"Original question: {question}\n"
                f"You asked for clarification: {current_response}\n"
                f"Human answer: {human_answer}\n"
                "Please now answer the original question using that answer."
            )
            if conditioning_text:
                followup_prompt += (
                    "\nGraph conditioning from the human's clue: "
                    f"{conditioning_text}"
                )
            current_response, pending_next, current_turn = _stream(
                followup_prompt
            )
            logger.info("model response (continuation):\n%s", current_response)
            logger.info("asks_human=%s", _asks_human(current_response, cfg))
            if pending_next is not None:
                pending_direction = pending_next
            context_lines.append(f"Response: {current_response}")
            _score_response(current_response, turn=current_turn)
        return current_response, None

    _score_response(response, turn=response_turn)
    paused = None
    if paused_result is None:
        response, paused = _clarify(response)
        if paused is not None:
            sequencer.flush_pending()
            return _quit_result(paused), verifier, response
    # The human directs from the input buffer: typed text is sent as
    # a direction on Enter; an empty buffer releases Zydah to finish
    # evaluating the current response in its accumulated context.
    # Buffered text ending in '?' is not a direction, though: it is
    # a new question for the model. Route it (recorded as a Human
    # question), score its answer, and gate again on that answer.
    question_rounds = 0
    while paused_result is None:
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
            sequencer.flush_pending()
            return _quit_result(paused), verifier, response
        if not direction.strip():
            break
        if not _ends_with_question(direction.strip()):
            context_lines.append(f"Human direction: {direction.strip()}")
            _condition_on_text(direction.strip())
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
        response, pending_next, gate_turn = _stream(followup_prompt)
        logger.info("model response (follow-up):\n%s", response)
        if pending_next is not None:
            pending_direction = pending_next
        context_lines.append(f"Response: {response}")
        _score_response(response, turn=gate_turn)
        if paused_result is None:
            response, paused = _clarify(response)
            if paused is not None:
                sequencer.flush_pending()
                return _quit_result(paused), verifier, response
    result = paused_result or _score_response(response, turn=turn_counter)
    if result is None or last_scored_response != response:
        logger.info(
            "zydah_unscored reason=no_substantive_response mode=%s",
            score_mode,
        )
        result = {
            "verdict": False,
            "needs_human": _asks_human(response, cfg),
            "followups": checkpoint_followups,
            "history": list(checkpoint_history),
            "score_mode": score_mode,
            "unscored_reason": "pure_question",
        }
        if human_conditioning is not None:
            result["graph_conditioning"] = human_conditioning
    if id(result) != rendered_result_id:
        render_result(
            result,
            out,
            show_zydah=show_zydah,
            capture_zydah_view=(
                capture_zydah_view and id(result) != captured_result_id
            ),
            turn=(
                turn_counter
                if result.get("unscored_reason")
                else last_scored_turn
            ),
        )
        if capture_zydah_view:
            captured_result_id = id(result)
    if zydah_panel:
        # No resume_panel in v1: a needs_human pause stays paused.
        final = result
    else:
        final = interact(
            result,
            verifier,
            ask=ask,
            emit=out,
            show_zydah=show_zydah,
            capture_zydah_view=capture_zydah_view,
        )
    sequencer.flush_pending()
    return final, verifier, response


def _append_manifest(save_path: Path, result: Mapping[str, object]) -> None:
    """Append one audit entry for a saved artifact (best effort).

    The manifest is the rerun record for outputs/: which artifact,
    its hash, the git state and wall-clock time that produced it,
    the scoring conditions (thresholds/model from YAML), and the
    verdict summary. Never raises; failures go to the error log.
    """
    try:
        data = save_path.read_bytes()
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
                cwd=str(REPO_ROOT),
            ).stdout.strip()
            branch = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True, text=True, timeout=5,
                cwd=str(REPO_ROOT),
            ).stdout.strip()
        except Exception:
            commit, branch = "", ""
        cfg = load_verifier_config()
        entry = {
            "artifact": str(save_path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "git_commit": commit,
            "branch": branch,
            "run_date": datetime.now(timezone.utc).isoformat(),
            "score_mode": result.get("score_mode"),
            "model_id": result.get("model_id"),
            "config": {
                "tau": cfg.get("tau"),
                "width_threshold": cfg.get("width_threshold"),
                "conflict_threshold": cfg.get("conflict_threshold"),
                "max_iterations": cfg.get("max_iterations"),
            },
            "verdict": result.get("verdict"),
            "needs_human": result.get("needs_human", False),
            "followups": result.get("followups", 0),
            "backfilled": False,
        }
        manifest = save_path.parent / "manifest.jsonl"
        with open(manifest, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        logger.info("manifest entry appended: %s", manifest)
    except Exception as exc:  # manifest must never break the demo
        _log_error("manifest", exc)


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
        if save_path is not None:
            _append_manifest(Path(save_path), result)
        emit(f"saved: {save_path}")
    emit(RESOLVED_INVITE)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive Zydah live demo")
    parser.add_argument("--question", help="ask one question, then still pause for any human choices")
    parser.add_argument("--env", default=str(REPO_ROOT / ".env"), help="env file to load")
    parser.add_argument("--save", type=Path, help="write the latest run artifact JSON here")
    parser.add_argument("--zydah", action="store_true", help="show Zydah's internal view: thinking semaphores, follow-up transcript, per-turn belief frames")
    parser.add_argument("--verbose", action="store_true", help="log input, model text, routing decisions, scored evidence, and Zydah's internal view to stderr (logging INFO)")
    parser.add_argument("--log-file", type=Path, help="write INFO logs to this file via a FileHandler (no stderr redirect needed)")
    parser.add_argument("--score-mode", choices=SCORE_MODES, default=SCORE_MODE_EVERY, help="score every model response (default) or only substantive responses")
    parser.add_argument("--zydah-panel", action="store_true", help="score with the 3-judge panel (quorum 2 of 3) instead of the single-elicitor direct path")
    parser.add_argument("--panel-source", choices=["yaml", "base"], default="yaml", help="where panel judges come from: conf/panel.yaml (default) or the hardcoded base models (Wonder/Spark/Spark Open)")
    args = parser.parse_args(argv)
    if args.save is not None:
        args.save = Path(args.save)
        if args.save.suffix != ".json":
            # A wrapped paste can split the path (outputs/live_ +
            # demo/latest.json); fail fast instead of saving to a
            # plausible-looking wrong file.
            parser.error(
                f"--save must name a .json file, got {args.save} "
                "(quote the path if the command wrapped)"
            )

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
    logger.info("score_mode=%s", args.score_mode)

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
                    question,
                    cfg=cfg,
                    emit=emit,
                    show_zydah=args.zydah,
                    score_mode=args.score_mode,
                    capture_zydah_view=args.verbose,
                    zydah_panel=args.zydah_panel,
                    panel_source=args.panel_source,
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



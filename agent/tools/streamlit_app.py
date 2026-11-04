"""Streamlit chat UI for the Zydah live demo.

Run from the repository root (``agent/``):

    uv run streamlit run tools/streamlit_app.py
    uv run streamlit run tools/streamlit_app.py -- --zydah-panel
    uv run streamlit run tools/streamlit_app.py -- --no-zydah-panel

Script args after ``--`` land in ``sys.argv``. Every flag below maps
to a sidebar control initialized from the arg, so CLI and UI stay in
sync (the sidebar wins interactively):

    --zydah-panel / --no-zydah-panel  panel visibility (default on)
    --panel-scoring                   3-judge panel verify() instead
                                      of the direct verify_direct()
    --panel-source yaml|base          judges from conf/panel.yaml
                                      (default) or the hardcoded
                                      Wonder/Spark/Spark Open models
    --panel-model <model>             override every judge seat's
                                      model (implies --panel-scoring)
    --score-mode every|substantive    score every response, or only
                                      substantive ones (demo semantics)
    --show-frames                     belief-frame JSON + follow-up
                                      transcript (the --zydah view)
    --show-usage                      per-turn seconds/tokens
    --mock-artifact <path>            replay a saved artifact instead
                                      of live calls (forces mock mode)

Streamlit only: no FastAPI, no second process. Local, single user.

Pipeline reuse: this app does not reimplement fusion. Each user prompt
streams one model response (``[muse turn N]``) through
``zydah.transport.stream_chat_complete``, seals it with
``capture_first_response``, and scores it with
``ZydahVerifier.verify_direct`` — the same non-panel path the terminal
demo (``tools/live_demo.py``) drives. ``run_question`` itself is not
reused: it blocks on terminal input (direction gate, ``interact``),
which cannot survive Streamlit's rerun-per-interaction model. The
verifier iterations inside one ``verify_direct`` call are rendered as
``turn N iteration i`` belief frames; they are not new turns.

Turn / frame / usage mapping: a scored exchange keeps the verifier
``history`` entries in order. Each entry contributes one turn record:
``score_turn`` is the ``[muse turn N]`` number it scored and
``iteration`` its 1-based position inside that score (mirroring the
tagging in ``live_demo._score_response``). The belief frame comes from
``zydah.fusion.belief_frame(entry["mass"], FRAME)`` (pass / partial /
fail / Omega masses); Bel, Pl, width, conflict and decision come from
the entry's ``zydah`` fused verdict. Per-turn ``usage`` is
``{calls, elapsed_s, tokens_in, tokens_out}``: the first-response call
(diffed via ``last_call_stats``) merged with the verifier's own usage
through ``live_demo._merge_usage``, shown next to ``[zydah turn N]``.

Mock mode: when no model endpoint is configured (the env vars named in
``conf/transport.yaml`` are unset, after loading the repo ``.env``), or
when ``ZYDAH_STREAMLIT_MOCK=1``, the app renders
``outputs/live_demo/latest.json`` (falling back to ``cycles.jsonl``)
and chat submissions replay that artifact instead of calling a model,
so the UI can be smoke tested without a live call. Artifact turns that
lack Bel/Pl are re-derived with ``fused_verdict`` on the reconstructed
mass — the existing deterministic-fusion code, not new math.

Secrets come only from the repo ``.env`` / environment and are never
printed; the sidebar shows env var *names* with set/unset marks only.
All thresholds (width, conflict, tau, iteration budget) are read from
``src/zydah/conf/verifier.yaml`` via ``load_verifier_config``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
TOOLS_DIR = REPO_ROOT / "tools"
ARTIFACT_DIR = REPO_ROOT / "outputs" / "live_demo"
DEFAULT_ARTIFACT = ARTIFACT_DIR / "latest.json"
CYCLES_ARTIFACT = ARTIFACT_DIR / "cycles.jsonl"
MOCK_ENV_VAR = "ZYDAH_STREAMLIT_MOCK"

for _path in (str(SRC_DIR), str(TOOLS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _load_live_demo():
    """Import tools/live_demo.py the way the test-suite does."""
    spec = importlib.util.spec_from_file_location(
        "live_demo", TOOLS_DIR / "live_demo.py"
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("live_demo", module)
    spec.loader.exec_module(module)
    return module


try:
    live_demo = _load_live_demo()
except Exception:  # mock rendering must survive a broken live import
    live_demo = None

FRAME = (
    live_demo.FRAME
    if live_demo is not None
    else frozenset({"pass", "partial", "fail"})
)

from dotenv import load_dotenv  # noqa: E402

_ENV_PATH = REPO_ROOT / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)

import streamlit as st  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from zydah.fusion import belief_frame  # noqa: E402
from zydah.interface import capture_first_response, fused_verdict  # noqa: E402
from zydah.transport import (  # noqa: E402
    last_call_stats,
    load_transport_config,
    stream_chat_complete,
)
from zydah.verifier import (  # noqa: E402
    ZydahVerifier,
    load_verifier_config,
    make_mass_elicitor,
    make_model_responder,
)

LABEL_COLORS = {
    "pass": "#2e7d32",
    "partial": "#e6a817",
    "fail": "#c62828",
    "Omega": "#9e9e9e",
    "empty": "#616161",
}


@st.cache_resource(show_spinner=False)
def _verifier_cfg():
    return load_verifier_config()


def _cfg_value(cfg, key: str):
    value = OmegaConf.select(cfg, key)
    if value is None:
        raise ValueError(f"verifier config is missing required key {key!r}")
    return value


def _endpoint_status() -> Dict[str, bool]:
    """Env var names from transport.yaml mapped to set/unset (no values)."""
    cfg = load_transport_config()
    names = [
        str(OmegaConf.select(cfg, key))
        for key in ("base_url_env", "api_key_env")
    ]
    return {name: bool(os.environ.get(name)) for name in names}


def _mock_mode() -> bool:
    forced = os.environ.get(MOCK_ENV_VAR, "").strip().lower()
    if forced in {"1", "true", "yes"}:
        return True
    return not all(_endpoint_status().values())


def _cli_options() -> argparse.Namespace:
    """Script args after ``--``; each maps to a sidebar control.

    Values are normalized manually (no argparse ``choices``) so a bad
    flag value degrades to the default instead of killing the app.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--zydah-panel", action="store_true")
    parser.add_argument("--no-zydah-panel", action="store_true")
    parser.add_argument("--panel-scoring", action="store_true")
    parser.add_argument("--panel-source", type=str, default="yaml")
    parser.add_argument("--panel-model", type=str, default=None)
    parser.add_argument("--score-mode", type=str, default="every")
    parser.add_argument("--show-frames", action="store_true")
    parser.add_argument("--show-usage", action="store_true")
    parser.add_argument("--mock-artifact", type=str, default=None)
    args, _unknown = parser.parse_known_args(sys.argv[1:])
    args.panel_visible = not args.no_zydah_panel  # default on
    args.panel_source = (
        args.panel_source if args.panel_source in ("yaml", "base") else "yaml"
    )
    args.score_mode = (
        args.score_mode
        if args.score_mode in ("every", "substantive")
        else "every"
    )
    args.panel_model = (
        args.panel_model.strip()
        if isinstance(args.panel_model, str) and args.panel_model.strip()
        else None
    )
    if args.panel_model:
        args.panel_scoring = True
    args.mock_path: Optional[Path] = None
    if args.mock_artifact:
        candidate = Path(args.mock_artifact)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        args.mock_path = candidate
    return args


def _model_id(cfg) -> str:
    if live_demo is not None:
        return live_demo._model_id(cfg)
    env_var = str(_cfg_value(cfg, "model.env_var"))
    return os.environ.get(env_var) or str(_cfg_value(cfg, "model.id"))


def _mass_from_belief_frame(frame_dict: Mapping) -> tuple:
    """Rebuild a Mass from a serialized belief_frame dict."""
    frame = frozenset(frame_dict.get("frame") or FRAME)
    mass: Dict = {}
    for key, value in (frame_dict.get("mass") or {}).items():
        if key == "Omega":
            focal = frame
        elif key == "empty":
            continue  # unnormalized residue; not a stored belief
        else:
            focal = frozenset(
                part.strip() for part in str(key).split(",") if part.strip()
            )
        if focal:
            mass[focal] = mass.get(focal, 0.0) + float(value)
    return mass, frame


def _derive_interval(frame_dict: Mapping, cfg) -> Dict:
    """Bel/Pl/width/conflict/decision for a frame via fused_verdict."""
    mass, frame = _mass_from_belief_frame(frame_dict)
    if not mass:
        return {}
    verdict = fused_verdict(
        [mass],
        frame,
        tau=float(_cfg_value(cfg, "tau")),
        rule=str(_cfg_value(cfg, "rule")),
    )
    return {
        "bel": verdict.get("bel"),
        "pl": verdict.get("pl"),
        "width": verdict.get("width"),
        "conflict": verdict.get("conflict"),
        "decision": verdict.get("decision"),
        "label": verdict.get("verdict"),
    }


def _turn_from_artifact(raw: Mapping, score_turn: int, cfg) -> Dict:
    """Normalize one artifact turn (belief_frame, optional Bel/Pl)."""
    frame_dict = raw.get("belief_frame")
    turn: Dict = {
        "score_turn": raw.get("score_turn", score_turn),
        "iteration": raw.get("iteration", raw.get("turn")),
        "belief_frame": frame_dict,
        "response": raw.get("response"),
        "followup": raw.get("followup"),
    }
    if raw.get("bel") is not None:
        turn.update(
            {
                "bel": float(raw["bel"]),
                "pl": float(raw["pl"]),
                "width": float(raw["width"]),
                "conflict": float(raw.get("conflict", 0.0)),
                "decision": raw.get("decision"),
                "label": None,
            }
        )
    elif isinstance(frame_dict, Mapping):
        turn.update(_derive_interval(frame_dict, cfg))
    return turn


def _turn_from_history(entry: Mapping, score_turn: int, iteration: int) -> Dict:
    """Normalize one verify_direct history entry (mass + zydah verdict)."""
    zydah = entry.get("zydah") or {}
    turn: Dict = {
        "score_turn": score_turn,
        "iteration": iteration,
        "belief_frame": (
            belief_frame(entry["mass"], FRAME)  # type: ignore[arg-type]
            if "mass" in entry
            else None
        ),
        "bel": zydah.get("bel"),
        "pl": zydah.get("pl"),
        "width": zydah.get("width"),
        "conflict": zydah.get("conflict"),
        "decision": zydah.get("decision"),
        "label": zydah.get("verdict"),
        "response": entry.get("response"),
        "followup": entry.get("followup"),
        "prompt": entry.get("prompt"),
        "route": entry.get("route"),
        "judges": entry.get("judges"),
        "judge_frames": (
            [
                belief_frame(judge_mass, FRAME)
                for judge_mass in entry["judge_masses"]  # type: ignore[union-attr]
            ]
            if entry.get("judge_masses")
            else None
        ),
    }
    return turn


def _exchange_from_result(
    question: str,
    muse_text: str,
    muse_turn: int,
    result: Mapping,
    usage: Optional[Mapping],
    source: str,
    usage_stages: Optional[List[Dict]] = None,
) -> Dict:
    """Build one chat exchange from a verify_direct-shaped result."""
    turns = [
        _turn_from_history(entry, muse_turn, idx)
        for idx, entry in enumerate(result.get("history") or [], start=1)  # type: ignore[union-attr]
        if isinstance(entry, Mapping)
    ]
    return {
        "question": question,
        "muse_turn": muse_turn,
        "muse_text": muse_text,
        "verdict": result.get("verdict"),
        "needs_human": bool(result.get("needs_human", False)),
        "human_question": result.get("human_question"),
        "choices": result.get("choices") or [],
        "followups": result.get("followups", 0),
        "usage": dict(usage) if isinstance(usage, Mapping) else None,
        "usage_stages": usage_stages,
        "turns": turns,
        "source": source,
        "model_id": result.get("model_id"),
        "graph_conditioning": result.get("graph_conditioning"),
    }


def _conclusion(exchange: Mapping) -> str:
    """One-sentence conclusion, mirroring live_demo's final line logic."""
    if exchange.get("unscored_reason"):
        return "not scored — no substantive claim (substantive mode)."
    conditioning = exchange.get("graph_conditioning")
    if isinstance(conditioning, Mapping):
        support = conditioning.get("support") or []
        if support:
            return f"you are asking about {', '.join(map(str, support))}."
    if exchange.get("needs_human"):
        return "I still need one detail before I can conclude."
    if exchange.get("verdict") == "good":
        turns = exchange.get("turns") or []
        if turns:
            label = turns[-1].get("label")
            if label == "fail":
                return "the response does not check out."
            if label == "partial":
                return "the response is only partly supported."
        return "the response checks out."
    return "the response is still not dependable."


def _render_text_or_json(text: str) -> None:
    """Render full text; pretty-print (jq-style) when it is pure JSON.

    A bare ``st.markdown`` shows a one-line JSON body as a single
    unwrapped string that reads as cut off; ``st.json`` renders the
    same bytes indented and expandable, with nothing dropped.
    """
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        try:
            parsed = json.loads(stripped)
        except ValueError:
            parsed = None
        if isinstance(parsed, (dict, list)):
            st.json(parsed)
            return
    st.markdown(text)


def _render_json_value(value: object) -> None:
    """Pretty-print a stored JSON value (dict, or JSON string)."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            st.code(value, language="json")
            return
    st.json(value)  # type: ignore[arg-type]


def _usage_text(usage: Optional[Mapping]) -> str:
    if not isinstance(usage, Mapping) or not usage.get("calls"):
        return "usage not recorded"
    parts = [f"{float(usage.get('elapsed_s') or 0):.2f}s"]
    parts.append(f"{int(usage.get('calls') or 0)} calls")
    tokens_in, tokens_out = usage.get("tokens_in"), usage.get("tokens_out")
    if isinstance(tokens_in, int) and isinstance(tokens_out, int):
        parts.append(f"{tokens_in + tokens_out} tok ({tokens_in} in / {tokens_out} out)")
    return " · ".join(parts)


def _load_artifact_exchange(path: Path, muse_turn: int, cfg) -> Optional[Dict]:
    """Load one saved artifact (latest.json shape) as a chat exchange."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, Mapping):
        return None
    turns = [
        _turn_from_artifact(raw, muse_turn, cfg)
        for raw in (data.get("turns") or [])  # type: ignore[union-attr]
        if isinstance(raw, Mapping)
    ]
    top_frame = data.get("belief_frame")
    if not turns and isinstance(top_frame, Mapping):
        turn = {"score_turn": muse_turn, "iteration": 1, "belief_frame": top_frame}
        turn.update(_derive_interval(top_frame, cfg))
        turns = [turn]
    return {
        "question": data.get("question") or data.get("exchange") or "(artifact)",
        "muse_turn": muse_turn,
        "muse_text": data.get("response") or data.get("answer") or "",
        "verdict": data.get("verdict"),
        "needs_human": bool(data.get("needs_human", False)),
        "human_question": data.get("human_question"),
        "choices": data.get("choices") or [],
        "followups": data.get("followups", 0),
        "usage": data.get("usage"),
        "turns": turns,
        "source": "mock",
        "model_id": data.get("model_id"),
        "graph_conditioning": data.get("graph_conditioning"),
        "artifact": path.name,
    }


def _load_cycles_exchange(muse_turn: int) -> Optional[Dict]:
    """Fallback replay from cycles.jsonl (metrics only, no frames)."""
    if not CYCLES_ARTIFACT.exists():
        return None
    turns: List[Dict] = []
    try:
        lines = CYCLES_ARTIFACT.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        if record.get("event") != "verify_direct":
            continue
        width = record.get("width")
        turns.append(
            {
                "score_turn": muse_turn,
                "iteration": len(turns) + 1,
                "belief_frame": None,
                "bel": None,
                "pl": None,
                "width": float(width) if width is not None else None,
                "conflict": record.get("conflict"),
                "decision": None,
                "label": None,
                "response": None,
                "followup": None,
                "cycle_verdict": record.get("verdict"),
            }
        )
    if not turns:
        return None
    return {
        "question": "(cycles replay)",
        "muse_turn": muse_turn,
        "muse_text": "",
        "verdict": turns[-1].get("cycle_verdict"),
        "needs_human": False,
        "human_question": None,
        "choices": [],
        "followups": 0,
        "usage": None,
        "turns": turns,
        "source": "mock",
        "model_id": None,
        "graph_conditioning": None,
        "artifact": CYCLES_ARTIFACT.name,
    }


def _default_mock_exchange(muse_turn: int, cfg) -> Optional[Dict]:
    if DEFAULT_ARTIFACT.exists():
        return _load_artifact_exchange(DEFAULT_ARTIFACT, muse_turn, cfg)
    return _load_cycles_exchange(muse_turn)


def _load_mock_exchange(
    path: Optional[Path], muse_turn: int, cfg
) -> Optional[Dict]:
    """Load any mock source: artifact JSON or cycles JSONL."""
    if path is None:
        return _default_mock_exchange(muse_turn, cfg)
    if path.suffix == ".jsonl":
        return _load_cycles_exchange(muse_turn)
    exchange = _load_artifact_exchange(path, muse_turn, cfg)
    if exchange is not None:
        return exchange
    return _default_mock_exchange(muse_turn, cfg)


def _build_panel(panel_source: str, panel_model: Optional[str] = None):
    """3-judge LivePanel from panel.yaml or the hardcoded base models.

    ``panel_model`` overrides the model in every judge seat for this
    run, preserving judge names and configured effort.
    """
    from zydah.judges import PANEL, LivePanel, MetaApiJudge, load_panel_config

    if panel_source == "base":
        names = ("wonder", "spark", "spark-open")
        return LivePanel(
            FRAME,
            judges=[
                MetaApiJudge(panel_model or model, name=name)
                for model, name in zip(PANEL, names)
            ],
            quorum=load_panel_config().quorum,
        )
    config = load_panel_config()
    if panel_model:
        config = config.model_copy(
            update={
                "judges": [
                    judge.model_copy(update={"model": panel_model})
                    for judge in config.judges
                ]
            }
        )
    return LivePanel.from_config(config, FRAME)


def _build_verifier(
    question: str,
    response: str,
    model_id: str,
    panel_scoring: bool = False,
    panel_source: str = "yaml",
    panel_model: Optional[str] = None,
) -> ZydahVerifier:
    """Live verifier wiring, mirroring live_demo.build_verifier.

    Difference: the follow-up responder is the plain (non-streaming)
    ``make_model_responder``; its answers are recovered from the
    verifier history ``response`` fields, so nothing writes to stdout.
    With ``panel_scoring``, an explicit judge panel is wired for
    ``verify()`` instead of the direct single-elicitor path.
    """
    cfg = _verifier_cfg()
    combined = f"{question}\n{response}".lower()
    graph_resolver = None
    graph_factory = None
    similarity_scorer = None
    if "springfield" in combined:
        from zydah.graph import build_place_graph
        from zydah.graph_resolver import NumpyResolver
        from zydah.similarity import SentenceTransformerScorer

        graph_resolver = NumpyResolver()
        graph_factory = lambda evidence: build_place_graph()  # noqa: E731
        similarity_scorer = SentenceTransformerScorer(
            str(_cfg_value(cfg, "similarity_model"))
        )
    return ZydahVerifier(
        FRAME,
        panel=_build_panel(panel_source, panel_model) if panel_scoring else None,
        responder=make_model_responder(),
        mass_elicitor=make_mass_elicitor(
            str(_cfg_value(cfg, "direct_mass_prompt_template"))
        ),
        config=cfg,
        model_id=model_id,
        graph_resolver=graph_resolver,
        graph_factory=graph_factory,
        similarity_scorer=similarity_scorer,
        session_id=f"demo-verifier-{uuid.uuid4().hex[:8]}",
        log_path=ARTIFACT_DIR / "cycles.jsonl",
    )


def _unscored_exchange(
    question: str,
    muse_text: str,
    muse_turn: int,
    model_id: str,
    first_usage: Optional[Mapping],
) -> Dict:
    """Substantive-mode skip: a pure question is routed, not scored."""
    stages = None
    if isinstance(first_usage, Mapping):
        stages = [
            {
                "stage": "[muse] response",
                "model": model_id,
                "calls": int(first_usage.get("calls") or 0),
                "seconds": float(first_usage.get("elapsed_s") or 0),
                "tokens in": first_usage.get("tokens_in"),
                "tokens out": first_usage.get("tokens_out"),
            }
        ]
    return {
        "question": question,
        "muse_turn": muse_turn,
        "muse_text": muse_text,
        "verdict": None,
        "needs_human": False,
        "human_question": None,
        "choices": [],
        "followups": 0,
        "usage": dict(first_usage) if isinstance(first_usage, Mapping) else None,
        "usage_stages": stages,
        "turns": [],
        "source": "live",
        "model_id": model_id,
        "graph_conditioning": None,
        "unscored_reason": "pure_question",
    }


def _run_live(
    question: str, muse_turn: int, muse_slot, settings: Mapping
) -> Dict:
    """Stream [muse turn N], then score it (direct or panel path)."""
    cfg = _verifier_cfg()
    model_id = _model_id(cfg)
    seq_before = int(last_call_stats().get("seq", 0))
    parts: List[str] = []
    for chunk in stream_chat_complete(question, model_id=model_id):
        parts.append(chunk)
        muse_slot.markdown("".join(parts))
    muse_text = "".join(parts)
    if not muse_text.strip():
        raise RuntimeError("model returned an empty response")
    stats = last_call_stats()
    first_usage = None
    if int(stats.get("seq", 0)) > seq_before:
        first_usage = {
            "calls": int(stats["seq"]) - seq_before,
            "elapsed_s": float(stats.get("elapsed_s") or 0),
            "tokens_in": stats.get("tokens_in"),
            "tokens_out": stats.get("tokens_out"),
        }
    if (
        settings.get("score_mode") == "substantive"
        and live_demo is not None
        and not live_demo._is_substantive_response(muse_text)
    ):
        return _unscored_exchange(
            question, muse_text, muse_turn, model_id, first_usage
        )
    panel_scoring = bool(settings.get("panel_scoring"))
    verifier = _build_verifier(
        question,
        muse_text,
        model_id,
        panel_scoring=panel_scoring,
        panel_source=str(settings.get("panel_source", "yaml")),
        panel_model=(
            str(settings["panel_model"])
            if settings.get("panel_model")
            else None
        ),
    )
    first = capture_first_response(
        text=muse_text,
        model_id=model_id,
        session_id=f"demo-model-{uuid.uuid4().hex[:8]}",
        playground="A",
    )
    evidence = (
        f"Question: {question}\n"
        f"Response: {muse_text}\n"
        f"Claim to evaluate: {muse_text}"
    )
    assumptions = (
        live_demo._assumptions_for(question, muse_text)  # type: ignore[union-attr]
        if live_demo is not None
        else []
    )
    with st.spinner("[zydah] scoring the response…"):
        if panel_scoring:
            result = verifier.verify(evidence, assumptions=assumptions)
        else:
            result = verifier.verify_direct(
                evidence, first_response=first, assumptions=assumptions
            )
    if live_demo is not None:
        usage = live_demo._merge_usage(first_usage, result.get("usage"))
    else:
        usage = result.get("usage") or first_usage
    usage_stages: List[Dict] = []
    if isinstance(first_usage, Mapping):
        usage_stages.append(
            {
                "stage": "[muse] response",
                "model": model_id,
                "calls": int(first_usage.get("calls") or 0),
                "seconds": float(first_usage.get("elapsed_s") or 0),
                "tokens in": first_usage.get("tokens_in"),
                "tokens out": first_usage.get("tokens_out"),
            }
        )
    zydah_usage = result.get("usage")
    if isinstance(zydah_usage, Mapping) and zydah_usage.get("calls"):
        usage_stages.append(
            {
                "stage": "[zydah] scoring",
                "model": "3-judge panel" if panel_scoring else model_id,
                "calls": int(zydah_usage.get("calls") or 0),
                "seconds": float(zydah_usage.get("elapsed_s") or 0),
                "tokens in": zydah_usage.get("tokens_in"),
                "tokens out": zydah_usage.get("tokens_out"),
            }
        )
    exchange = _exchange_from_result(
        question,
        muse_text,
        muse_turn,
        result,
        usage,
        source="live",
        usage_stages=usage_stages or None,
    )
    if exchange["needs_human"] and not panel_scoring:
        # resume_direct is a direct-path API (ledger-based); panel
        # pauses surface their question without a resume affordance.
        st.session_state["pending"] = {
            "exchange_idx": len(st.session_state["exchanges"]),
            "result": result,
            "verifier": verifier,
        }
    return exchange


def _all_turns(exchanges: List[Mapping]) -> List[Dict]:
    """Flat scored turns across exchanges, in scoring order."""
    flat: List[Dict] = []
    for exchange in exchanges:
        for turn in exchange.get("turns") or []:
            flat.append(turn)
    return flat


def _turn_label(turn: Mapping) -> str:
    return f"turn {turn.get('score_turn')} iteration {turn.get('iteration')}"


def _frame_figure(frame_dict: Mapping):
    import plotly.graph_objects as go

    mass = frame_dict.get("mass") or {}
    order = [key for key in ("pass", "partial", "fail", "Omega") if key in mass]
    order += sorted(key for key in mass if key not in order)
    fig = go.Figure()
    for key in order:
        fig.add_trace(
            go.Bar(
                x=[float(mass[key])],
                y=["mass"],
                name=key,
                orientation="h",
                marker_color=LABEL_COLORS.get(key, "#607d8b"),
                hovertemplate=f"{key}: %{{x:.3f}}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(showticklabels=False),
        height=150,
        margin=dict(l=10, r=10, t=40, b=20),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
        ),
        showlegend=True,
    )
    return fig


def _interval_figure(turns: List[Mapping], cfg):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    scored = [
        turn
        for turn in turns
        if turn.get("bel") is not None and turn.get("pl") is not None
    ]
    if not scored:
        return None
    labels = [_turn_label(turn) for turn in scored]
    bel = [float(turn["bel"]) for turn in scored]  # type: ignore[arg-type]
    pl = [float(turn["pl"]) for turn in scored]  # type: ignore[arg-type]
    width = [p - b for b, p in zip(bel, pl)]
    tau = float(_cfg_value(cfg, "tau"))
    width_threshold = float(_cfg_value(cfg, "width_threshold"))
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4],
    )
    fig.add_trace(
        go.Scatter(
            x=labels, y=pl, name="Pl", mode="lines+markers",
            line=dict(color="#607d8b"),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=labels, y=bel, name="Bel", mode="lines+markers",
            line=dict(color="#2e7d32"), fill="tonexty",
            fillcolor="rgba(46, 125, 50, 0.15)",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=labels, y=width, name="width (Pl - Bel)",
            marker_color="#e6a817", opacity=0.7,
        ),
        row=2, col=1,
    )
    fig.add_hline(
        y=tau, line_dash="dash", line_color="#c62828",
        annotation_text=f"tau = {tau:.2f} (verifier.yaml)",
        row=1, col=1,
    )
    fig.add_hline(
        y=width_threshold, line_dash="dot", line_color="#e6a817",
        annotation_text=f"width threshold = {width_threshold:.2f}",
        row=2, col=1,
    )
    width_top = max([width_threshold, *width]) * 1.15 if width else width_threshold
    fig.update_yaxes(range=[0, 1], title_text="[Bel, Pl]", row=1, col=1)
    fig.update_yaxes(range=[0, width_top], title_text="width", row=2, col=1)
    fig.update_xaxes(
        title_text="scored turn (score_turn / iteration)", row=2, col=1
    )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


@st.fragment
def _belief_panel() -> None:
    """Live belief-frame panel; reruns independently of the chat pane."""
    st.subheader("Belief frame")
    cfg = _verifier_cfg()
    exchanges = st.session_state.get("exchanges") or []
    turns = _all_turns(exchanges)
    if not turns:
        st.info("No scored turns yet. Ask a question to see the frame move.")
        return
    labels = [_turn_label(turn) for turn in turns]
    selected = st.select_slider(
        "Scored turn",
        options=list(range(len(turns))),
        value=len(turns) - 1,
        format_func=lambda idx: labels[idx],
        key=f"frame-turn-{len(turns)}",
    )
    turn = turns[selected]
    slot = st.empty()
    with slot.container():
        frame_dict = turn.get("belief_frame")
        if isinstance(frame_dict, Mapping) and frame_dict.get("mass"):
            st.plotly_chart(_frame_figure(frame_dict))
            mass = frame_dict["mass"]  # type: ignore[index]
            st.caption(
                " · ".join(
                    f"{key} {float(mass[key]):.2f}"  # type: ignore[index]
                    for key in ("pass", "partial", "fail", "Omega")
                    if key in mass  # type: ignore[operator]
                )
            )
        else:
            st.info("This turn carries no belief frame (metrics only).")
        def _fmt_turn(value: object) -> str:
            return f"{float(value):.3f}" if value is not None else "—"  # type: ignore[arg-type]

        metric_row_top = st.columns(2)
        metric_row_top[0].metric("Bel", _fmt_turn(turn.get("bel")))
        metric_row_top[1].metric("Pl", _fmt_turn(turn.get("pl")))
        metric_row_bottom = st.columns(2)
        metric_row_bottom[0].metric("width", _fmt_turn(turn.get("width")))
        metric_row_bottom[1].metric(
            "conflict", _fmt_turn(turn.get("conflict"))
        )
        decision = turn.get("decision")
        label = turn.get("label")
        st.caption(
            f"{_turn_label(turn)} belief_frame"
            + (f" · decision {decision}" if decision else "")
            + (f" · best {label}" if label else "")
        )
        if isinstance(frame_dict, Mapping) and frame_dict:
            cli = _cli_options()
            if bool(st.session_state.get("show_frames", cli.show_frames)):
                with st.expander("belief_frame JSON", expanded=False):
                    _render_json_value(frame_dict)
    st.subheader("[Bel, Pl] across turns")
    figure = _interval_figure(turns, cfg)
    if figure is not None:
        chart_slot = st.empty()
        chart_slot.plotly_chart(figure)
    else:
        st.info("No Bel/Pl intervals recorded yet.")


def _resume_pending(choice_key: str) -> None:
    """Resume a paused live exchange from a structured human choice."""
    pending = st.session_state.get("pending")
    if not pending:
        return
    from zydah.choices import OTHER_VALUE

    result = pending["result"]
    verifier = pending["verifier"]
    selected = None
    for choice in result.get("choices") or []:
        if str(choice.get("key", "")).upper() == str(choice_key).upper():
            selected = choice
            break
    other_text = None
    if selected is not None and str(selected.get("value")) == OTHER_VALUE:
        other_text = st.session_state.get("other_text", "")
    try:
        new_result = verifier.resume_direct(
            result, choice_key, other_text=other_text
        )
    except Exception as exc:  # keep the chat usable, as the terminal does
        if live_demo is not None:
            live_demo._log_error("streamlit_resume_direct", exc)
        st.error(f"Resume failed: {exc}")
        return
    idx = pending["exchange_idx"]
    exchange = st.session_state["exchanges"][idx]
    updated = _exchange_from_result(
        exchange["question"],
        exchange["muse_text"],
        exchange["muse_turn"],
        new_result,
        exchange.get("usage"),
        source=exchange.get("source", "live"),
        usage_stages=exchange.get("usage_stages"),
    )
    exchange.update(updated)
    if new_result.get("needs_human"):
        pending["result"] = new_result
    else:
        st.session_state.pop("pending", None)


def _render_exchange(exchange: Mapping, is_pending: bool) -> None:
    muse_turn = exchange["muse_turn"]
    with st.chat_message("user"):
        st.markdown(str(exchange["question"]))
    with st.chat_message("assistant"):
        st.markdown(f"**[muse turn {muse_turn}]**")
        muse_text = str(exchange.get("muse_text") or "")
        if muse_text:
            _render_text_or_json(muse_text)
        else:
            st.caption("(no model text recorded for this replay)")
        if exchange.get("source") == "mock":
            artifact = exchange.get("artifact") or "artifact"
            st.caption(f"mock replay · {artifact} · no live call made")
    with st.chat_message("assistant"):
        st.markdown(f"**[zydah turn {muse_turn}]** {_conclusion(exchange)}")
        turns = exchange.get("turns") or []
        last = turns[-1] if turns else {}

        def _fmt(value: object) -> str:
            return f"{float(value):.3f}" if value is not None else "—"  # type: ignore[arg-type]

        verdict = exchange.get("verdict")
        decision = last.get("decision")
        banner = f"decision: {decision or '—'} · verdict: {verdict if verdict else '—'}"
        if decision == "commit":
            st.success(banner)
        elif decision == "escalate" or exchange.get("needs_human"):
            st.warning(banner)
        else:
            st.info(banner)
        metric_cols = st.columns(4)
        for col, key, title in (
            (metric_cols[0], "bel", "Bel"),
            (metric_cols[1], "pl", "Pl"),
            (metric_cols[2], "width", "width"),
            (metric_cols[3], "conflict", "conflict"),
        ):
            col.metric(title, _fmt(last.get(key)))
        cli = _cli_options()
        show_usage = bool(st.session_state.get("show_usage", cli.show_usage))
        show_frames = bool(
            st.session_state.get("show_frames", cli.show_frames)
        )
        if show_usage:
            st.caption(
                f"usage: {_usage_text(exchange.get('usage'))}"  # type: ignore[arg-type]
                f" · followups {exchange.get('followups', 0)}"
                f" · source {exchange.get('source', '')}"
            )
            stages = exchange.get("usage_stages")
            if stages:
                total_calls = sum(
                    int(stage.get("calls") or 0) for stage in stages  # type: ignore[union-attr]
                )
                with st.expander(f"Model calls ({total_calls})"):
                    st.dataframe(stages, hide_index=True)
        else:
            st.caption(
                f"followups {exchange.get('followups', 0)}"
                f" · source {exchange.get('source', '')}"
            )
        if turns and show_frames:
            with st.expander(
                f"belief frames ({len(turns)} scored iteration(s))"
            ):
                for turn in turns:
                    frame_dict = turn.get("belief_frame")
                    mass_text = ""
                    if isinstance(frame_dict, Mapping):
                        mass = frame_dict.get("mass") or {}
                        mass_text = " · ".join(
                            f"{key} {float(mass[key]):.2f}"  # type: ignore[index]
                            for key in ("pass", "partial", "fail", "Omega")
                            if key in mass  # type: ignore[operator]
                        )
                    st.markdown(
                        f"**{_turn_label(turn)} belief_frame** — {mass_text}"
                    )
                    if isinstance(frame_dict, Mapping) and frame_dict:
                        _render_json_value(frame_dict)
                    judge_frames = turn.get("judge_frames")
                    if judge_frames:
                        judges = turn.get("judges") or []
                        for idx, judge_frame in enumerate(judge_frames):
                            judge = (
                                judges[idx]  # type: ignore[index]
                                if idx < len(judges)  # type: ignore[arg-type]
                                else {}
                            )
                            tag = judge.get("tag") or f"ZJ-{idx + 1:02d}"  # type: ignore[union-attr]
                            seat = judge.get("seat") or judge.get("model") or ""  # type: ignore[union-attr]
                            st.markdown(
                                f"judge {tag} (seat {seat}) belief_frame:"
                            )
                            _render_json_value(judge_frame)
                    if turn.get("prompt"):
                        st.markdown("follow-up sent to the model:")
                        _render_json_value(turn["prompt"])
                    elif turn.get("followup"):
                        st.markdown("follow-up payload:")
                        _render_json_value(turn["followup"])
                    if turn.get("response"):
                        _render_text_or_json(str(turn["response"]))
        if exchange.get("needs_human"):
            human_question = exchange.get("human_question")
            if human_question:
                st.markdown(f"**[human]** {human_question}")
            choices = exchange.get("choices") or []
            if choices and is_pending:
                from zydah.choices import OTHER_VALUE

                if any(
                    str(choice.get("value")) == OTHER_VALUE
                    for choice in choices
                ):
                    st.text_input("Other — free text", key="other_text")
                for choice in choices:
                    key = str(choice.get("key", ""))
                    value = str(choice.get("value", ""))
                    st.button(
                        f"{key}. {value}",
                        key=f"choice-{muse_turn}-{key}",
                        on_click=_resume_pending,
                        args=(key,),
                    )
            elif choices:
                st.caption("Choices are shown on the latest paused turn.")
            elif human_question:
                st.caption(
                    "Ask a refined follow-up below to continue."
                )


def _sidebar(cfg, mock: bool, cli) -> Optional[Path]:
    with st.sidebar:
        st.header("Zydah")
        if mock:
            st.info("Mock mode — replaying a saved artifact. No live call.")
        else:
            st.success("Live mode — endpoint configured.")
        st.caption("Endpoint (names only, values never shown):")
        for name, present in _endpoint_status().items():
            st.caption(f"{'●' if present else '○'} {name}")
        st.caption(
            "Thresholds from verifier.yaml: "
            f"width ≤ {float(_cfg_value(cfg, 'width_threshold')):.2f}, "
            f"conflict ≤ {float(_cfg_value(cfg, 'conflict_threshold')):.2f}, "
            f"tau {float(_cfg_value(cfg, 'tau')):.2f}, "
            f"rule {cfg.rule}, "
            f"max iterations {int(_cfg_value(cfg, 'max_iterations'))}"
        )
        st.caption(f"Verifier model: {_model_id(cfg)}")

        st.subheader("Demo options")
        st.toggle(
            "Zydah panel",
            value=cli.panel_visible,
            key="show_zydah_panel",
            help="Belief-frame panel visibility (--zydah-panel).",
        )
        panel_on = st.toggle(
            "Panel scoring (3 judges)",
            value=bool(cli.panel_scoring),
            key="panel_scoring",
            help="verify() with the judge panel instead of the "
            "single-elicitor direct path (--panel-scoring).",
        )
        st.selectbox(
            "Panel source",
            ["yaml", "base"],
            index=0 if cli.panel_source == "yaml" else 1,
            key="panel_source",
            disabled=not panel_on,
            help="Judges from conf/panel.yaml, or the hardcoded "
            "Wonder/Spark/Spark Open models (--panel-source).",
        )
        st.text_input(
            "Panel model",
            value=cli.panel_model or "",
            key="panel_model",
            disabled=not panel_on,
            help="Override every judge seat's model for this run "
            "(--panel-model muse-spark). Names, sessions, and effort "
            "are preserved.",
        )
        st.selectbox(
            "Score mode",
            ["every", "substantive"],
            index=0 if cli.score_mode == "every" else 1,
            key="score_mode",
            help="Score every response, or only substantive ones "
            "(--score-mode).",
        )
        st.toggle(
            "Show frames",
            value=bool(cli.show_frames),
            key="show_frames",
            help="Belief-frame JSON and follow-up transcript "
            "(--show-frames).",
        )
        st.toggle(
            "Show usage",
            value=bool(cli.show_usage),
            key="show_usage",
            help="Per-turn seconds/tokens in the verdict line "
            "(--show-usage).",
        )

        st.subheader("Mock artifact")
        artifact_paths = sorted(
            str(path)
            for path in list(ARTIFACT_DIR.glob("*.json"))
            + list(ARTIFACT_DIR.glob("*.jsonl"))
        )
        if cli.mock_path is not None and str(cli.mock_path) not in artifact_paths:
            artifact_paths.insert(0, str(cli.mock_path))
        selected = None
        if artifact_paths:
            if cli.mock_path is not None and str(cli.mock_path) in artifact_paths:
                default_idx = artifact_paths.index(str(cli.mock_path))
            elif str(DEFAULT_ARTIFACT) in artifact_paths:
                default_idx = artifact_paths.index(str(DEFAULT_ARTIFACT))
            else:
                default_idx = 0
            choice = st.selectbox(
                "Mock artifact",
                artifact_paths,
                index=default_idx,
                key="mock_artifact",
                format_func=lambda p: Path(p).name,
                help="Replay this artifact instead of live calls "
                "(--mock-artifact).",
            )
            selected = Path(choice)
            if st.button("Replay artifact"):
                muse_turn = int(st.session_state["muse_counter"]) + 1
                exchange = _load_mock_exchange(selected, muse_turn, cfg)
                if exchange is not None:
                    st.session_state["exchanges"].append(exchange)
                    st.session_state["muse_counter"] = muse_turn
                    st.rerun()
                else:
                    st.error(f"Could not load {selected.name}.")
        if st.button("Clear history"):
            st.session_state["exchanges"] = []
            st.session_state["muse_counter"] = 0
            st.session_state.pop("pending", None)
            st.rerun()
        st.caption(
            "Deterministic fusion owns the state; the LLM supplies "
            "responses and masses. Width and conflict are routing "
            "signals, not noise."
        )
    return selected


def _hide_dev_chrome() -> None:
    """Hide Streamlit dev chrome (toolbar, deploy, status widget)."""
    st.markdown(
        """
        <style>
        [data-testid="stToolbar"] { display: none; }
        [data-testid="stAppDeployButton"] { display: none; }
        [data-testid="stStatusWidget"] { display: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Zydah live demo", layout="wide")
    _hide_dev_chrome()
    cfg = _verifier_cfg()
    cli = _cli_options()
    mock = _mock_mode() or cli.mock_path is not None
    if "exchanges" not in st.session_state:
        st.session_state["exchanges"] = []
        st.session_state["muse_counter"] = 0

    st.title("Zydah live demo")
    st.caption(
        "Dempster-Shafer belief fusion over [muse] responses. "
        "Each prompt is streamed, sealed, and scored by deterministic "
        "fusion; the [Bel, Pl] width and conflict mass decide the verdict."
    )
    selected_artifact = _sidebar(cfg, mock, cli)
    if mock and not st.session_state["exchanges"]:
        seeded = _load_mock_exchange(selected_artifact, 1, cfg)
        if seeded is not None:
            seeded["muse_turn"] = 1
            st.session_state["exchanges"] = [seeded]
            st.session_state["muse_counter"] = 1

    panel_model_value = str(
        st.session_state.get("panel_model", cli.panel_model or "")
    ).strip()
    settings = {
        "panel_scoring": bool(
            st.session_state.get("panel_scoring", cli.panel_scoring)
        ),
        "panel_source": st.session_state.get("panel_source", cli.panel_source),
        "panel_model": panel_model_value or None,
        "score_mode": st.session_state.get("score_mode", cli.score_mode),
    }
    show_panel = bool(
        st.session_state.get("show_zydah_panel", cli.panel_visible)
    )
    if show_panel:
        chat_col, panel_col = st.columns([3, 2])
        with panel_col:
            _belief_panel()
        chat_area = chat_col
    else:
        chat_area = st.container()
    with chat_area:
        pending = st.session_state.get("pending")
        pending_idx = pending["exchange_idx"] if pending else None
        for idx, exchange in enumerate(st.session_state["exchanges"]):
            _render_exchange(exchange, is_pending=(idx == pending_idx))

        prompt = st.chat_input("Ask a question…")
        if prompt and pending and (pending["result"].get("choices")):
            st.warning(
                "Zydah is paused for a structured choice above. "
                "Pick one to resume, or clear the history."
            )
        elif prompt:
            muse_turn = int(st.session_state["muse_counter"]) + 1
            with st.chat_message("user"):
                st.markdown(prompt)
            if mock:
                exchange = _load_mock_exchange(
                    selected_artifact, muse_turn, cfg
                )
                if exchange is None:
                    st.error(
                        "No mock artifact found under outputs/live_demo/."
                    )
                else:
                    exchange["question"] = prompt
                    st.session_state["exchanges"].append(exchange)
                    st.session_state["muse_counter"] = muse_turn
                    st.rerun()
            else:
                with st.chat_message("assistant"):
                    st.markdown(f"**[muse turn {muse_turn}]**")
                    muse_slot = st.empty()
                try:
                    exchange = _run_live(
                        prompt, muse_turn, muse_slot, settings
                    )
                except Exception as exc:
                    if live_demo is not None:
                        live_demo._log_error("streamlit_live_call", exc)
                    st.error(f"Live call failed: {exc} (logged to errors.log)")
                else:
                    st.session_state["exchanges"].append(exchange)
                    st.session_state["muse_counter"] = muse_turn
                    st.rerun()


if __name__ == "__main__":
    main()

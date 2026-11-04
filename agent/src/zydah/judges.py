"""Heterogeneous Meta-model judge panel.

Judges emit structured JSON masses over the frame, including m(Omega)
(ignorance). Panel:
- Muse Wonder (Meta Model API)
- Muse Spark (Meta Model API)
- Muse Spark Open via Baseten (model ID below)

Secrets: API keys come only from environment variables (META_API_KEY,
BASETEN_API_KEY); they are never committed. Unit tests use SeededMockPanel
and make no API calls.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence

from pathlib import Path

from omegaconf import OmegaConf
from pydantic import BaseModel

from zydah.fusion import Mass, make_mass
from zydah.transport import chat_complete, load_transport_config

JUDGE_WONDER = "muse-wonder"
JUDGE_SPARK = "muse-spark"
JUDGE_SPARK_OPEN = "muse-spark-open"
PANEL = (JUDGE_WONDER, JUDGE_SPARK, JUDGE_SPARK_OPEN)

SPARK_OPEN_BASETEN_ID = "meta-models/muse-spark-open-1026"

_PANEL_CONFIG_PATH = Path(__file__).parent / "conf" / "panel.yaml"


class PanelJudge(BaseModel):
    """One judge entry in panel.yaml; model defaults to Spark."""

    name: str
    model: str = JUDGE_SPARK
    effort: Optional[str] = None


class PanelConfig(BaseModel):
    """Panel settings (conf/panel.yaml through this pydantic interface).

    v1 quorum is 2-of-3: at least ``quorum`` judges must return a mass
    for the cycle to fuse. Default effort is low. The judges list is
    required — a misconfigured YAML fails validation instead of
    silently falling back to a hardcoded panel.
    """

    quorum: int = 2
    effort: str = "low"
    judges: List[PanelJudge]


def load_panel_config(path: Optional[Path] = None) -> PanelConfig:
    cfg = OmegaConf.load(path or _PANEL_CONFIG_PATH)
    return PanelConfig(**OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]

OMEGA_KEYS = frozenset({"omega", "Ω", "*", "uncertain", "unknown", "ignorance"})
_COMPOUND_SEPARATORS = ("+", ",", "|")


def _split_compound(key: str) -> FrozenSet[str]:
    for sep in _COMPOUND_SEPARATORS:
        if sep in key:
            return frozenset(part.strip() for part in key.split(sep) if part.strip())
    return frozenset({key.strip()})


def parse_mass_json(payload: Mapping[str, float], frame: FrozenSet[str]) -> Mass:
    """Parse one judge's structured JSON output into a validated mass.

    Payload maps singleton names to mass, compound "a+b" keys to subset mass,
    and any OMEGA_KEYS alias ("omega", "*", ...) to m(Omega). Values must be
    non-negative and sum to 1; unknown labels are rejected.
    """
    frame = frozenset(frame)
    assign: Dict[FrozenSet[str], float] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise ValueError(f"mass JSON keys must be strings, got {key!r}")
        focal = frame if key.strip().lower() in OMEGA_KEYS else _split_compound(key)
        if not focal <= frame:
            raise ValueError(f"unknown label(s) {set(focal) - set(frame)} in {key!r}")
        assign[focal] = assign.get(focal, 0.0) + value
    return make_mass(frame, assign)


LABEL_ANCHORS = (
    "Label meanings: pass = the evidence supports the claim; "
    "partial = the evidence partly supports and partly contradicts the "
    "claim, or covers only part of it; fail = the evidence contradicts "
    "the claim; omega = the evidence is insufficient to decide. Do not "
    "use fail merely because a claim is unproven: unproven means omega."
)


def mass_prompt(evidence: str, frame: FrozenSet[str]) -> str:
    """Prompt template eliciting structured JSON masses from a judge."""
    labels = sorted(frame)
    return (
        "You are an expert evaluator. Assess the evidence below against the "
        f"labels {labels}.\n\n{LABEL_ANCHORS}\n\nReturn ONLY a JSON object "
        "mapping each label to a probability mass, plus an \"omega\" key "
        "for your ignorance (mass you cannot commit to any label). Masses "
        "must be non-negative numbers summing to 1. Example: "
        '{"' + '": 0.6, "'.join(labels) + '": 0.1, "omega": 0.3}.\n\n'
        f"Evidence:\n{evidence}"
    )


class SeededMockPanel:
    """Deterministic mock panel for tests and offline harness runs.

    NOT a model of real judges: each judge draws uniform random weights over
    singletons plus Omega (scaled by omega_scale), boosted toward `lean`
    when given. Seeded per (seed, item_id, run, judge) so stability
    experiments resample judges while holding the agent output fixed.
    """

    def __init__(
        self,
        frame: FrozenSet[str],
        seed: int = 0,
        judges: Sequence[str] = PANEL,
        lean: str | None = None,
        lean_boost: float = 2.0,
        omega_scale: float = 0.5,
    ) -> None:
        self.frame = frozenset(frame)
        self.seed = seed
        self.judges = list(judges)
        if lean is not None and lean not in self.frame:
            raise ValueError(f"lean label {lean!r} not in frame {set(frame)}")
        self.lean = lean
        self.lean_boost = lean_boost
        self.omega_scale = omega_scale

    def elicit(self, item_id: str = "", run: int = 0) -> List[Mass]:
        masses = []
        for judge in self.judges:
            rng = random.Random(f"{self.seed}|{item_id}|{run}|{judge}")
            weights = {w: rng.random() for w in self.frame}
            if self.lean is not None:
                weights[self.lean] += self.lean_boost
            weights["__omega__"] = rng.random() * self.omega_scale
            total = sum(weights.values())
            assign = {w: weights[w] / total for w in self.frame}
            assign[self.frame] = weights["__omega__"] / total
            masses.append(make_mass(self.frame, assign))
        return masses


def extract_json_object(text: str) -> Dict:
    """Extract the first JSON object from model text (fences tolerated)."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object in model response: {text[:80]!r}")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError(f"model response JSON is not an object: {text[:80]!r}")
    return parsed


class MetaApiJudge:
    """Judge backed by the model API transport (see zydah.transport).

    The endpoint model comes from the env var named in transport.yaml
    when set; otherwise the judge's own ``model`` name is sent.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        effort: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("META_API_KEY")
        self.effort = effort
        self.name = name or model

    def prompt(self, evidence: str, frame: FrozenSet[str]) -> str:
        return mass_prompt(evidence, frame)

    def parse(self, response: Mapping[str, float], frame: FrozenSet[str]) -> Mass:
        return parse_mass_json(response, frame)

    def parse_text(self, text: str, frame: FrozenSet[str]) -> Mass:
        return parse_mass_json(extract_json_object(text), frame)

    def judge(self, evidence: str, frame: FrozenSet[str]) -> Mass:
        # The judge's own configured model is sent; the transport may
        # still resolve an endpoint deployment override. (Previously an
        # env lookup here collapsed every judge to one model.)
        text = chat_complete(
            self.prompt(evidence, frame),
            model_id=self.model,
            reasoning_effort=self.effort,
        )
        return self.parse_text(text, frame)


class LivePanel:
    """Judge panel over live MetaApiJudge clients (elicit-compatible).

    ``item_id`` carries the evidence text, matching how the pipe-through
    interface calls panels. v1 quorum: at least ``quorum`` of the judges
    must return a mass (default 2 of 3); individual judge failures are
    recorded in ``last_failures`` instead of killing the cycle. Session
    ids are provenance labels (one per judge), carried alongside the
    masses — the transport itself is stateless.
    """

    def __init__(
        self,
        frame: FrozenSet[str],
        judges: Sequence[MetaApiJudge],
        quorum: int = 2,
        session_ids: Optional[Sequence[str]] = None,
        on_judge_start=None,
    ) -> None:
        self.frame = frozenset(frame)
        self.judges = list(judges)
        self.quorum = int(quorum)
        self.session_ids = list(session_ids) if session_ids else []
        self.last_failures: List[str] = []
        # Optional observer called with each judge before it is asked,
        # so UIs can identify which model is about to respond.
        self.on_judge_start = on_judge_start

    @classmethod
    def from_config(
        cls,
        config: PanelConfig,
        frame: FrozenSet[str],
        session_ids: Optional[Sequence[str]] = None,
        on_judge_start=None,
    ) -> "LivePanel":
        judges = [
            MetaApiJudge(
                entry.model,
                effort=entry.effort or config.effort,
                name=entry.name,
            )
            for entry in config.judges
        ]
        return cls(
            frame,
            judges=judges,
            quorum=config.quorum,
            session_ids=session_ids,
            on_judge_start=on_judge_start,
        )

    def elicit(self, item_id: str = "", run: int = 0) -> List[Mass]:
        masses: List[Mass] = []
        self.last_failures = []
        for judge in self.judges:
            if self.on_judge_start is not None:
                self.on_judge_start(judge)
            try:
                masses.append(judge.judge(item_id, self.frame))
            except Exception as exc:  # one judge down must not kill the cycle
                self.last_failures.append(
                    f"{getattr(judge, 'name', judge)}: {exc}"
                )
        if len(masses) < self.quorum:
            raise RuntimeError(
                f"panel quorum not met: {len(masses)}/{len(self.judges)} "
                f"judges returned (need {self.quorum}); "
                f"failures: {self.last_failures}"
            )
        return masses


class BasetenJudge(MetaApiJudge):
    """Muse Spark Open via Baseten (model meta-models/muse-spark-open-1026)."""

    def __init__(
        self,
        model_id: str = SPARK_OPEN_BASETEN_ID,
        api_key: str | None = None,
    ) -> None:
        self.model_id = model_id
        super().__init__(
            model=f"baseten:{model_id}", api_key=api_key or os.environ.get("BASETEN_API_KEY")
        )

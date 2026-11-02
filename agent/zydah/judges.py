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

import os
import random
from typing import Dict, FrozenSet, List, Mapping, Sequence

from zydah.fusion import Mass, make_mass

JUDGE_WONDER = "muse-wonder"
JUDGE_SPARK = "muse-spark"
JUDGE_SPARK_OPEN = "muse-spark-open"
PANEL = (JUDGE_WONDER, JUDGE_SPARK, JUDGE_SPARK_OPEN)

SPARK_OPEN_BASETEN_ID = "meta-models/muse-spark-open-1026"

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


def mass_prompt(evidence: str, frame: FrozenSet[str]) -> str:
    """Prompt template eliciting structured JSON masses from a judge."""
    labels = sorted(frame)
    return (
        "You are an expert evaluator. Assess the evidence below against the "
        f"labels {labels}.\n\nReturn ONLY a JSON object mapping each label to "
        "a probability mass, plus an \"omega\" key for your ignorance "
        "(mass you cannot commit to any label). Masses must be non-negative "
        "numbers summing to 1. Example: "
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


class MetaApiJudge:
    """Judge backed by the Meta Model API (client wiring is future work).

    Prompt construction and response parsing are implemented; transport is
    intentionally unwired rather than guessed. judge() raises until a real
    client lands.
    """

    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("META_API_KEY")

    def prompt(self, evidence: str, frame: FrozenSet[str]) -> str:
        return mass_prompt(evidence, frame)

    def parse(self, response: Mapping[str, float], frame: FrozenSet[str]) -> Mass:
        return parse_mass_json(response, frame)

    def judge(self, evidence: str, frame: FrozenSet[str]) -> Mass:
        raise NotImplementedError(
            "Meta Model API transport not wired; use prompt()/parse() with a real client"
        )


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

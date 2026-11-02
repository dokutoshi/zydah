"""Pipe-through interface: direct response vs. belief-fused response.

An agent or LLM pipes evidence in and gets two tagged views in one result:

- ``[muse]``  first response: the model's direct answer (lead judge's
  own verdict, no fusion applied).
- ``[zydah]`` verification cycle: the full judge panel combined via
  belief fusion (Yager by default, or Smets' TBM), decided on its
  [Bel, Pl] interval with conflict mass K exposed as a routing signal;
  the verifier's outcome rides along when one is supplied.

User-facing tagged answers must use the sanctioned attribution API:
capture the endpoint's first response with ``capture_first_response``,
run verification, then pair them with ``seal_cycle`` and render with
``format_attributed_cycle``. The sealed ``[muse]`` text is frozen and
cannot be revised from verifier output. ``format_tagged`` remains only
a lower-level formatter for pre-shaped verdict dictionaries.

Panel defaults to ``SeededMockPanel`` (deterministic, no API calls) until
live judge transport lands; any object with
``elicit(item_id, run) -> List[Mass]`` can be injected.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence

from zydah.fusion import (
    EMPTY,
    Mass,
    TotalConflictError,
    conjunctive_combine,
    conflict_mass,
    pignistic,
    yager_combine,
)
from zydah.judges import SeededMockPanel
from zydah.router import TAU_GENERAL, decide_best

DIRECT_TAG = "muse"
FUSED_TAG = "zydah"
MODEL_ENDPOINT_SOURCE = "model-endpoint"


@dataclass(frozen=True)
class FirstResponse:
    """Unedited model text captured before verification begins."""

    text: str
    model_id: str
    source: str
    verdict: Optional[str] = None


@dataclass(frozen=True)
class AttributedCycle:
    """Frozen pairing of one first response with one verification result."""

    first: FirstResponse
    zydah: Mapping[str, object]
    verification: Mapping[str, object]


def capture_first_response(
    *,
    text: str,
    model_id: str,
    direct: Optional[Mapping[str, object]] = None,
    source: str = MODEL_ENDPOINT_SOURCE,
) -> FirstResponse:
    """Freeze endpoint text before any verifier output exists.

    ``source`` must identify the model endpoint. This is the attribution
    contract: assistant prose summarized after verification must never be
    presented as the model's first response.
    """
    if source != MODEL_ENDPOINT_SOURCE:
        raise ValueError(
            f"first responses must come from {MODEL_ENDPOINT_SOURCE!r}, not {source!r}"
        )
    if not text or not text.strip():
        raise ValueError("first response text must be non-empty model output")
    verdict = direct.get("verdict") if direct is not None else None
    if verdict is not None and not isinstance(verdict, str):
        raise TypeError("direct verdicts must be strings or None")
    return FirstResponse(text=text, model_id=model_id, source=source, verdict=verdict)


def seal_cycle(first: FirstResponse, verification: Mapping[str, object]) -> AttributedCycle:
    """Pair a frozen first response with a completed verification result.

    The zydah and verification mappings are snapshot-copied, so edits made
    after sealing cannot change the attributed output.
    """
    if not isinstance(first, FirstResponse):
        raise TypeError("first responses must be captured with capture_first_response")
    if first.source != MODEL_ENDPOINT_SOURCE:
        raise ValueError("sealed cycles must start from a model-endpoint first response")
    try:
        result = verification["result"]
        zydah = result[FUSED_TAG]
    except (KeyError, TypeError) as exc:
        raise ValueError("verification must contain result['zydah']") from exc
    if not isinstance(zydah, Mapping):
        raise TypeError("verification['zydah'] must be a mapping")
    return AttributedCycle(
        first=first,
        zydah=MappingProxyType(deepcopy(dict(zydah))),
        verification=MappingProxyType(deepcopy(dict(verification))),
    )


def format_attributed_cycle(cycle: AttributedCycle) -> str:
    """Render a sealed cycle as [muse] / [zydah] tagged text."""
    if not isinstance(cycle, AttributedCycle):
        raise TypeError("tagged cycles must be sealed with seal_cycle")
    muse = cycle.first
    zydah = cycle.zydah
    verification = cycle.verification
    provenance = f"(first response, direct; model={muse.model_id}; source={muse.source}"
    if muse.verdict is not None:
        provenance += f"; direct verdict={muse.verdict}"
    provenance += ")"
    return (
        f"[muse]\n"
        f"{muse.text}\n"
        f"{provenance}\n"
        f"\n"
        f"[zydah]\n"
        f"verdict: {zydah['verdict']} decision={zydah['decision']} "
        f"bel={zydah['bel']:.4f} pl={zydah['pl']:.4f} "
        f"width={zydah['width']:.4f} conflict={zydah['conflict']:.4f} "
        f"rule={zydah['rule']}\n"
        f"verifier: {verification['verdict']} "
        f"followups={verification['followups']} "
        f"model={verification['model_id']}\n"
    )


def _argmax(dist: Dict[str, float]) -> str:
    return sorted(dist, key=lambda w: (-dist[w], w))[0]


def direct_verdict(masses: Sequence[Mass], frame: FrozenSet[str]) -> Dict:
    """Direct arm: lead judge's pignistic best label (no fusion)."""
    dist = pignistic(masses[0], frame)
    return {"verdict": _argmax(dist), "pignistic": dist}


def fused_verdict(
    masses: Sequence[Mass],
    frame: FrozenSet[str],
    tau: float = TAU_GENERAL,
    rule: str = "yager",
) -> Dict:
    """Fused arm: combine the panel, then decide on the [Bel, Pl] interval.

    Total conflict under TBM has no pignistic distribution; report an
    abstained escalate with conflict = 1.0 instead of raising.
    """
    frame = frozenset(frame)
    if rule == "yager":
        fused = yager_combine(masses, frame)
        kappa = conflict_mass(masses, frame)
    elif rule == "tbm":
        fused = conjunctive_combine(masses, frame)
        kappa = fused.get(EMPTY, 0.0)
    else:
        raise ValueError(f"unknown fusion rule {rule!r} (expected 'yager' or 'tbm')")
    try:
        out = decide_best(fused, frame, tau)
    except TotalConflictError:
        return {
            "verdict": None,
            "decision": "escalate",
            "bel": 0.0,
            "pl": 0.0,
            "width": 0.0,
            "conflict": 1.0,
            "pignistic": {},
            "rule": rule,
        }
    return {
        "verdict": out["best"],
        "decision": out["decision"],
        "bel": out["bel"],
        "pl": out["pl"],
        "width": out["width"],
        "conflict": kappa,
        "pignistic": out["pignistic"],
        "rule": rule,
    }


def fuse_response(
    evidence: str,
    frame: FrozenSet[str],
    panel: Optional[object] = None,
    tau: float = TAU_GENERAL,
    rule: str = "yager",
    run: int = 0,
) -> Dict:
    """Pipe evidence through both arms; return ``{"muse": ..., "zydah": ...}``.

    The mock panel seeds per (seed, evidence, run, judge), so the same
    evidence always yields the same result. Any object with
    ``elicit(item_id, run) -> List[Mass]`` works as the panel.
    """
    frame = frozenset(frame)
    panel = panel if panel is not None else SeededMockPanel(frame, seed=0)
    masses: List[Mass] = panel.elicit(item_id=evidence, run=run)  # type: ignore[attr-defined]
    if not masses:
        raise ValueError("panel returned no judge masses")
    return {
        DIRECT_TAG: direct_verdict(masses, frame),
        FUSED_TAG: fused_verdict(masses, frame, tau=tau, rule=rule),
    }


def format_tagged(result: Dict, verification: Optional[Dict] = None) -> str:
    """Render pre-shaped verdict dictionaries as [muse] / [zydah] text.

    This formatter does not enforce first-response provenance. User-facing
    tagged answers must use capture_first_response, seal_cycle, and
    format_attributed_cycle instead.
    """
    muse = result[DIRECT_TAG]
    zydah = result[FUSED_TAG]
    lines = (
        f"[muse]\n"
        f"verdict: {muse['verdict']} (first response, direct)\n"
        f"\n"
        f"[zydah]\n"
        f"verdict: {zydah['verdict']} decision={zydah['decision']} "
        f"bel={zydah['bel']:.4f} pl={zydah['pl']:.4f} "
        f"width={zydah['width']:.4f} conflict={zydah['conflict']:.4f} "
        f"rule={zydah['rule']}\n"
    )
    if verification is not None:
        lines += (
            f"verifier: {verification['verdict']} "
            f"followups={verification['followups']} "
            f"model={verification['model_id']}\n"
        )
    return lines

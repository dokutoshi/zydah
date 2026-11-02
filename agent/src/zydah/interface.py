"""Pipe-through interface: direct response vs. belief-fused response.

An agent or LLM pipes evidence in and gets two tagged views in one result:

- ``[muse]``  first response: the model's direct answer (lead judge's
  own verdict, no fusion applied).
- ``[zydah]`` verification cycle: the full judge panel combined via
  belief fusion (Yager by default, or Smets' TBM), decided on its
  [Bel, Pl] interval with conflict mass K exposed as a routing signal;
  the verifier's outcome rides along when one is supplied.

Panel defaults to ``SeededMockPanel`` (deterministic, no API calls) until
live judge transport lands; any object with
``elicit(item_id, run) -> List[Mass]`` can be injected.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Sequence

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
    """Render one evidence cycle as [muse] / [zydah] tagged text.

    Attribution is by cycle: [muse] is the model's first response (the
    direct arm); [zydah] is the later verification cycle (fused panel
    verdict, plus the verifier outcome when one is supplied).
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

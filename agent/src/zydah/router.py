"""Interval-based decision + escalation policy.

Rule: commit if Bel >= tau, reject if Pl < tau, escalate if the [Bel, Pl]
interval straddles tau. Escalation is driven by evidence (conflict mass,
interval width), never by a retry counter.

Thresholds are per-domain simple defaults:
- TAU_GENERAL = 0.80 for routine evaluation.
- TAU_HIGH_STAKES = 0.95 for high-stakes decisions.
"""

from __future__ import annotations

from typing import Dict, FrozenSet

from zydah.fusion import Mass, belief, pignistic, plausibility

TAU_GENERAL = 0.80
TAU_HIGH_STAKES = 0.95


def decide_interval(bel: float, pl: float, tau: float) -> str:
    """Trichotomy on one hypothesis interval. Bel >= tau wins ties."""
    if bel >= tau:
        return "commit"
    if pl < tau:
        return "reject"
    return "escalate"


def decide_hypothesis(mass: Mass, hypothesis: FrozenSet[str], tau: float) -> Dict:
    """Decide a named hypothesis, reporting its [Bel, Pl] interval."""
    bel = belief(mass, hypothesis)
    pl = plausibility(mass, hypothesis)
    return {
        "bel": bel,
        "pl": pl,
        "width": pl - bel,
        "decision": decide_interval(bel, pl, tau),
    }


def decide_best(mass: Mass, frame: FrozenSet[str], tau: float) -> Dict:
    """Decide the pignistic-best singleton (ties broken deterministically).

    Returns best label, full pignistic distribution, its [Bel, Pl] interval,
    and the commit/reject/escalate decision.
    """
    betp = pignistic(mass, frame)
    best = sorted(betp, key=lambda w: (-betp[w], w))[0]
    singleton = frozenset({best})
    bel = belief(mass, singleton)
    pl = plausibility(mass, singleton)
    return {
        "best": best,
        "pignistic": betp,
        "bel": bel,
        "pl": pl,
        "width": pl - bel,
        "decision": decide_interval(bel, pl, tau),
    }

"""Evidential belief fusion core.

Published methods only:
- Dempster's rule (Dempster 1967; Shafer 1976): normalized conjunctive
  combination; conflict mass K is renormalized away.
- Yager's rule (Yager 1987): conflict mass K is reassigned to Omega
  (ignorance), preserving disagreement as a signal.
- Smets' TBM (Smets & Kennes 1994): unnormalized conjunctive combination
  keeps K on the empty set; decisions use the pignistic transform.
- Shafer's reliability discounting (Shafer 1976): scale committed mass by
  alpha and move the remainder to Omega before combination.

Combination is simultaneous n-ary (one pass over the Cartesian product of
focal elements), hence order-invariant by construction rather than by
accident of a left fold.

Conventions:
- A frame is a frozenset of string hypotheses, e.g. {"pass","partial","fail"}.
- A mass is a dict {frozenset -> float} with finite, non-negative entries summing to 1.
- Inputs must be normalized (no empty-set mass). Unnormalized results (TBM)
  carry conflict on frozenset() and still sum to 1 including that entry.
"""

from __future__ import annotations

from itertools import product
from math import isfinite, prod
from typing import Dict, FrozenSet, Iterable, Mapping, Sequence, Union

Mass = Dict[FrozenSet[str], float]
EMPTY: FrozenSet[str] = frozenset()
_SUM_TOL = 1e-9
_CONFLICT_TOL = 1e-12

KeyLike = Union[str, Iterable[str]]


class TotalConflictError(ValueError):
    """Raised when conjunctive combination yields K == 1.

    Dempster's normalization divides by (1 - K) and is undefined here;
    Yager and TBM remain well-defined.
    """


def _as_focal(key: KeyLike) -> FrozenSet[str]:
    if isinstance(key, str):
        return frozenset({key})
    return frozenset(key)


def _check_finite(value: float, key: object) -> None:
    """Reject NaN and +/-Infinity as mass values.

    A mass is a probability, so it must be finite. This needs an explicit
    ``isfinite`` check: every comparison against NaN is False, so NaN would
    otherwise pass the negative and sum-to-one checks below and then be
    silently dropped by the ``value > 0`` filter in ``make_mass`` (e.g.
    ``{"pass": nan}`` returned ``{}``, a mass summing to 0). Judge JSON
    can carry such values (Python's ``json`` accepts ``NaN``/``Infinity``
    literals), so untrusted input must fail closed here.
    """
    if not isfinite(value):
        raise ValueError(f"non-finite mass {value!r} for {key!r}")


def make_mass(frame: FrozenSet[str], assign: Mapping[KeyLike, float]) -> Mass:
    """Build a validated normalized mass function.

    Keys: a string names a singleton; a tuple/list/set/frozenset names the
    corresponding subset (the full frame denotes Omega). Values must be
    finite (see ``_check_finite``), non-negative, and sum to 1. Empty-set
    keys are rejected: inputs are normalized beliefs, and m(empty) = 0
    by definition.
    """
    frame = frozenset(frame)
    total = 0.0
    mass: Mass = {}
    for key, value in assign.items():
        _check_finite(value, key)
        if value < 0:
            raise ValueError(f"negative mass {value!r} for {key!r}")
        focal = _as_focal(key)
        if not focal:
            raise ValueError("input masses must be normalized (no empty-set mass)")
        if not focal <= frame:
            raise ValueError(f"focal element {set(focal)} not a subset of frame {set(frame)}")
        mass[focal] = mass.get(focal, 0.0) + value
        total += value
    if abs(total - 1.0) > _SUM_TOL:
        raise ValueError(f"masses sum to {total}, expected 1.0")
    return {focal: value for focal, value in mass.items() if value > 0.0}


def discount_mass(mass: Mass, frame: FrozenSet[str], alpha: float) -> Mass:
    """Shafer's reliability discounting: trust a source only to degree alpha.

    m^alpha(X) = alpha * m(X) for X != Omega, and
    m^alpha(Omega) = 1 - alpha + alpha * m(Omega).

    alpha = 1 returns the source unchanged; alpha = 0 returns the vacuous
    mass on Omega. Discounting before combination models source
    unreliability explicitly: fully committed (Bayesian) sources are what
    trigger Zadeh's paradox, and moving (1 - alpha) to Omega keeps their
    disagreement from being amplified into certainty. alpha must be
    finite (see ``_check_finite``) and lie in [0, 1].
    """
    frame = frozenset(frame)
    _validate(mass, frame)
    _check_finite(alpha, "alpha")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")
    discounted: Mass = {}
    for focal, value in mass.items():
        if focal == frame:
            continue
        scaled = alpha * value
        if scaled > 0.0:
            discounted[focal] = scaled
    omega = 1.0 - alpha + alpha * mass.get(frame, 0.0)
    if omega > 0.0:
        discounted[frame] = omega
    return discounted


def _validate(mass: Mass, frame: FrozenSet[str]) -> None:
    total = 0.0
    for focal, value in mass.items():
        _check_finite(value, focal)
        if value < 0:
            raise ValueError(f"negative mass {value!r} for {set(focal)}")
        if not focal:
            raise ValueError("input masses must be normalized (no empty-set mass)")
        if not focal <= frame:
            raise ValueError(f"focal element {set(focal)} not a subset of frame {set(frame)}")
        total += value
    if abs(total - 1.0) > _SUM_TOL:
        raise ValueError(f"masses sum to {total}, expected 1.0")


def conjunctive_combine(masses: Sequence[Mass], frame: FrozenSet[str]) -> Mass:
    """Smets' unnormalized conjunctive combination (TBM), n-ary and simultaneous.

    m(C) = sum over A_1 cap ... cap A_n = C of prod_i m_i(A_i),
    including C = empty, whose mass is the conflict K.
    """
    frame = frozenset(frame)
    if not masses:
        raise ValueError("need at least one mass function")
    for mass in masses:
        _validate(mass, frame)
    focal_lists = [list(mass.items()) for mass in masses]
    combined: Mass = {}
    for combo in product(*focal_lists):
        focal = frame
        for candidate, _ in combo:
            focal = focal & candidate
        prob = prod(value for _, value in combo)
        if prob > 0.0:
            combined[focal] = combined.get(focal, 0.0) + prob
    return combined


def conflict_mass(masses: Sequence[Mass], frame: FrozenSet[str]) -> float:
    """Conflict mass K: conjunctive mass assigned to the empty set."""
    return conjunctive_combine(masses, frame).get(EMPTY, 0.0)


def dempster_combine(masses: Sequence[Mass], frame: FrozenSet[str]) -> Mass:
    """Dempster's rule: conjunctive combination normalized by (1 - K).

    Raises TotalConflictError when K == 1 (total conflict).
    """
    combined = conjunctive_combine(masses, frame)
    kappa = combined.pop(EMPTY, 0.0)
    if kappa >= 1.0 - _CONFLICT_TOL:
        raise TotalConflictError(f"total conflict: K = {kappa}")
    norm = 1.0 - kappa
    return {focal: value / norm for focal, value in combined.items()}


def yager_combine(masses: Sequence[Mass], frame: FrozenSet[str]) -> Mass:
    """Yager's rule: conjunctive combination with K reassigned to Omega.

    m_Y(Omega) = m_conj(Omega) + K; m_Y(empty) = 0. Well-defined even
    under total conflict, where the result is the vacuous mass on Omega.
    """
    frame = frozenset(frame)
    combined = conjunctive_combine(masses, frame)
    kappa = combined.pop(EMPTY, 0.0)
    if kappa > 0.0:
        combined[frame] = combined.get(frame, 0.0) + kappa
    return combined


def belief(mass: Mass, hypothesis: FrozenSet[str]) -> float:
    """Bel(H) = sum of mass of non-empty focal elements contained in H."""
    hypothesis = frozenset(hypothesis)
    return sum(value for focal, value in mass.items() if focal and focal <= hypothesis)


def plausibility(mass: Mass, hypothesis: FrozenSet[str]) -> float:
    """Pl(H) = sum of mass of focal elements intersecting H."""
    hypothesis = frozenset(hypothesis)
    return sum(value for focal, value in mass.items() if focal & hypothesis)


def pignistic(mass: Mass, frame: FrozenSet[str]) -> Dict[str, float]:
    """Pignistic transform: split each focal mass uniformly among its elements.

    BetP(w) = (1 / (1 - m(empty))) * sum_{A ni w} m(A) / |A|.
    The empty-set renormalization keeps TBM outputs decodable; raises
    TotalConflictError when m(empty) == 1 (nothing to distribute).
    """
    frame = frozenset(frame)
    denom = 1.0 - mass.get(EMPTY, 0.0)
    if denom <= _CONFLICT_TOL:
        raise TotalConflictError("pignistic transform undefined under total conflict")
    betp = {w: 0.0 for w in frame}
    for focal, value in mass.items():
        if not focal:
            continue
        share = value / (len(focal) * denom)
        for w in focal:
            betp[w] += share
    return betp


def belief_frame(mass: Mass, frame: FrozenSet[str]) -> Dict[str, object]:
    """Serialize the belief frame itself, not a derived scalar.

    The frame lists every hypothesis in the closed world; the mass
    mapping carries the full credal state, one entry per focal element
    (the whole frame renders as ``Omega``). Bel, Pl, and the pignistic
    distribution are all derivable from this value, which is why
    artifacts store it per turn instead of a single betting scalar:
    each prompt/response turn adjusts exactly this assignment.
    """
    frame = frozenset(frame)
    rendered: Dict[str, float] = {}
    for focal, value in mass.items():
        if not focal:
            key = "empty"
        elif focal == frame:
            key = "Omega"
        else:
            key = ",".join(sorted(focal))
        rendered[key] = rendered.get(key, 0.0) + float(value)
    return {"frame": sorted(frame), "mass": rendered}

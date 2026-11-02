"""Evaluation harness: majority vote vs. averaging vs. Dempster vs. Zydah.

Items carry reference outcomes and per-run judge masses; methods run on the
same masses so stability (decision flip rate under judge resampling),
correctness (accuracy vs. reference), and cost (escalation rate) are
directly comparable. All runs are seeded; manifests + JSONL logs make every
run reproducible.

Assumptions (documented, revisit with data):
- Judge labels/scores for the classical baselines derive from each judge's
  pignistic distribution (argmax label; full distribution as scores).
- A Zydah "escalate" counts as the pseudo-label "escalate" for flip rate
  and as an abstention for accuracy (accuracy_committed + coverage).
- Dempster is undefined under total conflict; compare() records that item
  as an abstention for the Dempster column rather than crashing.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, FrozenSet, List, Mapping, Sequence

from zydah.fusion import (
    Mass,
    TotalConflictError,
    conjunctive_combine,
    conflict_mass,
    dempster_combine,
    pignistic,
)
from zydah.fusion import yager_combine as _yager_combine
from zydah.router import TAU_GENERAL, decide_best

ESCALATE = "escalate"
METHODS = ("majority_vote", "average", "dempster", "zydah")


def _argmax_label(scores: Mapping[str, float]) -> str:
    return sorted(scores, key=lambda w: (-scores[w], w))[0]


def majority_vote(labels: Sequence[str]) -> str:
    """Plurality label; ties broken deterministically (sorted-first)."""
    if not labels:
        raise ValueError("need at least one label")
    counts = Counter(labels)
    return sorted(counts, key=lambda w: (-counts[w], w))[0]


def average_scores(score_dicts: Sequence[Mapping[str, float]]) -> str:
    """Argmax of per-label mean; missing labels count as 0. Ties sorted-first."""
    if not score_dicts:
        raise ValueError("need at least one score dict")
    totals: Dict[str, float] = {}
    for scores in score_dicts:
        for label, value in scores.items():
            totals[label] = totals.get(label, 0.0) + value
    means = {label: total / len(score_dicts) for label, total in totals.items()}
    return _argmax_label(means)


def dempster_verdict(masses: Sequence[Mass], frame: FrozenSet[str]) -> str:
    """Fuse with Dempster's rule; verdict is the pignistic argmax.

    Raises TotalConflictError under total conflict (rule undefined there).
    """
    return _argmax_label(pignistic(dempster_combine(masses, frame), frame))


def zydah_verdict(
    masses: Sequence[Mass],
    frame: FrozenSet[str],
    tau: float = TAU_GENERAL,
    rule: str = "yager",
) -> Dict:
    """Fuse (Yager default; "tbm" for unnormalized) and decide the interval.

    Returns verdict label, commit/reject/escalate decision, [Bel, Pl],
    conflict K, and the pignistic distribution. Total conflict under TBM
    has no pignistic distribution; it is reported as an abstained escalate.
    """
    frame = frozenset(frame)
    if rule == "yager":
        fused = _yager_combine(masses, frame)
    elif rule == "tbm":
        fused = conjunctive_combine(masses, frame)
    else:
        raise ValueError(f"unknown fusion rule {rule!r} (expected 'yager' or 'tbm')")
    kappa = fused.get(frozenset(), 0.0) if rule == "tbm" else conflict_mass(masses, frame)
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


def decision_flip_rate(run_a: Sequence[str | None], run_b: Sequence[str | None]) -> float:
    """Fraction of items whose output differs between two runs (same order)."""
    if len(run_a) != len(run_b):
        raise ValueError("runs must cover the same items in the same order")
    if not run_a:
        return 0.0
    return sum(a != b for a, b in zip(run_a, run_b)) / len(run_a)


def accuracy(
    verdicts: Sequence[str | None],
    references: Sequence[str],
    abstain: str | None = None,
) -> Dict:
    """Accuracy over committed verdicts plus coverage (1 - abstention rate)."""
    if len(verdicts) != len(references):
        raise ValueError("verdicts and references must align")
    committed = [(v, r) for v, r in zip(verdicts, references) if v != abstain]
    n_correct = sum(v == r for v, r in committed)
    return {
        "accuracy_committed": (n_correct / len(committed)) if committed else 0.0,
        "coverage": (len(committed) / len(verdicts)) if verdicts else 0.0,
        "n": len(verdicts),
        "n_committed": len(committed),
        "n_correct": n_correct,
    }


def run_manifest(seed: int, tau: float, frame: FrozenSet[str], rule: str = "yager") -> Dict:
    return {"seed": seed, "tau": tau, "frame": sorted(frame), "rule": rule, "methods": list(METHODS)}


def log_jsonl(path: str, record: Mapping) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _judge_labels_and_scores(
    masses: Sequence[Mass], frame: FrozenSet[str]
) -> tuple[List[str], List[Dict[str, float]]]:
    betps = [pignistic(mass, frame) for mass in masses]
    return [_argmax_label(betp) for betp in betps], betps


def compare(
    items: Sequence[Mapping],
    frame: FrozenSet[str],
    tau: float = TAU_GENERAL,
    seed: int = 0,
    rule: str = "yager",
) -> Dict:
    """Run all methods over items with per-run judge masses.

    Each item: {"id": str, "reference": str, "runs": [[Mass, ...], ...]}.
    Run 0 feeds accuracy; all runs feed flip rate between consecutive runs.
    """
    frame = frozenset(frame)
    references = [item["reference"] for item in items]
    outputs: Dict[str, List[List[str | None]]] = {m: [] for m in METHODS}
    records = []
    for item in items:
        runs = item["runs"]
        if not runs:
            raise ValueError(f"item {item.get('id')} has no runs")
        item_out: Dict[str, List[str | None]] = {m: [] for m in METHODS}
        zy = None
        for masses in runs:
            labels, scores = _judge_labels_and_scores(masses, frame)
            item_out["majority_vote"].append(majority_vote(labels))
            item_out["average"].append(average_scores(scores))
            try:
                item_out["dempster"].append(dempster_verdict(masses, frame))
            except TotalConflictError:
                item_out["dempster"].append(None)
            zy = zydah_verdict(masses, frame, tau=tau, rule=rule)
            item_out["zydah"].append(
                zy["verdict"] if zy["decision"] == "commit" else ESCALATE
            )
        for method in METHODS:
            outputs[method].append(item_out[method])
        records.append({"id": item["id"], "reference": item["reference"], **{m: item_out[m] for m in METHODS}})
    metrics = {}
    for method in METHODS:
        run0 = [per_item[0] for per_item in outputs[method]]
        acc = accuracy(run0, references, abstain=None if method != "zydah" else ESCALATE)
        flips = []
        n_runs = len(items[0]["runs"]) if items else 0
        for r in range(1, n_runs):
            flips.append(
                decision_flip_rate(
                    [per_item[r - 1] for per_item in outputs[method]],
                    [per_item[r] for per_item in outputs[method]],
                )
            )
        metrics[method] = {
            **acc,
            "flip_rate": (sum(flips) / len(flips)) if flips else 0.0,
        }
    return {"manifest": run_manifest(seed, tau, frame, rule), "metrics": metrics, "items": records}

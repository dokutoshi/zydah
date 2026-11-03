"""Graph resolver: candidate scores -> Dempster-Shafer masses (ZYDAH-114).

The deterministic baseline behind the ``Resolver`` protocol. Message
passing is deliberately simple: context affinity signals flow into
partitions and candidates, ``faster_than`` edges reward sources, and a
temperature softmax turns scores into a distribution. Entropy of that
distribution decides how much mass stays on Omega; a concentrated
partition earns a subset focal element instead of forcing a singleton.

The learned PyG scorer (ZYDAH-117) implements the same protocol and
must beat this baseline on held-out entities to replace it. This
module depends on numpy only — never on torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Mapping, Optional, Protocol, runtime_checkable

import numpy as np

from zydah.fusion import (
    Mass,
    belief,
    discount_mass,
    make_mass,
    pignistic,
    plausibility,
    yager_combine,
)
from zydah.graph import (
    AFFINITY,
    CANDIDATE,
    FASTER_THAN,
    MEMBER_OF,
    PARTITION,
    CandidateGraph,
    load_graph_config,
)

STATUS_RESOLVED = "resolved"
STATUS_SUBSET = "subset"
STATUS_UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class Resolution:
    """One resolver answer: credal mass plus its pignistic transform.

    ``candidate_scores`` is the pignistic (betting) distribution
    derived from ``mass`` — never a belief. Credal information lives
    in ``mass`` and in ``candidate_intervals`` ([Bel, Pl] per
    singleton); a uniform pignistic under all-Omega mass means
    complete ignorance, not equal belief.
    """

    frame: FrozenSet[str]
    mass: Mass
    candidate_scores: Mapping[str, float]
    partition_scores: Mapping[str, float]
    entropy_before: float
    entropy_after: float
    status: str
    best_partition: Optional[str] = None
    candidate_intervals: Mapping[str, tuple] = field(default_factory=dict)
    pignistic_scores: Mapping[str, float] = field(default_factory=dict)


@runtime_checkable
class Resolver(Protocol):
    """Protocol every resolver (baseline or learned) implements."""

    def resolve(self, graph: CandidateGraph) -> Resolution:
        ...


class NumpyResolver:
    """Edge-weighted message passing + entropy-calibrated mass conversion."""

    def __init__(self, config=None) -> None:
        self.cfg = config or load_graph_config()

    def _scores(self, graph: CandidateGraph) -> tuple[Dict[str, float], Dict[str, float]]:
        cfg = self.cfg
        candidates = graph.nodes_of_kind(CANDIDATE)
        cand_ids = [node.id for node in candidates]
        direct = {cid: 0.0 for cid in cand_ids}
        partition_signal: Dict[str, float] = {}
        members: Dict[str, List[str]] = {}
        for edge in graph.edges_of_kind(MEMBER_OF):
            members.setdefault(edge.target, []).append(edge.source)
        for edge in graph.edges_of_kind(AFFINITY):
            if edge.target in direct:
                direct[edge.target] += float(cfg.resolver.context_weight)
            elif graph.node(edge.target).kind == PARTITION:
                partition_signal[edge.target] = (
                    partition_signal.get(edge.target, 0.0)
                    + float(cfg.resolver.context_weight)
                )
        for edge in graph.edges_of_kind(FASTER_THAN):
            if edge.source in direct:
                margin = float(edge.features.get("margin", 0.0))
                direct[edge.source] += float(cfg.resolver.faster_weight) * (
                    margin / float(cfg.resolver.faster_margin_scale)
                )
        scores = dict(direct)
        for pid, signal in partition_signal.items():
            group = members.get(pid, [])
            if not group:
                continue
            share = float(cfg.resolver.partition_weight) * signal / len(group)
            for cid in group:
                scores[cid] += share
        partition_scores: Dict[str, float] = {}
        return scores, partition_scores

    def _build_resolution(
        self,
        frame: FrozenSet[str],
        mass: Mass,
        members: Optional[Dict[str, List[str]]] = None,
        best_partition: Optional[str] = None,
        entropy_before: Optional[float] = None,
        entropy_after: Optional[float] = None,
    ) -> Resolution:
        """Assemble a Resolution from a credal mass (single source).

        Candidate scores are recomputed as the pignistic transform of
        the mass, and intervals as [Bel, Pl] per singleton — the two
        are never conflated.
        """
        ids = sorted(frame)
        betp = pignistic(mass, frame)
        intervals = {
            cid: (
                belief(mass, frozenset({cid})),
                plausibility(mass, frozenset({cid})),
            )
            for cid in ids
        }
        groups = members or {}
        partition_scores = {
            pid: float(sum(betp[cid] for cid in group if cid in betp))
            for pid, group in groups.items()
        }
        if best_partition is None and groups:
            best_partition = max(
                groups,
                key=lambda pid: (
                    belief(mass, frozenset(groups[pid])),
                    pid,
                ),
            )
        if entropy_before is None:
            entropy_before = math.log(len(ids)) if len(ids) > 1 else 0.0
        if entropy_after is None:
            values = np.array([betp[cid] for cid in ids], dtype=float)
            entropy_after = float(-(values * np.log(values + 1e-12)).sum())
        omega = float(mass.get(frame, 0.0))
        max_fraction = float(self.cfg.calibration.max_entropy_fraction)
        top = max(betp, key=lambda cid: betp[cid]) if betp else None
        if omega >= max_fraction - 1e-9:
            status = STATUS_UNRESOLVED
        elif (
            best_partition is not None
            and len(groups.get(best_partition, [])) > 1
            and top is not None
            and betp[top] < float(self.cfg.resolver.resolved_threshold)
        ):
            status = STATUS_SUBSET
        else:
            status = STATUS_RESOLVED
        return Resolution(
            frame=frame,
            mass=mass,
            candidate_scores={cid: float(betp[cid]) for cid in ids},
            partition_scores=partition_scores,
            entropy_before=float(entropy_before),
            entropy_after=float(entropy_after),
            status=status,
            best_partition=best_partition,
            candidate_intervals=intervals,
            pignistic_scores={cid: float(betp[cid]) for cid in ids},
        )

    def condition(
        self,
        resolution: Resolution,
        support: FrozenSet[str],
        alpha: Optional[float] = None,
        best_partition: Optional[str] = None,
    ) -> Resolution:
        """Condition a resolution on new support via Yager's rule.

        The support focal element is built as a categorical source,
        discounted by ``alpha`` (Shafer reliability), and combined
        with Yager's rule so any conflict mass moves to Omega instead
        of being renormalized into certainty. This is the credal
        update: ``Bel(A | clue)`` and ``Pl(A | clue)`` change because
        the mass changed; the pignistic distribution follows.
        """
        frame = resolution.frame
        focal = frozenset(support)
        if not focal or not focal <= frame:
            raise ValueError("support must be a non-empty subset of the frame")
        reliability = float(
            self.cfg.resolver.conditioning_alpha if alpha is None else alpha
        )
        source = discount_mass(
            make_mass(frame, {focal: 1.0}), frame, reliability
        )
        combined = yager_combine([resolution.mass, source], frame)
        groups = {best_partition: sorted(focal)} if best_partition else {}
        return self._build_resolution(
            frame,
            combined,
            members=groups,
            best_partition=best_partition,
        )

    def resolve(self, graph: CandidateGraph) -> Resolution:
        candidates = graph.nodes_of_kind(CANDIDATE)
        if not candidates:
            raise ValueError("graph has no candidate nodes to resolve")
        frame = frozenset(node.id for node in candidates)
        scores, _ = self._scores(graph)
        ids = sorted(frame)
        raw = np.array([scores[cid] for cid in ids], dtype=float)
        temperature = float(self.cfg.calibration.temperature)
        scaled = raw / max(temperature, 1e-9)
        scaled = scaled - scaled.max()
        exp = np.exp(scaled)
        probs = exp / exp.sum()
        n = len(ids)
        entropy = float(-(probs * np.log(probs)).sum())
        entropy_before = math.log(n) if n > 1 else 0.0
        entropy_fraction = entropy / entropy_before if entropy_before > 0 else 0.0

        members: Dict[str, List[str]] = {}
        for edge in graph.edges_of_kind(MEMBER_OF):
            if edge.source in frame:
                members.setdefault(edge.target, []).append(edge.source)
        prob_by_id = dict(zip(ids, probs))
        partition_scores = {
            pid: float(sum(prob_by_id[cid] for cid in group))
            for pid, group in members.items()
        }
        best_partition: Optional[str] = None
        best_sum = 0.0
        best_excess = 0.0
        for pid, total in partition_scores.items():
            size = len(members[pid])
            excess = total - size / n  # concentration above uniform share
            if 1 < size < n and excess > best_excess:
                best_partition, best_sum, best_excess = pid, total, excess

        max_fraction = float(self.cfg.calibration.max_entropy_fraction)
        omega = min(entropy_fraction, max_fraction)
        if max_fraction - entropy_fraction <= 1e-12:
            omega = max_fraction  # numerical vacuity: treat as all-Omega
        committed = 1.0 - omega
        assignments: Dict[object, float] = {}
        if committed > 1e-12:
            subset_share = 0.0
            if best_partition is not None:
                size = len(members[best_partition])
                subset_share = max(0.0, min(1.0, best_sum - size / n))
            subset_mass = committed * subset_share
            if subset_mass > 1e-12 and best_partition is not None:
                assignments[frozenset(members[best_partition])] = subset_mass
            singleton_total = committed - subset_mass
            for cid, prob in prob_by_id.items():
                value = singleton_total * float(prob)
                if value > 1e-12:
                    assignments[cid] = value
        if omega > 1e-12:
            assignments[tuple(ids)] = omega
        mass = make_mass(frame, assignments)
        return self._build_resolution(
            frame,
            mass,
            members=members,
            best_partition=best_partition,
            entropy_before=entropy_before,
            entropy_after=entropy,
        )

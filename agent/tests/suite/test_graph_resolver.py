"""Baseline resolver tests (ZYDAH-114): scores -> DS masses, offline."""

import pytest

from zydah.graph import build_comparison_graph, build_place_graph
from zydah.graph_resolver import (
    STATUS_RESOLVED,
    STATUS_SUBSET,
    STATUS_UNRESOLVED,
    NumpyResolver,
    Resolver,
)


def test_no_context_springfield_is_all_omega():
    res = NumpyResolver().resolve(build_place_graph())
    assert res.status == STATUS_UNRESOLVED
    assert res.mass[res.frame] == pytest.approx(1.0)
    assert sum(res.mass.values()) == pytest.approx(1.0)


def test_timezone_context_yields_subset_mass_not_singleton():
    graph = build_place_graph(context={"timezone": "America/Chicago"})
    res = NumpyResolver().resolve(graph)
    assert res.status == STATUS_SUBSET
    assert sum(res.mass.values()) == pytest.approx(1.0)
    subset_focals = [f for f in res.mass if 1 < len(f) < len(res.frame)]
    assert subset_focals  # a partition earned focal mass
    members = set(subset_focals[0])
    assert "candidate:springfield:IL" in members
    assert "candidate:springfield:MO" in members
    assert "candidate:springfield:OR" not in members


def test_faster_than_edges_resolve_bird_horse():
    res = NumpyResolver().resolve(build_comparison_graph("bird", "horse"))
    top = max(res.candidate_scores, key=res.candidate_scores.get)
    assert top == "candidate:peregrine-falcon"
    assert res.status == STATUS_RESOLVED
    assert sum(res.mass.values()) == pytest.approx(1.0)


def test_car_scores_follow_spec_edges():
    res = NumpyResolver().resolve(build_comparison_graph("BMW", "Nissan"))
    top = max(res.candidate_scores, key=res.candidate_scores.get)
    assert top == "candidate:nissan-gt-r"  # 196 mph, fastest in graph
    assert sum(res.mass.values()) == pytest.approx(1.0)


def test_no_context_intervals_are_vacuous_not_uniform_belief():
    # All-Omega mass: pignistic is uniform (a betting transform), but
    # every singleton's credal interval is the vacuous [0, 1].
    res = NumpyResolver().resolve(build_place_graph())
    for cid in res.frame:
        bel, pl = res.candidate_intervals[cid]
        assert bel == pytest.approx(0.0)
        assert pl == pytest.approx(1.0)
        assert res.candidate_scores[cid] == pytest.approx(0.125)
        # (A, ~A, Omega): nothing committed for or against, all
        # uncommitted. The triple always sums to 1.
        a, not_a, omega = res.candidate_beliefs[cid]
        assert (a, not_a, omega) == pytest.approx((0.0, 0.0, 1.0))


def test_conditioning_on_partition_redistributes_weights():
    resolver = NumpyResolver()
    prior = resolver.resolve(build_place_graph())
    north = {
        "candidate:springfield:IL",
        "candidate:springfield:MA",
        "candidate:springfield:OH",
        "candidate:springfield:OR",
    }
    conditioned = resolver.condition(prior, north, alpha=0.75)
    assert conditioned.mass[frozenset(north)] == pytest.approx(0.75)
    assert conditioned.mass[conditioned.frame] == pytest.approx(0.25)
    # North singletons gain plausibility structure via the subset focal
    # element; their pignistic share rises above the southern share.
    assert conditioned.candidate_scores["candidate:springfield:MA"] == pytest.approx(
        0.75 / 4 + 0.25 / 8
    )
    assert conditioned.candidate_scores["candidate:springfield:GA"] == pytest.approx(
        0.25 / 8
    )
    assert conditioned.candidate_intervals["candidate:springfield:GA"][1] == pytest.approx(0.25)


def test_yager_conditioning_sends_conflict_to_omega():
    resolver = NumpyResolver()
    north = {
        "candidate:springfield:IL",
        "candidate:springfield:MA",
        "candidate:springfield:OH",
        "candidate:springfield:OR",
    }
    south = set(resolver.resolve(build_place_graph()).frame) - north
    conditioned = resolver.condition(
        resolver.condition(resolver.resolve(build_place_graph()), north, alpha=1.0),
        south,
        alpha=1.0,
    )
    # Total conflict under Yager becomes vacuous Omega mass, never an
    # empty-set entry and never a renormalized certainty.
    assert frozenset() not in conditioned.mass
    assert conditioned.mass[conditioned.frame] == pytest.approx(1.0)
    assert conditioned.status == STATUS_UNRESOLVED


def test_resolver_satisfies_protocol():
    assert isinstance(NumpyResolver(), Resolver)

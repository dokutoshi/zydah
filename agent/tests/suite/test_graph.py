"""Graph schema tests (ZYDAH-113): builders emit documented structure."""

import pytest

from zydah.graph import (
    AFFINITY,
    AMBIGUITY,
    CANDIDATE,
    FASTER_THAN,
    MEMBER_OF,
    PARTITION,
    build_comparison_graph,
    build_place_graph,
    load_graph_config,
)


def test_place_graph_root_partitions_candidates():
    graph = build_place_graph()
    roots = graph.nodes_of_kind(AMBIGUITY)
    assert [node.label for node in roots] == ["Springfield (?)"]
    candidates = graph.nodes_of_kind(CANDIDATE)
    assert {c.features["state"] for c in candidates} == {
        "IL", "MO", "MA", "OH", "OR", "TN", "GA", "VA",
    }
    directions = {
        node.label
        for node in graph.nodes_of_kind(PARTITION)
        if node.features.get("axis") == "direction"
    }
    assert directions == {"North", "South", "East", "West"}
    il = graph.node("candidate:springfield:IL")
    member_targets = {
        graph.node(edge.target).label
        for edge in graph.edges_of_kind(MEMBER_OF)
        if edge.source == il.id
    }
    assert "North" in member_targets and "West" in member_targets
    assert "America/Chicago" in member_targets


def test_place_graph_thresholds_come_from_yaml():
    cfg = load_graph_config()
    assert float(cfg.place.north_south_latitude) == pytest.approx(39.0)


def test_place_graph_candidates_carry_state_names():
    graph = build_place_graph()
    massachusetts = graph.node("candidate:springfield:MA")
    assert massachusetts.features["state_name"] == "Massachusetts"


def test_place_graph_context_affinity_edges():
    graph = build_place_graph(context={"timezone": "America/Chicago"})
    affinities = graph.edges_of_kind(AFFINITY)
    assert affinities
    assert any(
        edge.target == "candidate:springfield:IL" for edge in affinities
    )


def test_comparison_graph_cars_faster_than_edges():
    graph = build_comparison_graph("BMW", "Nissan")
    assert len(graph.nodes_of_kind(AMBIGUITY)) == 2
    edges = graph.edges_of_kind(FASTER_THAN)
    pairs = {(edge.source, edge.target) for edge in edges}
    # GT-R (196) beats M4 (180) on top speed; M4 beats Versa (122).
    assert ("candidate:nissan-gt-r", "candidate:bmw-m4") in pairs
    assert ("candidate:bmw-m4", "candidate:nissan-versa") in pairs
    classes = {
        node.label
        for node in graph.nodes_of_kind(PARTITION)
        if node.features.get("axis") == "model_class"
    }
    assert classes == {"sports", "sedan", "economy"}


def test_comparison_graph_bird_horse_interpretations():
    graph = build_comparison_graph("bird", "horse")
    edges = graph.edges_of_kind(FASTER_THAN)
    pairs = {(edge.source, edge.target) for edge in edges}
    # Fastest-member interpretation resolves in-graph: falcon > horse.
    assert ("candidate:peregrine-falcon", "candidate:thoroughbred") in pairs
    interpretations = {
        node.label
        for node in graph.nodes_of_kind(PARTITION)
        if node.features.get("axis") == "interpretation"
    }
    assert interpretations == {"fastest_member", "typical_member"}


def test_comparison_graph_unknown_term_raises():
    with pytest.raises(ValueError, match="no candidate dataset"):
        build_comparison_graph("BMW", "toaster")

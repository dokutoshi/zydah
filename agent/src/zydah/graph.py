"""Candidate graph schema and builders (ZYDAH-113).

A graph turns an ambiguous term into the structure a resolver scores:

- ``ambiguity``: root supernode for the ambiguous term (``Springfield (?)``)
- ``partition``: named splits over the candidates (North/South/East/West,
  timezone; for comparisons: model class, metric, interpretation)
- ``candidate``: concrete resolutions (Springfield IL, BMW M4, falcon)
- ``context``: caller/session signals with affinity edges into the graph

Edges are typed: ``candidate_of``, ``member_of``, ``faster_than``,
``affinity``. This module has no ML-framework dependency; the
deterministic baseline (ZYDAH-114) and the learned scorer (ZYDAH-117)
both consume these structures through the Resolver protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from omegaconf import OmegaConf

_CONFIG_PATH = Path(__file__).parent / "conf" / "graph.yaml"

AMBIGUITY = "ambiguity"
PARTITION = "partition"
CANDIDATE = "candidate"
CONTEXT = "context"

CANDIDATE_OF = "candidate_of"
MEMBER_OF = "member_of"
FASTER_THAN = "faster_than"
AFFINITY = "affinity"


def load_graph_config():
    return OmegaConf.load(_CONFIG_PATH)


@dataclass(frozen=True)
class GraphNode:
    id: str
    kind: str
    label: str
    features: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    kind: str
    features: Mapping[str, object] = field(default_factory=dict)


@dataclass
class CandidateGraph:
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)

    def add_node(self, node: GraphNode) -> GraphNode:
        self.nodes.append(node)
        return node

    def add_edge(self, edge: GraphEdge) -> GraphEdge:
        self.edges.append(edge)
        return edge

    def nodes_of_kind(self, kind: str) -> List[GraphNode]:
        return [node for node in self.nodes if node.kind == kind]

    def edges_of_kind(self, kind: str) -> List[GraphEdge]:
        return [edge for edge in self.edges if edge.kind == kind]

    def node(self, node_id: str) -> GraphNode:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(node_id)


# Fixture metadata (approximate public figures; builders treat them as
# data, never as computed truth).

SPRINGFIELDS: Sequence[Mapping[str, object]] = (
    {"state": "IL", "lat": 39.78, "lon": -89.65, "population": 114394, "timezone": "America/Chicago"},
    {"state": "MO", "lat": 37.21, "lon": -93.29, "population": 169176, "timezone": "America/Chicago"},
    {"state": "MA", "lat": 42.10, "lon": -72.59, "population": 154064, "timezone": "America/New_York"},
    {"state": "OH", "lat": 39.92, "lon": -83.81, "population": 58662, "timezone": "America/New_York"},
    {"state": "OR", "lat": 44.05, "lon": -123.02, "population": 61984, "timezone": "America/Los_Angeles"},
    {"state": "TN", "lat": 36.50, "lon": -86.88, "population": 18731, "timezone": "America/Chicago"},
    {"state": "GA", "lat": 32.37, "lon": -81.31, "population": 2703, "timezone": "America/New_York"},
    {"state": "VA", "lat": 38.79, "lon": -77.18, "population": 31813, "timezone": "America/New_York"},
)

VEHICLES: Mapping[str, Sequence[Mapping[str, object]]] = {
    "bmw": (
        {"model": "M4", "top_speed_mph": 180.0, "zero_to_sixty_s": 3.4, "model_class": "sports"},
        {"model": "320i", "top_speed_mph": 130.0, "zero_to_sixty_s": 7.1, "model_class": "sedan"},
    ),
    "nissan": (
        {"model": "GT-R", "top_speed_mph": 196.0, "zero_to_sixty_s": 2.9, "model_class": "sports"},
        {"model": "Versa", "top_speed_mph": 122.0, "zero_to_sixty_s": 9.5, "model_class": "economy"},
    ),
}

ANIMALS: Mapping[str, Sequence[Mapping[str, object]]] = {
    "bird": (
        {"species": "Peregrine falcon", "top_speed_mph": 240.0, "locomotion": "air", "typical": False},
        {"species": "Sparrow", "top_speed_mph": 25.0, "locomotion": "air", "typical": True},
        {"species": "Ostrich", "top_speed_mph": 43.0, "locomotion": "ground", "typical": False},
    ),
    "horse": (
        {"species": "Thoroughbred", "top_speed_mph": 55.0, "locomotion": "ground", "typical": True},
    ),
}


def _context_nodes(graph: CandidateGraph, context: Optional[Mapping[str, object]]) -> None:
    for key, value in (context or {}).items():
        node_id = f"context:{key}"
        graph.add_node(GraphNode(node_id, CONTEXT, str(key), {str(key): value}))
        # Affinity into any partition/candidate whose features match.
        for other in list(graph.nodes):
            if other.kind == CONTEXT:
                continue
            if value in other.features.values() or str(value) == other.label:
                graph.add_edge(GraphEdge(node_id, other.id, AFFINITY, {"signal": str(key)}))


def build_place_graph(
    term: str = "Springfield",
    candidates: Sequence[Mapping[str, object]] = SPRINGFIELDS,
    context: Optional[Mapping[str, object]] = None,
    config=None,
) -> CandidateGraph:
    """Place graph: ambiguity root -> quadrant/timezone partitions -> cities."""
    cfg = config or load_graph_config()
    lat_split = float(cfg.place.north_south_latitude)
    lon_split = float(cfg.place.east_west_longitude)
    graph = CandidateGraph()
    root_id = f"ambiguity:{term.lower()}"
    graph.add_node(GraphNode(root_id, AMBIGUITY, f"{term} (?)", {"term": term}))
    partitions: Dict[str, str] = {}
    for name in ("North", "South", "East", "West"):
        pid = f"partition:direction:{name.lower()}"
        partitions[name] = pid
        graph.add_node(GraphNode(pid, PARTITION, name, {"axis": "direction"}))
    tz_ids: Dict[str, str] = {}
    for cand in candidates:
        tz = str(cand["timezone"])
        if tz not in tz_ids:
            pid = f"partition:timezone:{tz}"
            tz_ids[tz] = pid
            graph.add_node(GraphNode(pid, PARTITION, tz, {"axis": "timezone"}))
        cid = f"candidate:{term.lower()}:{cand['state']}"
        features = dict(cand)
        state_name = OmegaConf.select(cfg, f"place.state_names.{cand['state']}")
        if state_name is not None:
            features["state_name"] = str(state_name)
        graph.add_node(
            GraphNode(cid, CANDIDATE, f"{term}, {cand['state']}", features)
        )
        graph.add_edge(GraphEdge(root_id, cid, CANDIDATE_OF))
        ns = "North" if float(cand["lat"]) >= lat_split else "South"
        ew = "East" if float(cand["lon"]) >= lon_split else "West"
        graph.add_edge(GraphEdge(cid, partitions[ns], MEMBER_OF, {"axis": "direction"}))
        graph.add_edge(GraphEdge(cid, partitions[ew], MEMBER_OF, {"axis": "direction"}))
        graph.add_edge(GraphEdge(cid, tz_ids[tz], MEMBER_OF, {"axis": "timezone"}))
    _context_nodes(graph, context)
    return graph


def build_comparison_graph(
    left: str,
    right: str,
    context: Optional[Mapping[str, object]] = None,
) -> CandidateGraph:
    """Comparison graph: two term roots, candidates, faster-than edges.

    Vehicle terms use spec attributes (top speed, 0-60, model class);
    animal terms use locomotion and fastest-vs-typical interpretations.
    """
    graph = CandidateGraph()
    datasets = []
    for term in (left, right):
        key = term.lower()
        if key in VEHICLES:
            entries = [
                {"name": f"{term} {e['model']}", **e} for e in VEHICLES[key]
            ]
            datasets.append(entries)
        elif key in ANIMALS:
            entries = [
                {"name": str(e["species"]), **e} for e in ANIMALS[key]
            ]
            datasets.append(entries)
        else:
            raise ValueError(f"no candidate dataset for term {term!r}")
        root_id = f"ambiguity:{key}"
        graph.add_node(GraphNode(root_id, AMBIGUITY, f"{term} (?)", {"term": term}))
        for entry in entries:
            cid = f"candidate:{entry['name'].lower().replace(' ', '-')}"
            graph.add_node(GraphNode(cid, CANDIDATE, str(entry["name"]), dict(entry)))
            graph.add_edge(GraphEdge(root_id, cid, CANDIDATE_OF))
    # Partition nodes from attributes present in the combined candidates.
    candidates = graph.nodes_of_kind(CANDIDATE)
    for axis, values in (
        ("model_class", {c.features.get("model_class") for c in candidates}),
        ("locomotion", {c.features.get("locomotion") for c in candidates}),
        ("interpretation", {"fastest_member", "typical_member"} if any("typical" in c.features for c in candidates) else set()),
    ):
        for value in sorted(v for v in values if v):
            pid = f"partition:{axis}:{value}"
            graph.add_node(GraphNode(pid, PARTITION, str(value), {"axis": axis}))
            for cand in candidates:
                if axis == "interpretation":
                    member = (value == "fastest_member") != bool(cand.features.get("typical", False))
                    if cand.features.get("typical") is None and value != "typical_member":
                        member = False
                    if member:
                        graph.add_edge(GraphEdge(cand.id, pid, MEMBER_OF, {"axis": axis}))
                elif cand.features.get(axis) == value:
                    graph.add_edge(GraphEdge(cand.id, pid, MEMBER_OF, {"axis": axis}))
    # Faster-than edges from the shared numeric metric.
    for a in candidates:
        for b in candidates:
            if a is b:
                continue
            a_speed = a.features.get("top_speed_mph")
            b_speed = b.features.get("top_speed_mph")
            if a_speed is not None and b_speed is not None and float(a_speed) > float(b_speed):
                graph.add_edge(
                    GraphEdge(a.id, b.id, FASTER_THAN, {"metric": "top_speed_mph", "margin": float(a_speed) - float(b_speed)})
                )
    _context_nodes(graph, context)
    return graph

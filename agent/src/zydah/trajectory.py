"""Bellman trajectory schema: state bucketing + (s, a, r, s') rows.

A trajectory row is one verifier iteration, logged as JSONL by the
verifier. The terminal row of an episode is the last row for its
``episode_id``; offline tooling assigns terminal reward from labeled
references (verdict is not correctness). Bucket edges and costs come
from ``conf/bellman.yaml`` — never literals in code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from omegaconf import DictConfig, OmegaConf

_CONFIG_PATH = Path(__file__).parent / "conf" / "bellman.yaml"

ACTION_COMMIT = "commit"
ACTION_MODEL_FOLLOWUP = "model-followup"
ACTION_ASK_HUMAN = "ask-human"
ACTION_PANEL = "re-elicit-panel"


def load_bellman_config() -> DictConfig:
    return OmegaConf.load(_CONFIG_PATH)


def _bucket(value: float, edges: Sequence[float]) -> int:
    idx = 0
    for edge in edges:
        if value >= float(edge):
            idx += 1
    return idx


def bucket_state(
    state: Mapping[str, object], cfg: Optional[DictConfig] = None
) -> Tuple[int, int, int, int]:
    """Discretize a raw state dict to the tabular Q state tuple.

    (width bucket, conflict bucket, follow-ups left, graph unresolved)
    """
    cfg = cfg or load_bellman_config()
    return (
        _bucket(float(state.get("width") or 0.0), list(cfg.width_edges)),
        _bucket(float(state.get("conflict") or 0.0), list(cfg.conflict_edges)),
        int(state.get("followups_left") or 0),
        1 if state.get("graph_unresolved") else 0,
    )


def state_key(state: Tuple[int, ...]) -> str:
    return ",".join(str(int(part)) for part in state)


def parse_state_key(key: str) -> Tuple[int, ...]:
    return tuple(int(part) for part in key.split(","))


def trajectory_row(
    *,
    episode_id: str,
    step: int,
    state: Mapping[str, object],
    action: str,
    policy: str,
    model_id: Optional[str],
    usage: Optional[Mapping[str, object]] = None,
    cfg: Optional[DictConfig] = None,
) -> Dict[str, object]:
    """One (s, a) row; reward/next-state are chained offline by step."""
    cfg = cfg or load_bellman_config()
    return {
        "event": "trajectory",
        "episode_id": episode_id,
        "step": int(step),
        "state": dict(state),
        "bucket": list(bucket_state(state, cfg)),
        "action": action,
        "policy": policy,
        "model_id": model_id,
        "usage": dict(usage) if usage else None,
    }

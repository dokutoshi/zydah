"""Offline temporal-difference TD(0)/Monte-Carlo learner over trajectories.

Ports the ``update_td`` temporal-difference update verbatim from SAB
Blackjack onto Zydah's discretized routing states. The Q function stays tabular (dict state
key -> per-action values); Bellman/gamma/lr/costs come from
``conf/bellman.yaml``. Episodes are capped JSONL logs; chaining is
sorted by (episode_id, step) and the last row of an episode is
terminal. Reward is attached offline: +1/-1 for a correct/incorrect
terminal commit, minus ``lambda_cost * action_cost`` per step. Live
routing is untouched — this module only reads logs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from omegaconf import DictConfig

from zydah.trajectory import (
    ACTION_ASK_HUMAN,
    ACTION_COMMIT,
    load_bellman_config,
    state_key,
)


class QTable:
    """Tabular Q: dict ``state_key -> {action: float}``."""

    def __init__(
        self,
        actions: Sequence[str],
        table: Optional[Mapping[str, Mapping[str, float]]] = None,
    ) -> None:
        self.actions: List[str] = list(actions)
        self.table: Dict[str, Dict[str, float]] = {
            key: {a: float(values.get(a, 0.0)) for a in self.actions}
            for key, values in (table or {}).items()
        }

    def ensure(self, state: Tuple[int, ...]) -> Dict[str, float]:
        return self.table.setdefault(
            state_key(state), {a: 0.0 for a in self.actions}
        )

    def q(self, state: Tuple[int, ...], action: str) -> float:
        return self.ensure(state).get(action, 0.0)

    def max_q(self, state: Tuple[int, ...]) -> float:
        return max(self.ensure(state).values())

    def greedy(self, state: Tuple[int, ...]) -> str:
        """Greedy action, with an ask-human fallback for no evidence.

        An unseen state, or a visited state whose best Q is not
        positive, has no learned basis for committing. When ask-human
        is an available action it is the safe fallback; action order
        alone must never turn an untrained bucket into a commit.
        """
        key = state_key(state)
        values = self.table.get(key)
        if values is None or max(values.values(), default=0.0) <= 0.0:
            if ACTION_ASK_HUMAN in self.actions:
                return ACTION_ASK_HUMAN
        return greedy_action(self.q_values(state), self.actions)

    def q_values(self, state: Tuple[int, ...]) -> Dict[str, float]:
        return dict(self.ensure(state))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"actions": self.actions, "table": self.table}, indent=2)
        )

    @classmethod
    def load(cls, path: Path) -> "QTable":
        data = json.loads(path.read_text())
        return cls(list(data["actions"]), data["table"])

    def __len__(self) -> int:
        return len(self.table)


def greedy_action(values: Mapping[str, float], actions: Sequence[str]) -> str:
    """Argmax with deterministic first-action tie-break."""
    best, best_value = None, None
    for action in actions:
        value = float(values.get(action, 0.0))
        if best_value is None or value > best_value:
            best, best_value = action, value
    assert best is not None
    return best


def update_td(
    q: QTable,
    state: Tuple[int, ...],
    action: str,
    reward: float,
    next_state: Optional[Tuple[int, ...]],
    gamma: float,
    alpha: float,
    terminal: bool = False,
) -> float:
    """TD(0): ``r + gamma * maxQ(s')``; terminal rows bootstrap from 0."""
    target = reward if terminal else reward + gamma * q.max_q(next_state)  # type: ignore[arg-type]
    current = q.q(state, action)
    new_value = current + alpha * (target - current)
    q.ensure(state)[action] = new_value
    return new_value


def update_mc(
    q: QTable,
    state: Tuple[int, ...],
    action: str,
    gain: float,
    alpha: float,
) -> float:
    """Every-visit MC: ``Q <- Q + alpha * (G - Q)``."""
    current = q.q(state, action)
    new_value = current + alpha * (gain - current)
    q.ensure(state)[action] = new_value
    return new_value


def load_episodes(path: Path) -> List[List[Dict[str, object]]]:
    """Group JSONL trajectory rows into episodes sorted by step."""
    rows: List[Dict[str, object]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("event") == "trajectory":
                rows.append(row)
    episodes: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        episodes.setdefault(str(row["episode_id"]), []).append(row)
    return [sorted(rs, key=lambda r: int(r["step"])) for rs in episodes.values()]


def action_cost(action: str, cfg: DictConfig) -> float:
    return float(cfg.action_costs[action])


def step_reward(row: Mapping[str, object], cfg: DictConfig) -> float:
    return -float(cfg.lambda_cost) * action_cost(str(row["action"]), cfg)


def train(
    episodes: List[List[Dict[str, object]]],
    outcomes: Mapping[str, int],
    method: str = "td",
    cfg: Optional[DictConfig] = None,
) -> QTable:
    """Train one pass over episodes.

    ``outcomes`` maps episode_id -> +1/-1 for the terminal commit's
    correctness. Episodes without an outcome contribute only step
    costs (terminal reward 0). Rows whose episode ended without a
    commit get no terminal reward.
    """
    cfg = cfg or load_bellman_config()
    gamma = float(cfg.gamma)
    alpha = float(cfg.learning_rate)
    q = QTable(list(cfg.actions))
    for rows in episodes:
        if not rows:
            continue
        episode_id = str(rows[0]["episode_id"])
        terminal_bonus = 0.0
        if rows[-1]["action"] == ACTION_COMMIT:
            terminal_bonus = float(outcomes.get(episode_id, 0))
        if method == "mc":
            gain = 0.0
            for row in reversed(rows):
                reward = step_reward(row, cfg)
                if row is rows[-1]:
                    reward += terminal_bonus
                gain = reward + gamma * gain
                state = tuple(int(x) for x in row["bucket"])  # type: ignore[arg-type]
                update_mc(q, state, str(row["action"]), gain, alpha)
            continue
        for index, row in enumerate(rows):
            reward = step_reward(row, cfg)
            terminal = index == len(rows) - 1
            if terminal:
                reward += terminal_bonus
            state = tuple(int(x) for x in row["bucket"])  # type: ignore[arg-type]
            next_state = (
                None
                if terminal
                else tuple(int(x) for x in rows[index + 1]["bucket"])  # type: ignore[arg-type]
            )
            update_td(q, state, str(row["action"]), reward, next_state, gamma, alpha, terminal)
    return q

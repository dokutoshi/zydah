"""Synthetic Bellman smoke check (not policy evidence).

Static harness items (``{"id", "reference", "runs": [[Mass, ...]]}``)
give single-step episodes: the fused state is observed, one routing
action is taken, and correctness comes from the item's reference label
(the independent oracle — never the verifier's verdict). Follow-up
actions cannot change static masses, so non-commit actions abstain at
their configured cost. This fixture exercises learner arithmetic on a
separable synthetic set only; it does not show that temporal-difference
routing is learnable or better than the threshold policy. Multi-step
episodes need live trajectory logs (``outputs/live_demo/cycles.jsonl``).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from zydah.fusion import make_mass
from zydah.td import QTable, load_episodes, train
from zydah.trajectory import (
    ACTION_ASK_HUMAN,
    ACTION_COMMIT,
    bucket_state,
    load_bellman_config,
)
from zydah.verifier import load_verifier_config

from eval.harness import zydah_verdict

FRAME = frozenset({"pass", "partial", "fail"})
OMEGA = ("pass", "partial", "fail")


def fused_state(item: Mapping[str, object]) -> Dict[str, object]:
    """Raw + bucketed routing state from an item's first run of masses."""
    verdict = zydah_verdict(list(item["runs"][0]), FRAME)  # type: ignore[arg-type]
    raw = {
        "width": verdict["width"],
        "conflict": verdict["conflict"],
        "bel": verdict["bel"],
        "pl": verdict["pl"],
        "best": verdict["verdict"],
        "decision": verdict["decision"],
        "followups_left": 0,
        "graph_unresolved": False,
    }
    return {"raw": raw, "bucket": bucket_state(raw), "best": verdict["verdict"]}


def pi0_action(state: Mapping[str, object]) -> str:
    """Current hand-written gate: commit iff width/conflict pass YAML."""
    cfg = load_verifier_config()
    if (
        float(state["width"]) <= float(cfg.width_threshold)
        and float(state["conflict"]) <= float(cfg.conflict_threshold)
    ):
        return ACTION_COMMIT
    return ACTION_ASK_HUMAN


def episode_row(item: Mapping[str, object], action: str) -> Dict[str, object]:
    info = fused_state(item)
    return {
        "event": "trajectory",
        "episode_id": str(item["id"]),
        "step": 0,
        "state": info["raw"],
        "bucket": list(info["bucket"]),
        "action": action,
        "policy": "pi_0",
        "model_id": None,
        "usage": None,
    }


def outcome(item: Mapping[str, object], action: str) -> Optional[int]:
    """+1/-1 when a commit's best label matches the reference, else None."""
    if action != ACTION_COMMIT:
        return None
    best = fused_state(item)["best"]
    return 1 if best == item["reference"] else -1


def evaluate(
    choose_action: Callable[[Mapping[str, object]], str],
    items: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    cfg = load_bellman_config()
    committed = correct = escalations = 0
    total_cost = 0.0
    for item in items:
        info = fused_state(item)
        action = choose_action(info["raw"])
        total_cost += float(cfg.action_costs[action])
        if action == ACTION_COMMIT:
            committed += 1
            if info["best"] == item["reference"]:
                correct += 1
        else:
            escalations += 1
    n = len(items)
    return {
        "n": n,
        "committed": committed,
        "coverage": committed / n if n else 0.0,
        "accuracy_committed": (correct / committed) if committed else 0.0,
        "escalate_rate": escalations / n if n else 0.0,
        "mean_cost": total_cost / n if n else 0.0,
    }


def compare_policies(
    items: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """Train TD/MC on even-index items, evaluate all policies on odd."""
    train_items = [item for i, item in enumerate(items) if i % 2 == 0]
    test_items = [item for i, item in enumerate(items) if i % 2 == 1]
    episodes: List[List[Dict[str, object]]] = []
    outcomes: Dict[str, int] = {}
    for item in train_items:
        action = pi0_action(fused_state(item)["raw"])
        episodes.append([episode_row(item, action)])
        result = outcome(item, action)
        if result is not None:
            outcomes[str(item["id"])] = result
    cfg = load_bellman_config()
    q_td = train(episodes, outcomes, method="td", cfg=cfg)
    q_mc = train(episodes, outcomes, method="mc", cfg=cfg)

    def greedy(q: QTable) -> Callable[[Mapping[str, object]], str]:
        return lambda state: q.greedy(bucket_state(state, cfg))

    return {
        "train_n": len(train_items),
        "test_n": len(test_items),
        "q_states_td": len(q_td),
        "q_states_mc": len(q_mc),
        "pi_0": evaluate(pi0_action, test_items),
        "td": evaluate(greedy(q_td), test_items),
        "mc": evaluate(greedy(q_mc), test_items),
    }


def synthetic_items(n: int = 40, seed: int = 0) -> List[Dict[str, object]]:
    """Deterministic separable smoke fixture (labels follow width bucket).

    Confident items (width in the lowest bucket) carry a wrong reference;
    hedged items (width in the next bucket) carry a right one. Because
    the label is a function of the bucket, separation is by construction
    and proves arithmetic only, not learnability.
    """
    rng = random.Random(seed)
    items: List[Dict[str, object]] = []
    for i in range(n):
        confident_wrong = i % 2 == 0
        if confident_wrong:
            mass = make_mass(FRAME, {"pass": 0.95, OMEGA: 0.05})
            reference = "fail"
        else:
            mass = make_mass(FRAME, {"pass": 0.65, OMEGA: 0.35})
            reference = "pass"
        items.append(
            {"id": f"synthetic-{i}", "reference": reference, "runs": [[mass]]}
        )
    rng.shuffle(items)
    return items


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the synthetic Bellman smoke check (not policy evidence).")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = compare_policies(synthetic_items(n=args.n, seed=args.seed))
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()

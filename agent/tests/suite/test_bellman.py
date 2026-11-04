"""Bellman v2: trajectory schema, bucketing, TD/MC learner, persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zydah.td import QTable, greedy_action, load_episodes, train, update_mc, update_td
from zydah.trajectory import bucket_state, load_bellman_config, state_key

# Golden Q after the reference update sequence (hand-checked against
# the SAB Blackjack TD arithmetic: 0.05, then 0.05 + 0.05*(-1.05) = -0.0025).
GOLDEN_Q = {
    "0,0,0,0": {"ask-human": 0.05, "commit": 0.0, "model-followup": 0.0, "re-elicit-panel": 0.0},
    "3,3,1,0": {"ask-human": 0.0, "commit": -0.0025000000000000022, "model-followup": 0.0, "re-elicit-panel": 0.0},
}


def _row(action: str, bucket=(3, 3, 1, 0), step: int = 0, ep: str = "e1"):
    return {
        "event": "trajectory",
        "episode_id": ep,
        "step": step,
        "state": {"width": 0.9, "conflict": 0.9},
        "bucket": list(bucket),
        "action": action,
        "policy": "pi_0",
        "model_id": "mock",
        "usage": None,
    }


def _golden_updates(gamma: float = 0.95, alpha: float = 0.05):
    """SAB Blackjack reference ops on Zydah states: reuse, not analogy."""
    cfg = load_bellman_config()
    assert float(cfg.gamma) == gamma and float(cfg.learning_rate) == alpha
    q = QTable(list(cfg.actions))
    s, s2 = (3, 3, 1, 0), (0, 0, 0, 0)
    update_td(q, s, "commit", 1.0, s2, gamma, alpha)  # bootstrap
    update_td(q, s, "commit", -1.0, None, gamma, alpha, terminal=True)
    update_td(q, s2, "ask-human", 1.0, None, gamma, alpha, terminal=True)
    return {k: dict(v) for k, v in q.table.items()}


def test_verifier_emits_trajectory_rows(tmp_path: Path):
    from zydah.fusion import make_mass
    from zydah.verifier import ZydahVerifier

    frame = frozenset({"pass", "partial", "fail"})
    omega = ("pass", "partial", "fail")

    log_path = tmp_path / "cycles.jsonl"
    verifier = ZydahVerifier(
        frame,
        mass_elicitor=lambda view, fr, model: make_mass(fr, {"pass": 0.9, omega: 0.1}),
        log_path=log_path,
    )
    out = verifier.verify_direct("Claim: it is cloudy in Springfield, IL today.")
    assert out["verdict"] == "good"
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    traj = [r for r in rows if r.get("event") == "trajectory"]
    assert len(traj) == 1
    row = traj[0]
    assert row["action"] == "commit" and row["policy"] == "pi_0"
    assert row["episode_id"] and row["step"] == 0
    assert row["bucket"] == [0, 0, row["bucket"][2], 0]
    assert row["state"]["width"] == pytest.approx(0.1)


def test_bucketing_edges():
    assert bucket_state({"width": 0.05, "conflict": 0.0, "followups_left": 2}) == (0, 0, 2, 0)
    assert bucket_state({"width": 0.35, "conflict": 0.9, "followups_left": 0, "graph_unresolved": True}) == (2, 3, 0, 1)
    assert state_key((2, 3, 0, 1)) == "2,3,0,1"


def test_td_reference_reused_not_analogy():
    assert _golden_updates() == GOLDEN_Q


def test_td_terminal_no_bootstrap_and_persistence(tmp_path: Path):
    cfg = load_bellman_config()
    episodes = [[_row("model-followup", step=0), _row("commit", step=1)]]
    q = train(episodes, {"e1": 1}, method="td", cfg=cfg)
    state = "3,3,1,0"
    # terminal commit: -cost(0) + 1 -> Q = 0 + lr*(1-0)
    assert q.table[state]["commit"] == pytest.approx(float(cfg.learning_rate))
    path = tmp_path / "q.json"
    q.save(path)
    loaded = QTable.load(path)
    assert loaded.table == q.table
    assert loaded.greedy((3, 3, 1, 0)) == "commit"


def test_mc_wrong_commit_negative_and_costs():
    cfg = load_bellman_config()
    episodes = [[_row("model-followup", step=0), _row("commit", step=1)]]
    q = train(episodes, {"e1": -1}, method="mc", cfg=cfg)
    state = "3,3,1,0"
    assert q.table[state]["commit"] < 0
    # model-followup pays lambda * 1 before the bad return
    assert q.table[state]["model-followup"] < 0


def test_greedy_unseen_and_zero_states_ask_human():
    cfg = load_bellman_config()
    q = QTable(list(cfg.actions))

    assert q.greedy((9, 9, 9, 9)) == "ask-human"
    q.ensure((0, 0, 0, 0))
    assert q.greedy((0, 0, 0, 0)) == "ask-human"


def test_td_bootstraps_from_nonzero_next_state():
    cfg = load_bellman_config()
    q = QTable(list(cfg.actions))
    state, next_state = (0, 0, 0, 0), (1, 0, 0, 0)
    q.ensure(next_state)["commit"] = 1.0

    update_td(q, state, "model-followup", 0.0, next_state, 0.95, 0.05)
    update_mc(q, state, "ask-human", 0.0, 0.05)

    assert q.q(state, "model-followup") == pytest.approx(0.0475)
    assert q.q(state, "ask-human") == 0.0


def test_greedy_tie_break_and_episode_loading(tmp_path: Path):
    cfg = load_bellman_config()
    assert greedy_action({"commit": 0.0, "ask-human": 0.0}, list(cfg.actions)) == "commit"
    path = tmp_path / "traj.jsonl"
    path.write_text(
        json.dumps(_row("commit", step=1, ep="e2")) + "\n"
        + json.dumps({"event": "other"}) + "\n"
        + json.dumps(_row("model-followup", step=0, ep="e2")) + "\n"
    )
    episodes = load_episodes(path)
    assert len(episodes) == 1 and [r["step"] for r in episodes[0]] == [0, 1]

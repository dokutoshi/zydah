# Zydah

Evidential belief fusion for agent evaluation. Replace single LLM-as-judge /
majority vote with a panel of judges fused via Dempster-Shafer theory, Yager's
rule, and Smets' transferable belief model (TBM). Conflict between judges is
preserved as a signal, not normalized away.

## Building

A Dempster-Shafer+ belief-fusion RL agent. System 1 is deterministic fusion,
and System 2 is an LLM reasoner that handles only the high-conflict cases.
Conflict mass is a routing signal to use, not noise to normalize away.
Dependability is tested with order-invariance checks, seed variance, and cost
versus an always-LLM baseline, all on synthetic data in a Gymnasium
environment.

## Looking for

Feedback on reward design for process-level signals, and anyone working on
agent reliability or reasoning under uncertainty. Happy to compare notes!

P.S. Two doctors, one patient... Dempster says tumor. 🙂

> **Provenance:** new work built during the event, Oct 27 – Nov 4, 2026.
> Public repo; synthetic or public data only.

## Thesis

- **Decision invariance under judge variability:** same evidence → same decision,
  even when individual judges vary run to run.
- **Escalate on evidence** (high conflict mass / wide [Bel, Pl] interval),
  not on a retry counter.
- **Decide on belief intervals** with per-domain thresholds
  (τ = 0.80 general, 0.95 high-stakes):
  commit if Bel ≥ τ, reject if Pl < τ, escalate if the interval straddles τ.

## Layout

- `agent/` — fusion core (`zydah/`), judge panel, router, eval harness, and
  tests. See [`agent/README.md`](agent/README.md) for details.

## Quickstart

```sh
cd agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

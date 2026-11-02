# Zydah — evidential belief fusion for agent evaluation

Replace single LLM-as-judge / majority vote with a panel of judges fused via
Dempster-Shafer theory, Yager's rule, and Smets' transferable belief model (TBM).
Conflict between judges is preserved as a signal, not normalized away.

> **Provenance:** new work built during the event, Oct 27 – Nov 4, 2026.
> Public repo; synthetic or public data only. No medical-advice use cases.
> See [Provenance](#provenance).

## Thesis

- **Decision invariance under judge variability:** same evidence → same decision,
  even when individual judges vary run to run.
- **Escalate on evidence** (high conflict mass / wide [Bel, Pl] interval),
  not on a retry counter.
- **Decide on belief intervals** with per-domain thresholds
  (τ = 0.80 general, 0.95 high-stakes):
  commit if Bel ≥ τ, reject if Pl < τ, escalate if the interval straddles τ.

## Provenance

- **Prior work:** evidential belief fusion research at Lotus Research (Vū-Mē,
  2014–2020), applied to fusing heterogeneous model signals for recommendation.
- **This project is new work (Oct 27 – Nov 4, 2026):** it applies evidential
  fusion to a present-day problem — evaluating and improving the reasoning of
  AI agents, individually and in teams.
- **No prior code is reused;** the fusion core is re-implemented from published
  methods (Dempster, Shafer, Yager, Smets). Everything judges see in this repo
  and demo was built during the event.

## Math (published sources only)

- Dempster's rule (Dempster 1967; Shafer 1976) — normalized conjunctive combination.
- Yager's rule (Yager 1987) — conflict mass K reassigned to Ω (ignorance).
- Smets' TBM (Smets & Kennes 1994) — unnormalized conjunctive combination;
  decision via the pignistic transform.
- Simultaneous n-ary combination (order-invariant by construction).

No proprietary stabilization parameters. Defaults are simple and documented
in [`zydah/fusion.py`](zydah/fusion.py).

## Judges (Meta models)

Heterogeneous panel via structured JSON masses incl. m(Ω):

- Muse Wonder (Meta Model API)
- Muse Spark (Meta Model API)
- Muse Spark Open via Baseten (`meta-models/muse-spark-open-1026`)

Keys come from environment variables (`META_API_KEY`, `BASETEN_API_KEY`).
Never commit secrets. Unit tests use a seeded mock panel — no API calls.

## Layout

- `zydah/fusion.py` — mass functions, Dempster / Yager / TBM, Bel, Pl, conflict K
- `zydah/judges.py` — panel interface, JSON mass parsing, mock + Meta clients
- `zydah/router.py` — interval-based decision + escalation policy
- `eval/harness.py` — majority vote vs. averaging vs. Dempster vs. Zydah
- `tests/` — Zadeh's paradox, order invariance, sum-to-1 (no API calls)

## Baselines to beat

Majority vote, score averaging, and multi-judge "juries" (Verga et al., 2024).

## Quickstart

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

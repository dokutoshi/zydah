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

- **Thesis work:** evidential belief fusion thesis at the University of
  Pennsylvania — the Wharton School and Penn Engineering — through the
  Executive Master's degree in Technology Management.
- **Post-thesis work:** evidential belief fusion research at Lotus Research
  (Vū-Mē, 2014–2020), applied to fusing heterogeneous model signals for
  recommendation.
- **This task (Oct 27 – Nov 4, 2026):** new work built during the event. It
  applies evidential fusion to a present-day problem — evaluating and
  improving the reasoning of AI agents, individually and in teams.
- **No prior code is reused;** the fusion core is re-implemented from
  published methods (Dempster, Shafer, Yager, Smets). Everything judges see
  in this repo and demo was built during the event.

## Math (published sources only)

- Dempster's rule (Dempster 1967; Shafer 1976) — normalized conjunctive combination.
- Yager's rule (Yager 1987) — conflict mass K reassigned to Ω (ignorance).
- Smets' TBM (Smets & Kennes 1994) — unnormalized conjunctive combination;
  decision via the pignistic transform.
- Simultaneous n-ary combination (order-invariant by construction).

No proprietary stabilization parameters. Defaults are simple and documented
in [`src/zydah/fusion.py`](src/zydah/fusion.py).

## Judges (Meta models)

Heterogeneous panel via structured JSON masses incl. m(Ω):

- Muse Wonder (Meta Model API)
- Muse Spark (Meta Model API)
- Muse Spark Open via Baseten (`meta-models/muse-spark-open-1026`)

Keys come from environment variables (`META_API_KEY`, `BASETEN_API_KEY`).
Never commit secrets. Unit tests use a seeded mock panel — no API calls.

## Layout

- `src/zydah/fusion.py` — mass functions, Dempster / Yager / TBM, Bel, Pl, conflict K
- `src/zydah/judges.py` — panel interface, JSON mass parsing, mock + Meta clients
- `src/zydah/router.py` — interval-based decision + escalation policy
- `src/zydah/interface.py` — [muse] direct / [zydah] fused pipe-through API
- `src/zydah/verifier.py` — System 2 verifier loop (Hydra/OmegaConf YAML config)
- `src/zydah/evidence.py` — provenance ledger + anti-eavesdropping session rule
- `eval/harness.py` — majority vote vs. averaging vs. Dempster vs. Zydah
- `tests/` — Zadeh's paradox, order invariance, sum-to-1 (no API calls)

## Baselines to beat

Majority vote, score averaging, and multi-judge "juries" (Verga et al., 2024).

## Quickstart

```
uv sync --extra dev
uv run pytest
```

## Live demo (interactive)

The demo talks to the configured live model (`.env` at the repo root),
shows the first response under `[muse]`, verifies it under `[zydah]`,
and pauses at the terminal whenever a human choice is needed:

```sh
uv run python tools/live_demo.py
uv run python tools/live_demo.py --question "What is the weather in Springfield today?"
uv run python tools/live_demo.py --save outputs/live_demo/latest.json
uv run python tools/live_demo.py --zydah --save outputs/live_demo/latest.json
uv run python tools/live_demo.py --verbose
uv run python tools/live_demo.py --log-file trace.log
uv run python tools/live_demo.py --score-mode substantive --log-file trace.log
```

At the prompts:

- `[you]` — the input buffer. Type and press Enter to send; Enter on
  an empty buffer sends nothing and redraws the prompt. After a
  verdict you are back at this buffer for the next question.
- `[you: choice key]` — type a choice key (e.g. `B` for North); `Other`
  then asks for free text (e.g. `the celtics are a great team`).
- `[you: answer]` — answer the model's own clarifying question.
- Typed text at the buffer right after a `[muse]` response is sent
  as a human direction before Zydah evaluates; `q` or Ctrl-C quits
  gracefully (`[zydah] bye`). Enter alone never quits.

`--log-file trace.log` captures `logging` INFO to a file via a
`FileHandler` — no stderr redirect, nothing logged to the screen:
every input line, model prompt/response, `asks_human` decision,
gate direction, and the exact evidence Zydah scores, so swallowed
or misrouted text can be traced end to end. `--verbose` alone logs
the same records to stderr; both flags together write to both.
Uncaught exceptions are logged with their traceback and echoed to
the terminal as one line: `[zydah] fatal: ... (see log)`.

`--score-mode` controls checkpoint scoring: `every` (the default)
scores each model response as it lands; `substantive` skips
pure-question responses and logs the explicit skip reason.

`--verbose` also captures Zydah's internal view — `[zydah]
...Thinking` semaphores, the follow-up transcript, and per-turn
belief frames — into the log, without showing it on the terminal.
`--zydah` controls only the terminal display of that same view.

`--zydah` shows Zydah's internal view: `[zydah] ...Thinking`
semaphores while scoring, the follow-up transcript, the belief
frame as it adjusts per turn, and any graph conditioning. Without
it, only the concluded `[zydah]` sentence and verdict line print.

`--save` mirrors the terminal into the JSON artifact tee-style: the
response appears in the terminal and in the file as it happens
(`transcript` streams live; the structured artifact — question,
response, verdict, turns, belief frames — lands on finish).

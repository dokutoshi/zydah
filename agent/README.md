# Zydah — evidential belief fusion for agent evaluation

Zydah checks an LLM's answer before you rely on it. System 1 is
deterministic belief fusion (Dempster-Shafer masses, Yager's rule);
System 2 is an LLM reasoner invoked only when System 1 flags a
response as speculative — a wide [Bel, Pl] interval or high conflict
mass $K$. Conflict is a routing signal, not noise to normalize away.

The demo is closed-world only: it relies on the model's trained state
and the evidence in the prompt. It does not browse or fetch.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). uv manages the virtualenv
and provisions the pinned Python 3.12 — do not use `pip` or
`python -m venv`.

```sh
uv sync --extra dev
uv run pytest
```

The test suite runs fully offline: no API keys, no network, no model
calls. The live examples below need model credentials in `.env`
(see “Configuration”).

One live-mode exception: the Springfield graph path uses a small
sentence-transformer encoder (MiniLM, `all-MiniLM-L6-v2`) as a
similarity cross-check. Its weights are a one-time download from
Hugging Face (~90 MB) into `~/.cache/huggingface`; after that first
load it runs fully offline. The tests inject a fake scorer and never
download anything.

## Try it (terminal)

All commands run from this directory (`agent/`).

**Example A — one judge (direct path, the default)**

```sh
uv run python tools/live_demo.py --question "a dog is a canine and a cat is a what?"
```

```text
[muse turn 1]
A cat is a feline.
[zydah turn 1] the response checks out.
verdict=good needs_human=False followups=0
```

One elicitor scores the response; Zydah fuses that single mass and
reports the verdict. This is the cheapest path.

**Example B — see Zydah's internal view**

```sh
uv run python tools/live_demo.py --zydah --question "a dog is a canine and a cat is a what?"
```

Adds `[zydah] ...Thinking` semaphores, the follow-up transcript, and
the belief frame as it adjusts per turn.

**Example C — the 3-judge panel**

```sh
uv run python tools/live_demo.py --zydah-panel --zydah --question "What is the weather in Springfield today?"
```

```text
...launching zydah panel
[zydah] judge ZJ-01 (seat muse-wonder) ...Thinking
[zydah] judge ZJ-02 (seat muse-spark) ...Thinking
[zydah] judge ZJ-03 (seat muse-spark-open) ...Thinking
                        Zydah Chamber
┃ tag   ┃ seat          ┃ fail ┃ partial ┃ pass ┃ Omega ┃
┃ ZJ-01 ┃ muse-wonder   ┃ ...  ┃ ...     ┃ ...  ┃ ...   ┃
┃ ZJ-02 ┃ muse-spark    ┃ ...  ┃ ...     ┃ ...  ┃ ...   ┃
┃ ZJ-03 ┃ muse-spark-open┃ ... ┃ ...     ┃ ...  ┃ ...   ┃
```

Three judges each return a mass; the masses are Shafer-discounted by
$\alpha$ and fused by Yager's rule (quorum 2 of 3). The fused
Bel/Pl/width/conflict prints under the table.

**Example D — all three seats on one model**

```sh
uv run python tools/live_demo.py --panel-model muse-spark --zydah --question "What is the weather in Springfield today?"
```

`--panel-model` implies `--zydah-panel`. Judge names, separate
sessions, and effort are preserved; only the seat occupant changes.
The same model in three seats gives three independently sessioned
draws from one model's distribution.

**Example E — Springfield with a human clue**

Run Example C or D. When Zydah pauses, answer at the prompt:

- `[you: choice key]` — type a key (e.g. `B` for North); `Other`
  then asks for free text.
- `[you: answer]` — type the clue, e.g. `the celtics are a great team`.

Zydah parses the clue, conditions the Springfield candidate graph on
it (Celtics → Boston, MA → Springfield, MA), and re-scores. Pressing
Enter sends typed text; Enter alone never quits. `q` or Ctrl-C quits
gracefully (`[zydah] bye`).

## 1 judge vs the Zydah panel (terminal)

| Path | Command | Scoring | Terminal output |
|---|---|---|---|
| 1 judge (direct) | `tools/live_demo.py` (no panel flag) | One elicitor mass, fused alone | `[muse turn N]` + `[zydah turn N]` verdict |
| 3 judges (panel) | `tools/live_demo.py --zydah-panel` | 3 judge masses, $\alpha$-discounted, Yager fusion, quorum 2/3 | Zydah Chamber table + fused Bel/Pl/width/$K$ |

Use the direct path for cheap checks. Use the panel when independent
judgments matter: disagreement shows up as conflict $K$ and interval
width instead of being averaged away. Each turn's verdict line
carries elapsed seconds and token usage.

## Streamlit UI

Launch:

```sh
uv run streamlit run tools/streamlit_app.py
```

Open the local URL Streamlit prints (default
http://localhost:8501). With the panel on Spark seats, frames, and
usage:

```sh
uv run streamlit run tools/streamlit_app.py -- \
  --panel-model muse-spark \
  --show-frames \
  --show-usage
```

What you see:

- A chat column: ask a question, watch the [muse] response stream,
  then the [zydah] verdict.
- A belief panel: the current belief frame and [Bel, Pl] across
  turns, plus per-judge frames for panel turns.
- Structured choices in the UI when Zydah pauses for a human choice.

Sidebar controls mirror the terminal flags: panel visibility and
scoring, panel source (`yaml`/`base`), panel model, score mode, show
frames, and show usage.

**No credentials? Replay a saved run:**

```sh
uv run streamlit run tools/streamlit_app.py -- \
  --mock-artifact outputs/live_demo/latest.json
```

Replay renders a saved artifact with no model calls. Use “Clear
history” in the sidebar to reset the session.

## Reading the output

- `[muse turn N]` — the model's Nth response.
- `[zydah turn N]` — Zydah's score of that turn.
- `verdict=good` — interval width and conflict are within thresholds.
- `decision` — commit / escalate / reject from the [Bel, Pl] interval
  against $\tau$: commit if $Bel \ge \tau$, reject if $Pl < \tau$,
  escalate when the interval straddles $\tau$.
- `width` = $Pl - Bel$ (ignorance); `conflict` = $K$ (judge
  disagreement). Both are signals, not errors.
- `ZJ-01…03` — judge seat tags; `seat` names the model in that seat.
- `ASSISTED` — the cycle used follow-up or human input.
  `CONTEXT_INDEPENDENT` — first response came from a separate session
  with no follow-ups consumed.

## Configuration

- `.env` (this directory, never committed): `MODEL_API_BASE_URL`,
  `MODEL_API_KEY`, and optionally `MODEL_API_MODEL`.
- `MODEL_API_MODEL`, when set, is the parent endpoint model: it is
  sent on the wire for every call, ahead of any caller-supplied model.
  When unset, caller models apply — the verifier model from
  `src/zydah/conf/verifier.yaml`, judge seats from
  `src/zydah/conf/panel.yaml`, `--panel-source`, or `--panel-model`.
  Judge names/sessions can therefore differ from the wire model;
  check `chat_complete model=...` in `trace.log`.
- Panel config (`src/zydah/conf/panel.yaml`): judge names/models,
  quorum 2 of 3, effort `low`, and Shafer reliability $\alpha$
  (panel-global, per-judge override; shipped at 1.0).
- Human assertions are conditioned at $\alpha = 1.0$ by design: the
  operator is the authority in this evaluator. Model-derived
  conditioning uses $\alpha = 0.75$.
- Thresholds and marker lists live in `src/zydah/conf/*.yaml`; no
  hyperparameters in code.

Use `--log-file trace.log` for end-to-end tracing (inputs, prompts,
responses, routing, scored evidence). `--verbose` alone logs to
stderr; with `--log-file`, records go to the file only so the Rich
terminal view stays clean. `--save outputs/live_demo/latest.json`
mirrors the run into the JSON artifact that Streamlit replay reads.

## Concepts in one page

**Math (published sources only).** Dempster's rule (Dempster 1967;
Shafer 1976); Shafer reliability discounting — scale a source's
committed mass by $\alpha$ before combination, remainder to $\Omega$;
Yager's rule (Yager 1987) — conflict $K$ reassigned to $\Omega$;
Smets' TBM (Smets & Kennes 1994) — unnormalized conjunctive
combination, decision via the pignistic transform. The library
implements all four; the live pipeline defaults to Yager, Dempster
is a harness baseline, and TBM is an optional rule. Full derivation:
[Zadeh's Paradox, Yager, and Smets](<docs/Zadeh's Paradox, Yager, and Smets A Derivation from First Principles.md>).

**Credal vs pignistic.** The belief frame $(A, \sim A, \Omega)$ is
the stored value; Bel/Pl and pignistic probabilities are derived
from it, never stored as scalars.

**Seats and epistemic distributions.** A seat is the experimental
slot; its occupant model is the treatment. Each occupant induces an
epistemic distribution over mass assignments for the same evidence:
repeated independent calls sample from it, and changing the occupant
changes the distribution sampled. Same model in all seats = iid
draws, so $K$ measures intra-model disagreement; different models =
independent but not identically distributed.

**Model collapse vs mode collapse.** Model collapse is about which
models/distributions are present (e.g. named judges secretly routing
to one deployment). Mode collapse is about how concentrated one
model's output distribution is (repeated samples landing on the same
judgment). Provenance shows whether model collapse is a risk; the
realized masses and $K$ show whether mode collapse happened on that
cycle.

**Closed world, detect vs repair.** Frame failure is detectable:
under TBM, conflict on the empty set is the alarm, and the local
response is to suspend and escalate. Repairing the frame — supplying
the correct unlisted hypothesis — needs new evidence, which this
closed-world demo does not gather. The `world: closed` code gate and
the evidence-access gap are separate blockers.

## Tests and layout

```sh
uv run pytest
```

Suites split by kind under `tests/` (`suite`, `direct`, `levels`);
outputs mirror that split under `outputs/`. No API calls in tests.

- `src/zydah/fusion.py` — mass functions, combination rules, Bel/Pl/$K$
- `src/zydah/judges.py` — judge panel, YAML config, quorum, $\alpha$
- `src/zydah/verifier.py` — validator, follow-up loop, graph conditioning
- `src/zydah/router.py` — interval decision against $\tau$
- `src/zydah/evidence.py` — provenance ledger + anti-eavesdropping session rule
- `tools/live_demo.py` — interactive terminal demo
- `tools/streamlit_app.py` — Streamlit UI (live + replay)
- `eval/harness.py` — majority vote vs averaging vs Dempster vs Zydah vs discounted Zydah
- [docs/Pipeline Call Tree.md](<docs/Pipeline Call Tree.md>) — full call tree, panel paths, invariants

## Further RL Research

Bellman / temporal-difference (TD) routing is future research, not
current routing. A tabular TD(0)/Monte-Carlo learner is ported and
unit-tested offline and verifier trajectories are logged, but
learned routing is not wired. The next evidence step is live panel
trajectories with three judge masses, varying $K$, episodes of two
or more steps, and held-out labels. See the Pipeline Call Tree.

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

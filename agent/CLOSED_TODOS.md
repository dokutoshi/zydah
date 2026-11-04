# Closed TODOs — zydah

Finished work, newest first. `Jira:` fills in once the Cloud sync in
`OPEN_TODOS.md` (ZYDAH-002) is wired; `-` means not yet synced.

## ZYDAH-112 — Probe GNN framework on this machine (PyTorch/MPS via uv)

- Status: done
- Area: graph / tooling
- Source: user direction (Apple Metal PyTorch page; install with uv)
- Jira: ZYDAH-10
- Closed: 2026-11-03
- Details: probed in isolated uv envs: torch 2.15.0 installs on Python
  3.15 but torch-geometric 2.8.1 crashes constructing `GATConv` there,
  so PyG needs the pinned Python 3.12 graph environment (forward pass
  OK on CPU, output shape (3, 4)); DGL not needed. Installed as the
  optional `graph` group (`uv add --optional graph`); the core install
  stays numpy/polars. `src/zydah/conf/graph.yaml` created with
  device/model/training/calibration hyperparameters; device policy is
  `auto` (MPS when available, else CPU). MPS reported False in the
  sandboxed session — re-check on an unsandboxed run remains open.
- Verification: `GATConv` forward pass re-verified in the project
  `.venv` (shape (3, 4)); full `uv run pytest -q` → 126 passed,
  3 skipped.

## ZYDAH-115 — Route follow-up answers: [human] terminal vs [agent] resolution

- Status: done
- Area: verifier / routing
- Source: live Springfield run (model asked "Which Springfield?" five
  times and the loop burned the full Hydra budget)
- Jira: ZYDAH-12
- Closed: 2026-11-02
- Details: responder answers are classified before entering the
  ledger (`human_question_markers` in `conf/verifier.yaml`). A
  question addressed to the human returns `needs_human` with the
  verbatim `human_question`, consumes zero evidence follow-ups, adds
  no `MODEL_FOLLOWUP` chunk, and renders via `format_needs_human`
  as `[human]` / `[zydah]` pending. Agent answers may route through
  an injected graph resolver: resolved/subset resolutions append a
  `graph_resolution` evidence chunk; an unresolved graph generates
  the `[human]` question from the direction partitions (East, North,
  South, West). Both `verify` and `verify_direct` route identically.
- Verification: `uv run pytest -q` → 110 passed, 1 skipped (3 new
  routing tests using the live model's exact "Which Springfield"
  phrasing; no new live call was made for this closure).
- Addendum (human-choice resume): paused results now carry
  Pydantic/Enum structured choices (`src/zydah/choices.py`), and
  `resume_direct` maps a key (B -> North; `Other` -> free text) into a
  `HumanFollowup` JSON sent to the model; the revised response is
  appended and re-elicited, with the selection recorded as
  `human_feedback` (an RLHF-style signal). 114 passed, 1 skipped.
- Addendum (graph context): the follow-up now expands the chosen
  partition via the graph — member candidates, subset-renormalized
  resolver scores, and the original question (`HumanChoiceContext`).
  Live RLHF test: the model received the 4 North Springfields at
  0.25 each and correctly replied that North alone does not resolve
  the claim, asking the next narrowing question (`needs_human`).

## ZYDAH-114 — Deterministic baseline resolver + DS mass conversion

- Status: done
- Area: graph / fusion
- Source: GNN resolver plan (prove the interface before learning)
- Jira: ZYDAH-11
- Closed: 2026-11-02
- Details: `src/zydah/graph_resolver.py` implements the `Resolver`
  protocol with `NumpyResolver`: context affinity and `faster_than`
  signals score candidates, a temperature softmax (from
  `conf/graph.yaml`) forms the distribution, and entropy converts to
  Dempster-Shafer mass — singletons when confident, a subset focal
  element when a partition concentrates (best partition by excess
  probability over uniform share), residual on Omega. Resolution
  reports frame, mass, candidate/partition scores, entropy
  before/after, and resolved/subset/unresolved status.
- Verification: `uv run pytest -q` → 107 passed, 1 skipped (5 new
  tests in `tests/suite/test_graph_resolver.py`: all-Omega without
  context, Chicago-timezone subset mass, falcon resolution, GT-R
  top score, protocol conformance). Zero new dependencies.

## ZYDAH-113 — Candidate graph schema and builders

- Status: done
- Area: graph
- Source: GNN resolver plan (graph before GNN)
- Jira: ZYDAH-16
- Closed: 2026-11-02
- Details: `src/zydah/graph.py` defines `GraphNode` / `GraphEdge` /
  `CandidateGraph` with node kinds ambiguity/partition/candidate/
  context and edge kinds candidate_of/member_of/faster_than/affinity.
  `build_place_graph` emits the Springfield root, North/South/East/
  West + timezone partitions (thresholds from `conf/graph.yaml`), and
  8 candidate cities with context affinity edges.
  `build_comparison_graph` covers BMW-vs-Nissan (model class, spec
  faster-than edges) and bird-vs-horse (locomotion, fastest-vs-typical
  interpretation partitions; falcon > horse resolves in-graph). No
  ML-framework dependency.
- Verification: `uv run pytest -q` → 99 passed, 1 skipped (6 new
  tests in `tests/suite/test_graph.py`).

## ZYDAH-111 — Level tests: Simple vs Context-Independence

- Status: done
- Area: tests
- Source: user question (tests per level?)
- Jira: ZYDAH-17
- Closed: 2026-11-02
- Details: `tests/test_levels.py` — Simple level: weather questions over
  seeded-random states answered directly with zero follow-ups;
  Springfield-without-state honestly returns a follow-up question.
  Context-Independence level: car A vs car B comparisons earn
  CONTEXT_INDEPENDENT across sessions and JOINT_CONTEXT in one session.
- Verification: 6/6 level tests; full suite 91/91.

## ZYDAH-110 — Evidence boundary: provenance ledger + session rule

- Status: done
- Area: evidence / verifier
- Source: approved day2 plan (anti-eavesdropping; Playgrounds A/B)
- Jira: ZYDAH-18
- Closed: 2026-11-02
- Details: `src/zydah/evidence.py` adds provenance chunks (kind, model,
  session, playground, turn), original-only vs assisted views, and
  `classify_independence`: same session ⇒ JOINT_CONTEXT, follow-ups ⇒
  ASSISTED, separate sessions + original-only ⇒ CONTEXT_INDEPENDENT.
  `verify_direct` elicits from ledger views, records follow-ups as B
  chunks, returns independence + ledger, and appends JSONL cycle logs
  when `log_path` is set. `FirstResponse` carries session/playground,
  rendered in sealed output. Mixed models deferred to Playgrounds.
- Verification: `uv run pytest -q` → 85/85 pass (8 new tests in
  `tests/test_evidence.py`: view hiding, session rule, A-prompt purity,
  JSONL logging).

## ZYDAH-109 — Migrate environment to uv (closes ZYDAH-001)

- Status: done
- Area: packaging
- Source: user direction ("prefer uv over .venv and pip")
- Jira: ZYDAH-19
- Closed: 2026-11-02
- Details: uv 0.13.6 installed; `uv sync --extra dev` created `uv.lock`
  and a uv-managed `.venv` (old pip venv removed). README quickstart and
  AGENTS.md common commands now use `uv sync` / `uv run pytest`.
  Sandbox note: uv cache needed `UV_CACHE_DIR=/tmp/uv-cache` inside the
  agent sandbox; unrestricted shells use the default cache.
- Verification: `uv run pytest -q` → 72/72 pass; `uv.lock` present.

## ZYDAH-108 — Vectorized matrix fusion (numpy/polars)

- Status: done
- Area: fusion / performance
- Source: user direction (replace focal-pair loops with a matrix form)
- Jira: ZYDAH-20
- Closed: 2026-11-02
- Details: `src/zydah/matrix.py` encodes masses as power-set bitmask
  vectors; conjunctive combination is a `bincount` contraction over a
  cached intersection index; Dempster/Yager mirror the dict rules.
  Polars summarizes stacked batch vectors. numpy>=2 and polars>=1 added
  to `pyproject.toml` and installed in the project `.venv`.
- Verification: `tests/test_matrix.py` — 100 randomized trials match the
  dict API on all three rules; Zadeh conflict 0.9999 reproduced; full
  suite 72/72. Note: this speeds fusion math, not model-call latency.

## ZYDAH-107 — Live transport, judge wiring, and verifier responder

- Status: done
- Area: transport / judges / verifier
- Source: user request ("let's build it"); closes ZYDAH-005 and ZYDAH-010
- Jira: ZYDAH-21
- Closed: 2026-11-02
- Details: `src/zydah/transport.py` (stdlib urllib, OpenAI-compatible)
  reads endpoint/key/model env names, chat path, timeout, and retries
  from `src/zydah/conf/transport.yaml`. `MetaApiJudge.judge()` wired via
  transport with fence-tolerant JSON extraction; `LivePanel` added.
  `make_model_responder()` wires the verifier loop. Endpoint env model
  wins over caller model names (endpoint serves one deployment).
- Verification: 7 offline tests in `tests/test_transport.py` (fake
  opener). Live probe passed: chat smoke, single judge mass, then the
  full Ohio-claim loop — iter 0 flagged (width 1.0), 1 follow-up,
  iter 1 good (width 0.0667, conflict 0.0647). Full suite 64/64.

## ZYDAH-106 — ZydahVerifier System 2 loop + src layout + Hydra config

- Status: done
- Area: verifier / packaging
- Source: user spec (validator, prompt injector, iteration loop)
- Jira: ZYDAH-22
- Closed: 2026-11-02
- Details: `ZydahVerifier` class in `src/zydah/verifier.py`: validator
  flags width/conflict above YAML thresholds (0.35); prompt injector
  builds the System 2 prompt; responder loop appends answers as evidence
  and re-fuses until good or budget ends. Model ID resolves explicit →
  env (`ZYDAH_VERIFIER_MODEL`) → YAML default. All hyperparameters live
  in `src/zydah/conf/verifier.yaml` via Hydra/OmegaConf; no magic numbers
  in code. Package moved to src layout (`src/zydah/`); hatchling and
  pytest `pythonpath` updated. Open-world mode raises `NotImplementedError`
  (web fetch is a follow-up).
- Verification: `python -m pytest -q` → 57/57 pass (7 new verifier tests);
  demo on "cloudy in Ohio" flags speculative and emits the System 2 prompt.

## ZYDAH-105 — Pipe-through interface with [muse]/[zydah] tags

- Status: done
- Area: interface
- Source: user request; Python API + mock panel (chosen via questions)
- Jira: ZYDAH-23
- Closed: 2026-11-02
- Details: `src/zydah/interface.py` exposes `fuse_response`, `direct_verdict`,
  `fused_verdict`, and `format_tagged`. Direct arm = lead judge's
  pignistic best; fused arm = Yager/TBM + interval decision with conflict
  exposed; TBM total conflict abstains as escalate.
- Verification: `python -m pytest -q` → 50/50 pass (7 new interface tests);
  demo renders both tags with conflict/width shown.

## ZYDAH-101 — Doc-derived fusion tests (derivation §§2–11)

- Status: done
- Area: fusion / tests
- Source: `docs/Zadeh's Paradox, Yager, and Smets A Derivation from First Principles.md`
- Jira: ZYDAH-28
- Closed: 2026-11-02
- Details: 13 tests pinning the doc's own numbers — §§4–5 examples,
  §6 parametric Zadeh, §7 discontinuity paths, §§3/9 duality, §9
  `BetP(TBM) = BetP(Dempster)` property, pignistic total-conflict, §10
  side-by-side table, §8 n-ary-vs-fold, §11 discount example.
- Verification: `python -m pytest tests/test_fusion.py -v` → 28/28 pass;
  full `python -m pytest -q` → 43/43 pass.

## ZYDAH-102 — Implement Shafer reliability discounting

- Status: done
- Area: fusion
- Source: derivation doc §11; gap found during review
- Jira: ZYDAH-29
- Closed: 2026-11-02
- Details: added `discount_mass(mass, frame, alpha)` in `src/zydah/fusion.py`
  with formula docstring, reusing `_validate` and `_check_finite` plus a
  `[0, 1]` range check; module docstring lists the method.
- Verification: new tests failed first (`ImportError`), then 28/28 fusion
  and 43/43 full suite green, including the α = 0.9 end-to-end numbers.

## ZYDAH-103 — Non-finite mass guard with rationale docstring

- Status: done
- Area: fusion
- Source: review finding (NaN slipped past checks, returned `{}`)
- Jira: ZYDAH-30
- Closed: 2026-11-02
- Details: added `_check_finite` (NaN/±Infinity → `ValueError`) wired
  into `make_mass` and `_validate`, with a docstring explaining the NaN
  comparison trap and the judge-JSON path.
- Verification: 2 regression tests failed authentically before the fix;
  15/15 fusion and 30/30 full suite green after.

## ZYDAH-104 — Fusion core review against published methods

- Status: done
- Area: fusion
- Source: review request for `zydah/fusion.py`
- Jira: ZYDAH-31
- Closed: 2026-11-02
- Details: verified conjunctive/Dempster/Yager/TBM, Bel/Pl, and pignistic
  against textbook definitions; confirmed n-ary order invariance and
  Zadeh behavior; found the NaN bug (→ ZYDAH-103).
- Verification: 13/13 fusion tests green plus 200-trial randomized
  invariant probes (n-ary vs sequential Dempster, TBM sums, Bel ≤ Pl).

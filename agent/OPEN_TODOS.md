# Open TODOs — zydah

Task management for this repo. Each item keeps a stable local ID; the
`Jira:` field stays empty until the Jira Cloud sync is wired (after env
keys are provided), at which point the sync fills in the issue key.
Move finished items to `CLOSED_TODOS.md` with verification notes.

Sync contract (for the future script, not yet built):

- One `##` section per issue; the `ZYDAH-###` heading ID is the sync key.
- `Status:` is one of `open`, `in_progress`, `blocked`.
- `Jira:` is `-` until synced, then the Cloud issue key (e.g. `ABC-123`).
- Secrets come only from environment variables; never paste keys here.


## ZYDAH-002 — Wire Jira Cloud sync for todo files

- Status: blocked (waiting on user env keys)
- Priority: high
- Area: tooling
- Source: user request
- Jira: ZYDAH-1
- Updated: 2026-11-02
- Details: build the sync between `OPEN_TODOS.md` / `CLOSED_TODOS.md` and
  Jira Cloud once the user provides credentials. Expected shape: a small
  script reading config from env, creating/updating one issue per `##`
  section keyed by `ZYDAH-###`, and writing the key back to `Jira:`.
  Blocked until keys/endpoint are supplied; do not invent a project key.
- Acceptance: round-trip creates each open item once, updates edits, and
  closes issues moved to `CLOSED_TODOS.md`, without duplicating keys.

## ZYDAH-003 — Decide per-judge discount source and wire alpha

- Status: open
- Priority: medium
- Area: fusion / judges
- Source: derivation doc §11 follow-up (deferred in plan)
- Jira: ZYDAH-2
- Updated: 2026-11-02
- Details: `discount_mass` exists in `src/zydah/fusion.py` but nothing calls
  it. The doc says a robust system does both (discount before, keep
  conflict after), but where each judge's `alpha` comes from is a product
  decision: fixed per-judge reliability, calibrated from data, or
  judge-reported. Decide, then wire the chosen source.
- Acceptance: documented alpha policy; discount applied at one clear call
  site; tests cover the wiring.

## ZYDAH-004 — Decide whether router consumes K / interval width directly

- Status: open
- Priority: medium
- Area: router
- Source: README thesis vs `src/zydah/router.py`
- Jira: ZYDAH-3
- Updated: 2026-11-02
- Details: README says "escalate on evidence (high conflict mass / wide
  [Bel, Pl] interval)". `decide_best`/`decide_hypothesis` decide on the
  `[Bel, Pl]` straddle only; `eval/harness.py` records `conflict` but the
  router never sees it. Decide if `tau`-straddle is sufficient or if K /
  width thresholds belong in the decision, then implement or record why not.
- Acceptance: decision recorded; router and README agree; tests updated.

## ZYDAH-006 — Verify Baseten Spark Open path

- Status: open
- Priority: low
- Area: judges
- Source: `src/zydah/judges.py` (`BasetenJudge`, `meta-models/muse-spark-open-1026`)
- Jira: ZYDAH-4
- Updated: 2026-11-02
- Details: model ID is recorded but the Baseten path is unverified.
  Confirm model ID, auth via `BASETEN_API_KEY`, and response parsing
  against one real call or recorded fixture.
- Acceptance: verified call or fixture; failures documented.

## ZYDAH-007 — Document discounting in README math section

- Status: open
- Priority: low
- Area: docs
- Source: `discount_mass` added; README still lists only Dempster / Yager / TBM
- Jira: ZYDAH-5
- Updated: 2026-11-02
- Details: add Shafer (1976) reliability discounting to the README "Math"
  section and the `src/zydah/fusion.py` layout line so docs match code.
- Acceptance: README names all four published methods with one-line roles.

## ZYDAH-008 — Add discounted-Zydah arm to harness comparison

- Status: open
- Priority: low
- Area: eval
- Source: derivation doc §11 "robust system does both"; `eval/harness.py`
- Jira: ZYDAH-6
- Updated: 2026-11-02
- Details: harness compares majority vote / average / Dempster / Zydah
  but has no discounted-fusion arm. Depends on ZYDAH-003's alpha policy.
  Add the arm and record how discounting shifts accuracy, coverage, and
  flip rate on the seeded mock panel.
- Acceptance: `compare()` includes the arm; metrics documented; tests updated.

## ZYDAH-009 — Open-world verifier mode via web fetch

- Status: open
- Priority: medium
- Area: verifier
- Source: user note (closed world now; open world will utilize webfetch)
- Jira: ZYDAH-7
- Updated: 2026-11-02
- Details: `ZydahVerifier` currently raises `NotImplementedError` when
  `world: open` in `src/zydah/conf/verifier.yaml`. Implement evidence
  gathering via web fetch so flagged questions can pull external sources,
  then re-fuse. Keep closed-world path unchanged.
- Acceptance: open mode fetches for the flagged question, cites sources,
  and iterates like the closed loop; tests use fixtures, no live calls.

## ZYDAH-011 — Structured logging + prompt/response-weighted mass assignment

- Status: open
- Priority: high
- Area: logging / verifier
- Source: user hypothesis (prompt + response determine follow-up need)
- Jira: ZYDAH-8
- Updated: 2026-11-02
- Details: hypothesis — mass assignment improves when the model weighs a
  response in context, not an isolated claim. Today `verify_direct` asks
  for mass on a bare claim and the model returns all-Ω (width 1.0), so
  follow-ups never converge. Two parts: (1) structured logging per cycle
  — original prompt, model response, elicited mass, Bel/Pl/width/conflict,
  validator output, follow-up prompt/response, model IDs, timings;
  (2) feed prompt+response (not claim alone) into mass elicitation, and
  let the validator decide follow-ups from that pair.
- Acceptance: JSONL logs per verify run; test shows prompt+response input
  commits mass where claim-only input returned all-Ω.

## ZYDAH-012 — Prompt taxonomy and pass/intercept routing gate

- Status: open
- Priority: high
- Area: routing / verifier
- Source: user insight (a greeting is not a pass/partial/fail claim)
- Jira: ZYDAH-9
- Updated: 2026-11-02
- Details: the {pass, partial, fail} frame only fits evaluative claims.
  A routing gate should classify input before fusion: greeting/smalltalk
  and plain commands pass through untouched; checkable factual claims
  and agent answers intercept into the verifier; underspecified prompts
  clarify first; high-stakes content intercepts with stricter thresholds.
  Today "hi" burns 3 follow-ups because nothing routes it away.
  Partial: the session-isolation half landed in ZYDAH-110 (session rule,
  provenance ledger); the prompt-type gate itself remains open.
- Acceptance: taxonomy table implemented as a gate in front of
  `verify_direct`; greeting/command fixtures pass with zero model mass
  calls; claim fixtures still intercept.

## ZYDAH-112 — Probe GNN framework on this machine (PyTorch/MPS via uv)

- Status: in_progress
- Priority: high
- Area: graph / tooling
- Source: user direction (Apple Metal PyTorch page; install with uv)
- Jira: ZYDAH-10
- Updated: 2026-11-02
- Details: project venv is Python 3.15.2 on arm64 macOS with no ML
  framework installed. Probe in an isolated uv env first: install
  `torch` (MPS backend per developer.apple.com/metal/pytorch), verify
  `torch.backends.mps.is_available()` and a tensor op on `mps`, then
  attempt `torch-geometric` (import `GATConv`, one forward pass). DGL is
  the fallback if PyG wheels fail. Nothing enters `pyproject.toml` /
  `uv.lock` until a winner is proven; the winner lands as an optional
  `graph` dependency group so the core install stays numpy/polars.
  Probe outcome (2026-11-02, uv isolated envs, installer: uv):
  torch 2.15.0 installs on Python 3.15, but torch-geometric 2.8.1
  crashes constructing `GATConv` on 3.15
  (`TypeError: _eval_type() missing 1 required positional argument`),
  so PyG is incompatible with the project interpreter today. On
  Python 3.12 the same versions install and the `GATConv` forward
  pass succeeds (output shape (3, 4), CPU). Chosen framework:
  PyTorch Geometric on a pinned Python 3.12 graph environment
  (uv-managed, project-local when built); DGL not needed.
  `torch.backends.mps.is_available()` reported False in this
  sandboxed session on both interpreters, so Metal acceleration is
  unverified here; CPU execution is confirmed and sufficient for
  these small graphs. MPS re-check on an unsandboxed run remains.
  Device policy: this machine is Apple Neo; the Metal (MPS) backend
  stays in as an optional plugin, not a requirement. Device selection
  will be a `graph.yaml` setting (`auto` -> MPS when available, else
  CPU) so the resolver runs everywhere and accelerates where Metal
  is present, with no code change.
  Installed (2026-11-02): `torch 2.15.0` + `torch-geometric 2.8.1`
  added via `uv add --optional graph`; `GATConv` forward pass verified
  in the project `.venv` (Python 3.12, CPU; MPS still unavailable in
  this sandboxed session). `src/zydah/conf/graph.yaml` created with
  device/model/training/calibration hyperparameters.
- Acceptance: recorded probe outcome (versions, MPS availability,
  forward-pass result) for torch and torch-geometric; chosen framework
  named with the evidence; core `uv run pytest -q` unaffected.

## ZYDAH-116 — Synthetic training data for the GNN resolver

- Status: open
- Priority: medium
- Area: graph / eval
- Source: GNN resolver plan (no labels exist yet)
- Jira: ZYDAH-13
- Updated: 2026-11-02
- Details: seeded generators producing labeled graphs. Place family:
  candidate sets with context signals (region, timezone, locale) and
  known true candidates. Comparison family: car models and animal
  species with spec attributes and pairwise `faster-than` labels,
  split so some entities are held out entirely (train/test by entity,
  not by edge, to measure generalization). Deterministic seeds; data
  generated in-code, nothing downloaded.
- Acceptance: generator emits train/held-out splits with documented
  seeds; label distributions reported; regenerating with the same
  seed reproduces byte-identical datasets.

## ZYDAH-117 — Learned GNN scorer (PyG) with calibration gate

- Status: in_progress (framework installed; blocked on ZYDAH-113/114/116)
- Priority: medium
- Area: graph
- Source: GNN resolver plan; depends on ZYDAH-112, ZYDAH-113,
  ZYDAH-116
- Jira: ZYDAH-14
- Updated: 2026-11-02
- Details: heterogeneous graph attention model over the ZYDAH-113
  schema, trained on ZYDAH-116 data, implementing the same `Resolver`
  protocol as the ZYDAH-114 baseline. Runs in the pinned Python 3.12
  graph environment chosen in ZYDAH-112 (PyG 2.8.1 breaks on the
  project Python 3.15); the core project never imports torch.
  Hyperparameters (hidden dim,
  layers, heads, lr, epochs) in `src/zydah/conf/graph.yaml`.
  Temperature/entropy calibration on a held-out split converts scores
  to DS masses; the alpha discount derives from measured calibration
  error, not a hand-set value. The GNN replaces the baseline only if
  it beats it; otherwise the baseline ships and the GNN result is
  recorded as a finding.
- Acceptance: candidate top-1 accuracy and calibration error reported
  against the deterministic baseline on held-out entities; winner
  named with numbers; checkpoint + config recorded; verifier consumes
  the winner through the `Resolver` protocol without importing torch.

## ZYDAH-118 — Resolver level tests: places, cars, bird-vs-horse

- Status: open
- Priority: medium
- Area: tests / levels
- Source: user questions ("works also for which car is faster? what
  is faster a bird or a horse?")
- Jira: ZYDAH-15
- Updated: 2026-11-02
- Details: level tests in `tests/levels/` exercising the resolver end
  to end. Springfield: resolution or `[human]` root-partition
  question (North/South/East/West). BMW-vs-Nissan: model-class and
  metric partitions (0-60 vs top speed can flip the winner).
  Bird-vs-horse: fastest-member interpretation resolves in-graph
  (peregrine falcon > horse), typical-member stays uncertain or
  routes to `[human]`. Live variant gated like the existing live
  Springfield test.
- Acceptance: artifacts under `outputs/levels/` show graph, scores,
  emitted masses, and route taken per case; offline suite green;
  live Springfield run terminates by resolution or `needs_human`,
  not by budget exhaustion.

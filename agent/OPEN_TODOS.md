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

- Status: in_progress (script built, dry-run + reads verified; live close-path proof pending)
- Priority: high
- Area: tooling
- Source: user request
- Jira: ZYDAH-1
- Updated: 2026-11-03
- Details: build the sync between `OPEN_TODOS.md` / `CLOSED_TODOS.md` and
  Jira Cloud once the user provides credentials. Expected shape: a small
  script reading config from env, creating/updating one issue per `##`
  section keyed by `ZYDAH-###`, and writing the key back to `Jira:`.
  Credentials supplied (`src/.env`); `tools/jira_sync.py` built and
  dry-run verified. Live proof of the close path (moved-to-closed items
  transitioning to Done) still pending.
- Acceptance: round-trip creates each open item once, updates edits, and
  closes issues moved to `CLOSED_TODOS.md`, without duplicating keys.

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

- Status: in_progress (framework installed; blocked on ZYDAH-116)
- Priority: medium
- Area: graph
- Source: GNN resolver plan; depends on ZYDAH-112, ZYDAH-113,
  ZYDAH-116
- Jira: ZYDAH-14
- Updated: 2026-11-03
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

## ZYDAH-013 — Optimizer: bitmask focal sets + NumPy combination in fusion core

- Status: open
- Jira: ZYDAH-32
- Priority: low
- Area: fusion / performance
- Source: user optimizer note, 2026-11-04
- Updated: 2026-11-04
- Details: `src/zydah/fusion.py` represents focal elements as
  frozensets and combines via `itertools.product` (correctness first:
  simultaneous n-ary, order-invariant). Prior measurement: switching
  focal sets to bitmask integers gave ~2x in pure Python, and NumPy
  added another ~2x on top of `itertools.product`. Worth doing only
  when panel size or frame size grows; current 3-judge x <=4-focal
  cycles are <=64 combinations, so this is headroom, not a bottleneck
  fix. Public `Mass` type stays dict-of-frozensets; bitmask/NumPy is
  an internal representation behind the same functions.
- Acceptance: benchmark script compares current vs bitmask vs
  NumPy paths on representative frames; results identical within
  float tolerance on the existing fusion test suite (order invariance,
  Zadeh, total conflict); speedup numbers recorded in the benchmark
  output before any path becomes default.

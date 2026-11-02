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

## ZYDAH-001 — Decide uv vs pixi and add lockfile

- Status: open
- Priority: high
- Area: packaging
- Source: user request (future package additions)
- Jira: -
- Updated: 2026-11-02
- Details: repo currently uses `python3 -m venv .venv` + `pip install -e
  ".[dev]"` with no lockfile (`pyproject.toml` pins nothing). Pick one
  manager, add its lockfile, and update `README.md` quickstart and
  `AGENTS.md` common commands.
- Acceptance: clean-clone install reproduces the environment from the
  lockfile; `pytest` passes under the new flow.

## ZYDAH-002 — Wire Jira Cloud sync for todo files

- Status: blocked (waiting on user env keys)
- Priority: high
- Area: tooling
- Source: user request
- Jira: -
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
- Jira: -
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
- Jira: -
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
- Jira: -
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
- Jira: -
- Updated: 2026-11-02
- Details: add Shafer (1976) reliability discounting to the README "Math"
  section and the `src/zydah/fusion.py` layout line so docs match code.
- Acceptance: README names all four published methods with one-line roles.

## ZYDAH-008 — Add discounted-Zydah arm to harness comparison

- Status: open
- Priority: low
- Area: eval
- Source: derivation doc §11 "robust system does both"; `eval/harness.py`
- Jira: -
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
- Jira: -
- Updated: 2026-11-02
- Details: `ZydahVerifier` currently raises `NotImplementedError` when
  `world: open` in `src/zydah/conf/verifier.yaml`. Implement evidence
  gathering via web fetch so flagged questions can pull external sources,
  then re-fuse. Keep closed-world path unchanged.
- Acceptance: open mode fetches for the flagged question, cites sources,
  and iterates like the closed loop; tests use fixtures, no live calls.


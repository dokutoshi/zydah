# Closed TODOs — zydah

Finished work, newest first. `Jira:` fills in once the Cloud sync in
`OPEN_TODOS.md` (ZYDAH-002) is wired; `-` means not yet synced.

## ZYDAH-108 — Vectorized matrix fusion (numpy/polars)

- Status: done
- Area: fusion / performance
- Source: user direction (replace focal-pair loops with a matrix form)
- Jira: -
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
- Jira: -
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
- Jira: -
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
- Jira: -
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
- Jira: -
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
- Jira: -
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
- Jira: -
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
- Jira: -
- Closed: 2026-11-02
- Details: verified conjunctive/Dempster/Yager/TBM, Bel/Pl, and pignistic
  against textbook definitions; confirmed n-ary order invariance and
  Zadeh behavior; found the NaN bug (→ ZYDAH-103).
- Verification: 13/13 fusion tests green plus 200-trial randomized
  invariant probes (n-ary vs sequential Dempster, TBM sums, Bel ≤ Pl).

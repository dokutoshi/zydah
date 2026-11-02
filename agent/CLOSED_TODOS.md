# Closed TODOs — zydah

Finished work, newest first. `Jira:` fills in once the Cloud sync in
`OPEN_TODOS.md` (ZYDAH-002) is wired; `-` means not yet synced.

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
- Details: added `discount_mass(mass, frame, alpha)` in `zydah/fusion.py`
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

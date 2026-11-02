"""Fusion core tests (no API calls).

Conventions: frame elements are plain strings; a mass is a dict mapping
frozenset -> probability mass. EMPTY (frozenset()) carries conflict mass
only in unnormalized (TBM) results.
"""

import itertools
import math
import random

import pytest

from zydah.fusion import (
    TotalConflictError,
    belief,
    conjunctive_combine,
    conflict_mass,
    dempster_combine,
    discount_mass,
    make_mass,
    pignistic,
    plausibility,
    yager_combine,
)

PASS = frozenset({"pass"})
PARTIAL = frozenset({"partial"})
FAIL = frozenset({"fail"})


def zadeh_pair():
    """Zadeh's paradox (1984): two near-certain but conflicting doctors.

    Toy numbers: m1 says {a: 0.99, b: 0.01}, m2 says {b: 0.01, c: 0.99}.
    Conjunctive: only b*b = 0.0001 survives; K = 0.9999.
    """
    frame = frozenset({"a", "b", "c"})
    m1 = make_mass(frame, {"a": 0.99, "b": 0.01})
    m2 = make_mass(frame, {"b": 0.01, "c": 0.99})
    return frame, [m1, m2]


def test_make_mass_validates_sum_to_one():
    frame = frozenset({"pass", "fail"})
    with pytest.raises(ValueError):
        make_mass(frame, {"pass": 0.6, "fail": 0.5})
    with pytest.raises(ValueError):
        make_mass(frame, {"pass": -0.1, "fail": 1.1})
    with pytest.raises(ValueError):
        make_mass(frame, {"pass": 0.5, "other": 0.5})


def test_make_mass_rejects_non_finite_values():
    frame = frozenset({"pass", "fail"})
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="non-finite"):
            make_mass(frame, {"pass": bad, "fail": 1.0})
    # Regression: NaN alone previously passed validation and returned {}.
    with pytest.raises(ValueError, match="non-finite"):
        make_mass(frame, {"pass": math.nan})


def test_combine_rejects_hand_built_non_finite_mass():
    frame = frozenset({"pass", "fail"})
    bad = {PASS: math.nan, FAIL: 0.5}
    with pytest.raises(ValueError, match="non-finite"):
        conjunctive_combine([bad], frame)


def test_zadeh_conflict_mass():
    frame, masses = zadeh_pair()
    assert conflict_mass(masses, frame) == pytest.approx(0.9999)


def test_zadeh_dempster_certainty_on_minority():
    """Dempster normalizes conflict away -> full certainty on {b}, which
    each source assigned only 0.01. The paradox this project avoids."""
    frame, masses = zadeh_pair()
    fused = dempster_combine(masses, frame)
    assert fused[frozenset({"b"})] == pytest.approx(1.0)
    assert sum(fused.values()) == pytest.approx(1.0)


def test_zadeh_yager_preserves_conflict_as_ignorance():
    frame, masses = zadeh_pair()
    fused = yager_combine(masses, frame)
    assert fused[frozenset({"b"})] == pytest.approx(0.0001)
    assert fused[frame] == pytest.approx(0.9999)
    assert sum(fused.values()) == pytest.approx(1.0)


def test_zadeh_tbm_keeps_conflict_on_empty_set():
    frame, masses = zadeh_pair()
    fused = conjunctive_combine(masses, frame)
    assert fused[frozenset()] == pytest.approx(0.9999)
    assert fused[frozenset({"b"})] == pytest.approx(0.0001)
    assert sum(fused.values()) == pytest.approx(1.0)


def test_masses_sum_to_one_after_each_rule():
    frame = frozenset({"pass", "partial", "fail"})
    masses = [
        make_mass(frame, {"pass": 0.7, ("pass", "partial", "fail"): 0.3}),
        make_mass(frame, {"pass": 0.5, "partial": 0.2, ("pass", "partial", "fail"): 0.3}),
        make_mass(frame, {"fail": 0.6, ("pass", "partial", "fail"): 0.4}),
    ]
    for combine in (conjunctive_combine, dempster_combine, yager_combine):
        assert sum(combine(masses, frame).values()) == pytest.approx(1.0)


def assert_masses_equal(m1, m2):
    assert set(m1) == set(m2)
    for k in m1:
        assert m1[k] == pytest.approx(m2[k])


def test_order_invariance_under_source_shuffling():
    """Simultaneous n-ary combination: every permutation fuses identically."""
    frame = frozenset({"pass", "partial", "fail"})
    masses = [
        make_mass(frame, {"pass": 0.7, ("pass", "partial", "fail"): 0.3}),
        make_mass(frame, {"pass": 0.5, "partial": 0.2, ("pass", "partial", "fail"): 0.3}),
        make_mass(frame, {"fail": 0.6, ("pass", "partial", "fail"): 0.4}),
    ]
    rng = random.Random(0)
    for combine in (conjunctive_combine, dempster_combine, yager_combine):
        base = combine(masses, frame)
        for perm in itertools.permutations(masses):
            assert_masses_equal(base, combine(list(perm), frame))
        shuffled = list(masses)
        rng.shuffle(shuffled)
        assert_masses_equal(base, combine(shuffled, frame))


def test_bel_pl_toy_binary():
    """m({pass}) = 0.6, m(Omega) = 0.4 -> Bel(pass) = 0.6, Pl(pass) = 1.0;
    Bel(fail) = 0, Pl(fail) = 0.4."""
    frame = frozenset({"pass", "fail"})
    m = make_mass(frame, {"pass": 0.6, ("pass", "fail"): 0.4})
    assert belief(m, PASS) == pytest.approx(0.6)
    assert plausibility(m, PASS) == pytest.approx(1.0)
    assert belief(m, FAIL) == pytest.approx(0.0)
    assert plausibility(m, FAIL) == pytest.approx(0.4)


def test_bel_pl_compound_focal_elements():
    """m({a,b}) = 0.5, m({b,c}) = 0.3, m(Omega) = 0.2 over {a, b, c}."""
    frame = frozenset({"a", "b", "c"})
    m = make_mass(frame, {("a", "b"): 0.5, ("b", "c"): 0.3, ("a", "b", "c"): 0.2})
    assert belief(m, frozenset({"b"})) == pytest.approx(0.0)
    assert plausibility(m, frozenset({"b"})) == pytest.approx(1.0)
    assert belief(m, frozenset({"a", "b"})) == pytest.approx(0.5)
    assert plausibility(m, frozenset({"a", "b"})) == pytest.approx(1.0)
    assert belief(m, frame) == pytest.approx(1.0)


def test_bel_never_exceeds_pl():
    frame = frozenset({"pass", "partial", "fail"})
    m = make_mass(
        frame,
        {"pass": 0.2, ("pass", "partial"): 0.3, ("partial", "fail"): 0.1,
         ("pass", "partial", "fail"): 0.4},
    )
    for r in range(1, 4):
        for combo in itertools.combinations(sorted(frame), r):
            h = frozenset(combo)
            assert belief(m, h) <= plausibility(m, h) + 1e-12


def test_total_conflict_dempster_raises_yager_goes_to_omega():
    frame = frozenset({"a", "b"})
    masses = [make_mass(frame, {"a": 1.0}), make_mass(frame, {"b": 1.0})]
    assert conflict_mass(masses, frame) == pytest.approx(1.0)
    with pytest.raises(TotalConflictError):
        dempster_combine(masses, frame)
    fused = yager_combine(masses, frame)
    assert fused[frame] == pytest.approx(1.0)


def test_pignistic_sums_to_one_and_splits_compound_mass():
    """m({a,b}) = 0.5, m({c}) = 0.5 -> BetP = {a: 0.25, b: 0.25, c: 0.5}."""
    frame = frozenset({"a", "b", "c"})
    m = make_mass(frame, {("a", "b"): 0.5, "c": 0.5})
    betp = pignistic(m, frame)
    assert math.isclose(sum(betp.values()), 1.0)
    assert betp == pytest.approx({"a": 0.25, "b": 0.25, "c": 0.5})


def test_pignistic_renormalizes_tbm_empty_mass():
    _, masses_pair = zadeh_pair()
    frame = frozenset({"a", "b", "c"})
    fused = conjunctive_combine(masses_pair, frame)
    betp = pignistic(fused, frame)
    assert math.isclose(sum(betp.values()), 1.0)


# --- Doc-derived tests: "Zadeh's Paradox, Yager, and Smets" (docs/) ---
# Expected values are the doc's own numbers (Sections cited per test),
# not values recomputed through the code under test.

DOC_FRAME = frozenset({"M", "T", "C"})
M_SET = frozenset({"M"})
T_SET = frozenset({"T"})
C_SET = frozenset({"C"})
EMPTY_SET = frozenset()


def doc_mass(assign):
    return make_mass(DOC_FRAME, assign)


def zadeh_doctors(eps, delta):
    return (
        doc_mass({"M": 1 - eps, "T": eps}),
        doc_mass({"C": 1 - delta, "T": delta}),
    )


def all_subsets(frame):
    items = sorted(frame)
    for r in range(len(items) + 1):
        for combo in itertools.combinations(items, r):
            yield frozenset(combo)


def test_doc_example_4a_no_conflict():
    # Doc Section 4a: m1 {T}=0.6, Omega=0.4; m2 {T,C}=0.5, Omega=0.5.
    m1 = doc_mass({"T": 0.6, ("M", "T", "C"): 0.4})
    m2 = doc_mass({("T", "C"): 0.5, ("M", "T", "C"): 0.5})
    fused = conjunctive_combine([m1, m2], DOC_FRAME)
    assert_masses_equal(
        fused,
        {T_SET: 0.60, frozenset({"T", "C"}): 0.20, DOC_FRAME: 0.20},
    )
    assert conflict_mass([m1, m2], DOC_FRAME) == pytest.approx(0.0)


def test_doc_example_4b_partial_conflict_and_dempster():
    # Doc Sections 4b/5: K=0.30; Dempster divides by 0.70.
    m1 = doc_mass({"M": 0.6, ("M", "T", "C"): 0.4})
    m2 = doc_mass({"T": 0.5, ("M", "T", "C"): 0.5})
    fused = conjunctive_combine([m1, m2], DOC_FRAME)
    assert_masses_equal(
        fused,
        {EMPTY_SET: 0.30, M_SET: 0.30, T_SET: 0.20, DOC_FRAME: 0.20},
    )
    assert_masses_equal(
        dempster_combine([m1, m2], DOC_FRAME),
        {M_SET: 0.4285714285714286, T_SET: 0.2857142857142857, DOC_FRAME: 0.2857142857142857},
    )


def test_doc_zadeh_parametric_any_eps_delta():
    # Doc Section 6: K = 1 - eps*delta and m_D({T}) = 1 for any eps, delta.
    for eps, delta in [(0.01, 0.01), (0.1, 0.05), (0.001, 0.2)]:
        m1, m2 = zadeh_doctors(eps, delta)
        assert conflict_mass([m1, m2], DOC_FRAME) == pytest.approx(1 - eps * delta)
        fused = conjunctive_combine([m1, m2], DOC_FRAME)
        assert fused[T_SET] == pytest.approx(eps * delta)
        assert dempster_combine([m1, m2], DOC_FRAME)[T_SET] == pytest.approx(1.0)


def test_doc_dempster_discontinuity_paths_a_b():
    # Doc Section 7: inputs within eps of total conflict, outputs stay apart.
    for eps in (0.1, 0.01, 0.001):
        a1, a2 = zadeh_doctors(eps, eps)
        b1 = doc_mass({"M": 1 - eps, "C": eps})
        b2 = doc_mass({"C": 1 - eps, "M": eps})
        assert dempster_combine([a1, a2], DOC_FRAME)[T_SET] == pytest.approx(1.0)
        demp_b = dempster_combine([b1, b2], DOC_FRAME)
        assert demp_b[M_SET] == pytest.approx(0.5)
        assert demp_b[C_SET] == pytest.approx(0.5)
        # Yager removes the jump: the two paths differ by < 2*eps in any mass.
        yag_a = yager_combine([a1, a2], DOC_FRAME)
        yag_b = yager_combine([b1, b2], DOC_FRAME)
        keys = set(yag_a) | set(yag_b)
        assert max(abs(yag_a.get(k, 0.0) - yag_b.get(k, 0.0)) for k in keys) < 2 * eps


def test_doc_partial_bel_pl_table_and_duality():
    # Doc Section 3: Partial source and Pl(A) = 1 - Bel(complement).
    m = doc_mass({"T": 0.6, ("T", "C"): 0.3, ("M", "T", "C"): 0.1})
    assert belief(m, T_SET) == pytest.approx(0.6)
    assert plausibility(m, T_SET) == pytest.approx(1.0)
    assert belief(m, C_SET) == pytest.approx(0.0)
    assert plausibility(m, C_SET) == pytest.approx(0.4)
    assert belief(m, M_SET) == pytest.approx(0.0)
    assert plausibility(m, M_SET) == pytest.approx(0.1)
    assert belief(m, frozenset({"T", "C"})) == pytest.approx(0.9)
    assert plausibility(m, frozenset({"T", "C"})) == pytest.approx(1.0)
    for hyp in all_subsets(DOC_FRAME):
        assert plausibility(m, hyp) == pytest.approx(1 - belief(m, DOC_FRAME - hyp))


def test_doc_bayesian_and_vacuous_boundaries():
    # Doc Sections 3/9: Bayesian intervals have zero width; vacuous is [0, 1]
    # and its BetP is uniform; Bayesian BetP equals its own masses.
    m1, _ = zadeh_doctors(0.01, 0.01)
    assert belief(m1, T_SET) == pytest.approx(0.01)
    assert plausibility(m1, T_SET) == pytest.approx(0.01)
    assert pignistic(m1, DOC_FRAME) == pytest.approx({"M": 0.99, "T": 0.01, "C": 0.0})
    vacuous = doc_mass({("M", "T", "C"): 1.0})
    for hyp in all_subsets(DOC_FRAME):
        if hyp and hyp != DOC_FRAME:
            assert belief(vacuous, hyp) == pytest.approx(0.0)
            assert plausibility(vacuous, hyp) == pytest.approx(1.0)
    assert pignistic(vacuous, DOC_FRAME) == pytest.approx(
        {"M": 1 / 3, "T": 1 / 3, "C": 1 / 3}
    )


def test_doc_tbm_duality_and_closed_world_failure():
    # Doc Section 9: Pl_S(A) = 1 - m(empty) - Bel_S(complement); the
    # closed-world form fails on Zadeh's numbers (would give Pl({T}) = 1).
    m1, m2 = zadeh_doctors(0.01, 0.01)
    fused = conjunctive_combine([m1, m2], DOC_FRAME)
    for hyp in all_subsets(DOC_FRAME):
        assert plausibility(fused, hyp) == pytest.approx(
            1 - fused[EMPTY_SET] - belief(fused, DOC_FRAME - hyp)
        )
    assert plausibility(fused, T_SET) == pytest.approx(0.0001)
    assert 1 - belief(fused, DOC_FRAME - T_SET) == pytest.approx(1.0)


def test_doc_betp_tbm_equals_betp_dempster_random():
    # Doc Section 9: BetP of the TBM result equals BetP of Dempster's result.
    rng = random.Random(7)
    focal_choices = [M_SET, T_SET, C_SET, frozenset({"M", "T"}), DOC_FRAME]
    for _ in range(50):
        masses = []
        for _ in range(3):
            weights = [rng.random() + 0.05 for _ in focal_choices]
            total = sum(weights)
            masses.append({f: w / total for f, w in zip(focal_choices, weights)})
        betp_tbm = pignistic(conjunctive_combine(masses, DOC_FRAME), DOC_FRAME)
        betp_demp = pignistic(dempster_combine(masses, DOC_FRAME), DOC_FRAME)
        assert betp_tbm == pytest.approx(betp_demp)


def test_doc_pignistic_total_conflict_raises():
    masses = [doc_mass({"M": 1.0}), doc_mass({"C": 1.0})]
    fused = conjunctive_combine(masses, DOC_FRAME)
    assert fused[EMPTY_SET] == pytest.approx(1.0)
    with pytest.raises(TotalConflictError):
        pignistic(fused, DOC_FRAME)


def test_doc_side_by_side_zadeh_table():
    # Doc Section 10: all three rules on K = 0.9999.
    m1, m2 = zadeh_doctors(0.01, 0.01)
    demp = dempster_combine([m1, m2], DOC_FRAME)
    yag = yager_combine([m1, m2], DOC_FRAME)
    tbm = conjunctive_combine([m1, m2], DOC_FRAME)
    assert_masses_equal(demp, {T_SET: 1.0})
    assert_masses_equal(yag, {T_SET: 0.0001, DOC_FRAME: 0.9999})
    assert_masses_equal(tbm, {T_SET: 0.0001, EMPTY_SET: 0.9999})
    assert (belief(demp, T_SET), plausibility(demp, T_SET)) == pytest.approx((1.0, 1.0))
    assert (belief(yag, T_SET), plausibility(yag, T_SET)) == pytest.approx((0.0001, 1.0))
    assert (belief(yag, M_SET), plausibility(yag, M_SET)) == pytest.approx((0.0, 0.9999))
    assert (belief(tbm, T_SET), plausibility(tbm, T_SET)) == pytest.approx((0.0001, 0.0001))
    assert (belief(tbm, M_SET), plausibility(tbm, M_SET)) == pytest.approx((0.0, 0.0))
    assert pignistic(demp, DOC_FRAME)["T"] == pytest.approx(1.0)
    assert pignistic(tbm, DOC_FRAME)["T"] == pytest.approx(1.0)
    betp_yag = pignistic(yag, DOC_FRAME)
    assert betp_yag["T"] == pytest.approx(0.3334, abs=1e-4)
    assert betp_yag["M"] == pytest.approx(0.3333, abs=1e-4)
    assert betp_yag["C"] == pytest.approx(0.3333, abs=1e-4)


def test_doc_yager_nary_not_sequential_fold():
    # Doc Section 8 cost: Yager is not associative; n-ary single-step is the
    # remedy, so a pairwise fold must be shown to differ (test-local fold).
    def fold_yager(masses):
        acc = masses[0]
        for nxt in masses[1:]:
            acc = yager_combine([acc, nxt], DOC_FRAME)
        return acc

    m1, m2 = zadeh_doctors(0.01, 0.01)
    m3 = doc_mass({"M": 0.8, ("M", "T", "C"): 0.2})
    nary = yager_combine([m1, m2, m3], DOC_FRAME)
    assert_masses_equal(nary, {T_SET: 0.00002, DOC_FRAME: 0.99998})
    folded = fold_yager([m1, m2, m3])
    assert folded[M_SET] == pytest.approx(0.79992)
    assert folded.get(M_SET, 0.0) != pytest.approx(nary.get(M_SET, 0.0))


def test_discount_mass_formula_and_edges():
    # Doc Section 11: m^a(X) = a*m(X); m^a(Omega) = 1 - a + a*m(Omega).
    m = doc_mass({"T": 0.6, ("T", "C"): 0.3, ("M", "T", "C"): 0.1})
    assert_masses_equal(
        discount_mass(m, DOC_FRAME, 0.5),
        {T_SET: 0.30, frozenset({"T", "C"}): 0.15, DOC_FRAME: 0.55},
    )
    assert_masses_equal(discount_mass(m, DOC_FRAME, 0.0), {DOC_FRAME: 1.0})
    assert_masses_equal(discount_mass(m, DOC_FRAME, 1.0), m)
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match="alpha"):
            discount_mass(m, DOC_FRAME, bad)
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="non-finite"):
            discount_mass(m, DOC_FRAME, bad)


def test_doc_discounted_zadeh_alpha_0_9():
    # Doc Section 11: discounting to 0.9 brings {T} from 1 back to ~0.0099.
    m1, m2 = zadeh_doctors(0.01, 0.01)
    d1 = discount_mass(m1, DOC_FRAME, 0.9)
    d2 = discount_mass(m2, DOC_FRAME, 0.9)
    assert_masses_equal(d1, {M_SET: 0.891, T_SET: 0.009, DOC_FRAME: 0.1})
    assert_masses_equal(d2, {C_SET: 0.891, T_SET: 0.009, DOC_FRAME: 0.1})
    assert conflict_mass([d1, d2], DOC_FRAME) == pytest.approx(0.809919)
    assert_masses_equal(
        conjunctive_combine([d1, d2], DOC_FRAME),
        {EMPTY_SET: 0.809919, M_SET: 0.0891, C_SET: 0.0891, T_SET: 0.001881, DOC_FRAME: 0.01},
    )
    demp = dempster_combine([d1, d2], DOC_FRAME)
    assert demp[M_SET] == pytest.approx(0.4687, abs=1e-4)
    assert demp[C_SET] == pytest.approx(0.4687, abs=1e-4)
    assert demp[T_SET] == pytest.approx(0.0099, abs=1e-4)
    assert demp[DOC_FRAME] == pytest.approx(0.0526, abs=1e-4)

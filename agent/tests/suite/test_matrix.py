"""Matrix (vectorized) fusion equivalence tests vs the dict API."""

import random

import numpy as np
import pytest

from zydah.fusion import conjunctive_combine, dempster_combine, make_mass, yager_combine
from zydah.matrix import (
    combine_matrix,
    conjunctive_vectors,
    conflict_of,
    mass_to_vector,
    summarize_vectors,
    vector_to_mass,
)

FRAME = frozenset({"pass", "partial", "fail"})
OMEGA = ("pass", "partial", "fail")
FOCALS = [
    frozenset({"pass"}),
    frozenset({"partial"}),
    frozenset({"fail"}),
    frozenset({"pass", "partial"}),
    FRAME,
]


def assert_masses_equal(m1, m2):
    assert set(m1) == set(m2)
    for k in m1:
        assert m1[k] == pytest.approx(m2[k])


def random_masses(rng, n=3):
    masses = []
    for _ in range(n):
        weights = [rng.random() + 0.05 for _ in FOCALS]
        total = sum(weights)
        masses.append({f: w / total for f, w in zip(FOCALS, weights)})
    return masses


def test_vector_roundtrip():
    m = make_mass(FRAME, {"pass": 0.6, OMEGA: 0.4})
    assert_masses_equal(vector_to_mass(mass_to_vector(m, FRAME), FRAME), m)


def test_matrix_matches_dict_rules_random():
    rng = random.Random(3)
    for _ in range(100):
        masses = random_masses(rng)
        assert_masses_equal(combine_matrix(masses, FRAME, "conjunctive"), conjunctive_combine(masses, FRAME))
        assert_masses_equal(combine_matrix(masses, FRAME, "yager"), yager_combine(masses, FRAME))
        assert_masses_equal(combine_matrix(masses, FRAME, "dempster"), dempster_combine(masses, FRAME))


def test_matrix_conflict_matches_zadeh():
    m1 = make_mass(FRAME, {"pass": 0.99, "partial": 0.01})
    m2 = make_mass(FRAME, {"partial": 0.01, "fail": 0.99})
    vec = conjunctive_vectors([mass_to_vector(m1, FRAME), mass_to_vector(m2, FRAME)], 3)
    assert conflict_of(vec) == pytest.approx(0.9999)


def test_summarize_vectors_batch_frame():
    rng = random.Random(5)
    vectors = []
    for _ in range(10):
        masses = random_masses(rng, n=2)
        vec = conjunctive_vectors([mass_to_vector(m, FRAME) for m in masses], 3)
        assert vec.sum() == pytest.approx(1.0)
        vectors.append(vec)
    df = summarize_vectors(vectors, FRAME)
    assert df.height > 0
    assert df["mean_mass"].sum() == pytest.approx(1.0)

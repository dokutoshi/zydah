"""Vectorized belief fusion over the power set (numpy / polars).

A mass becomes a vector indexed by subset bitmask: for a frame of n
labels, entry ``a`` (0..2**n-1) is the mass on the subset whose bits
select labels from the sorted frame. Combination is then a tensor
contraction, not a Python loop over focal-element pairs:

    combined[c] = sum_{a & b == c} m1[a] * m2[b]

computed as ``bincount(intersection_index, weights=outer(m1, m2))``.
The dict API in ``zydah.fusion`` remains the source of truth; this
module mirrors it for batch work and must agree with it (tested).

Polars enters at the batch layer: stacked vectors summarize into a
DataFrame of per-subset mean mass across many fused items.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, FrozenSet, List, Sequence

import numpy as np
import polars as pl

from zydah.fusion import EMPTY, Mass


def frame_labels(frame: FrozenSet[str]) -> List[str]:
    return sorted(frame)


def mask_of(subset: FrozenSet[str], labels: Sequence[str]) -> int:
    index = {label: i for i, label in enumerate(labels)}
    mask = 0
    for label in subset:
        mask |= 1 << index[label]
    return mask


def subset_of(mask: int, labels: Sequence[str]) -> FrozenSet[str]:
    return frozenset(label for i, label in enumerate(labels) if mask >> i & 1)


@lru_cache(maxsize=None)
def _power_set_size(n_labels: int) -> int:
    return 1 << n_labels


@lru_cache(maxsize=None)
def intersection_index(n_labels: int) -> np.ndarray:
    """Flattened (P x P) table mapping (a, b) -> bitmask of a & b."""
    masks = np.arange(_power_set_size(n_labels), dtype=np.int64)
    return (masks[:, None] & masks[None, :]).ravel()


def mass_to_vector(mass: Mass, frame: FrozenSet[str]) -> np.ndarray:
    labels = frame_labels(frame)
    vec = np.zeros(_power_set_size(len(labels)), dtype=np.float64)
    for subset, value in mass.items():
        vec[mask_of(subset, labels)] += value
    return vec


def vector_to_mass(vec: np.ndarray, frame: FrozenSet[str]) -> Mass:
    labels = frame_labels(frame)
    return {
        subset_of(int(mask), labels): float(value)
        for mask, value in enumerate(vec)
        if value > 0.0
    }


def conjunctive_vectors(vectors: Sequence[np.ndarray], n_labels: int) -> np.ndarray:
    """N-ary conjunctive combination of power-set vectors (folded)."""
    if not vectors:
        raise ValueError("need at least one vector")
    index = intersection_index(n_labels)
    size = _power_set_size(n_labels)
    combined = vectors[0].astype(np.float64)
    for vec in vectors[1:]:
        weights = np.outer(combined, vec).ravel()
        combined = np.bincount(index, weights=weights, minlength=size)
    return combined


def combine_matrix(masses: Sequence[Mass], frame: FrozenSet[str], rule: str = "yager") -> Mass:
    """Dict-API mirror of fusion rules computed on vectors."""
    labels = frame_labels(frame)
    vectors = [mass_to_vector(mass, frame) for mass in masses]
    combined = conjunctive_vectors(vectors, len(labels))
    kappa = float(combined[0])
    if rule == "conjunctive":
        return vector_to_mass(combined, frame)
    combined[0] = 0.0
    if rule == "dempster":
        if kappa >= 1.0:
            raise ValueError("total conflict: Dempster undefined at K = 1")
        return vector_to_mass(combined / (1.0 - kappa), frame)
    if rule == "yager":
        combined[mask_of(frame, labels)] += kappa
        return vector_to_mass(combined, frame)
    raise ValueError(f"unknown fusion rule {rule!r}")


def conflict_of(vec: np.ndarray) -> float:
    return float(vec[0])


def summarize_vectors(vectors: Sequence[np.ndarray], frame: FrozenSet[str]) -> pl.DataFrame:
    """Batch summary: mean mass per subset across fused items (polars)."""
    labels = frame_labels(frame)
    stacked = np.vstack(vectors)
    means = stacked.mean(axis=0)
    return pl.DataFrame(
        {
            "subset": ["+".join(sorted(subset_of(int(m), labels))) or "∅" for m in range(stacked.shape[1])],
            "mean_mass": means,
        }
    ).filter(pl.col("mean_mass") > 0)

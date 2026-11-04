"""Sentence-embedding similarity: the encoder cross-check.

A small sentence transformer (MiniLM class) is a good fit for one
narrow job: given the model's short per-candidate relationship
summary ("The Boston Celtics play in Boston, Massachusetts;
Springfield, MA is also in Massachusetts") and a candidate
descriptor, score how close they sit semantically. The LLM supplies
the world knowledge in prose; System 1's literal parsing proposes
support; this scorer cross-checks that proposal before the credal
state is conditioned on it.

The model loads lazily and only when configured. Tests inject a
deterministic fake through the same protocol, so the offline suite
never downloads weights.
"""

from __future__ import annotations

from typing import List, Protocol, Sequence, runtime_checkable


@runtime_checkable
class SimilarityScorer(Protocol):
    """Scores one text against many; returns cosine-like similarities."""

    def similarities(self, text: str, candidates: Sequence[str]) -> List[float]:
        ...


class SentenceTransformerScorer:
    """Lazy sentence-transformers scorer (MiniLM by default config)."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def similarities(self, text: str, candidates: Sequence[str]) -> List[float]:
        model = self._load()
        vectors = model.encode(
            [text, *candidates], normalize_embeddings=True
        )
        anchor = vectors[0]
        return [float(anchor @ vector) for vector in vectors[1:]]

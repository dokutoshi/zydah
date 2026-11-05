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

from typing import Callable, List, Optional, Protocol, Sequence, runtime_checkable


@runtime_checkable
class SimilarityScorer(Protocol):
    """Scores one text against many; returns cosine-like similarities."""

    def similarities(self, text: str, candidates: Sequence[str]) -> List[float]:
        ...


class SentenceTransformerScorer:
    """Lazy sentence-transformers scorer (MiniLM by default config).

    ``on_first_load`` fires exactly once, immediately before the
    model is constructed, so a caller can tell the user what is
    about to happen: the first construction may download the model
    weights (a one-time Hugging Face download for the default
    MiniLM), after which the cached model loads offline. Without a
    callback the scorer stays silent — library code never prints.
    """

    def __init__(
        self,
        model_name: str,
        on_first_load: Optional[Callable[[], None]] = None,
    ) -> None:
        self.model_name = model_name
        self._on_first_load = on_first_load
        self._model = None

    def _load(self):
        if self._model is None:
            if self._on_first_load is not None:
                self._on_first_load()
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def similarities(self, text: str, candidates: Sequence[str]) -> List[float]:
        model = self._load()
        vectors = model.encode(
            [text, *candidates],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        anchor = vectors[0]
        return [float(anchor @ vector) for vector in vectors[1:]]

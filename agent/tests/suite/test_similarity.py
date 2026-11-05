"""SentenceTransformerScorer: first-load notice and quiet encode.

Offline: a fake ``sentence_transformers`` module stands in for the
real one, so no weights are downloaded.
"""

import sys
import types

import numpy as np

from zydah.similarity import SentenceTransformerScorer


class _FakeModel:
    instances = []

    def __init__(self, model_name):
        self.model_name = model_name
        self.encode_kwargs = []
        _FakeModel.instances.append(self)

    def encode(self, texts, **kwargs):
        self.encode_kwargs.append(kwargs)
        return np.eye(len(texts), dtype=float)


def _install_fake(monkeypatch):
    _FakeModel.instances = []
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_first_load_callback_fires_once_and_encode_is_quiet(monkeypatch):
    _install_fake(monkeypatch)
    calls = []
    scorer = SentenceTransformerScorer(
        "fake-model", on_first_load=lambda: calls.append("load")
    )

    first = scorer.similarities("anchor", ["a", "b"])
    second = scorer.similarities("anchor", ["a"])

    assert calls == ["load"]  # exactly once, before first construction
    assert len(_FakeModel.instances) == 1
    assert first == [0.0, 0.0]
    assert second == [0.0]
    for kwargs in _FakeModel.instances[0].encode_kwargs:
        assert kwargs["show_progress_bar"] is False
        assert kwargs["normalize_embeddings"] is True


def test_no_callback_stays_silent_and_loads(monkeypatch):
    _install_fake(monkeypatch)
    scorer = SentenceTransformerScorer("fake-model")
    assert scorer.similarities("anchor", ["a"]) == [0.0]

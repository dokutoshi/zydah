"""Transport/judge/verifier-responder tests (fake opener, no network)."""

import json

import pytest

import zydah.judges as judges_mod
import zydah.verifier as verifier_mod
from zydah.judges import LivePanel, MetaApiJudge, extract_json_object
from zydah.transport import chat_complete, load_transport_config
from zydah.verifier import make_model_responder

FRAME = frozenset({"pass", "partial", "fail"})


def fake_opener_factory(captured, content='{"pass": 0.7, "omega": 0.3}'):
    def opener(url, data, headers, timeout):
        captured["url"] = url
        captured["payload"] = json.loads(data.decode("utf-8"))
        captured["headers"] = headers
        captured["timeout"] = timeout
        return {"choices": [{"message": {"content": content}}]}

    return opener


@pytest.fixture()
def api_env(monkeypatch):
    cfg = load_transport_config()
    monkeypatch.setenv(cfg.base_url_env, "https://api.example.test/v1")
    monkeypatch.setenv(cfg.api_key_env, "test-key")
    monkeypatch.setenv(cfg.model_env, "test-model")
    return cfg


def test_chat_complete_sends_openai_payload(api_env):
    captured = {}
    out = chat_complete("hello", opener=fake_opener_factory(captured, "world"))
    assert out == "world"
    assert captured["url"].endswith("/chat/completions")
    assert captured["payload"]["model"] == "test-model"
    assert captured["payload"]["messages"][0]["content"] == "hello"
    assert captured["headers"]["Authorization"] == "Bearer test-key"


def test_chat_complete_retries_once_on_url_error(api_env):
    from omegaconf import OmegaConf
    import urllib.error

    cfg = OmegaConf.create(
        {
            "base_url_env": "MODEL_API_BASE_URL",
            "api_key_env": "MODEL_API_KEY",
            "model_env": "MODEL_API_MODEL",
            "chat_path": "/chat/completions",
            "timeout_seconds": 5,
            "retries": 1,
            "retry_backoff_seconds": 0,
        }
    )
    calls = {"n": 0}

    def flaky(url, data, headers, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError("flaky")
        return {"choices": [{"message": {"content": "recovered"}}]}

    assert chat_complete("hi", config=cfg, opener=flaky) == "recovered"
    assert calls["n"] == 2


def test_chat_complete_missing_env_raises(monkeypatch):
    cfg = load_transport_config()
    monkeypatch.delenv(cfg.base_url_env, raising=False)
    monkeypatch.setenv(cfg.api_key_env, "k")
    monkeypatch.setenv(cfg.model_env, "m")
    with pytest.raises(RuntimeError, match=cfg.base_url_env):
        chat_complete("hi", opener=fake_opener_factory({}))


def test_extract_json_object_tolerates_fences():
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json_object("no object here")


def test_judge_parses_model_text_into_mass(api_env):
    judge = MetaApiJudge("muse-wonder")
    mass = judge.parse_text('{"pass": 0.7, "partial": 0.1, "omega": 0.2}', FRAME)
    assert sum(mass.values()) == pytest.approx(1.0)


def test_live_panel_elicits_one_mass_per_judge(monkeypatch):
    class FakeJudge(MetaApiJudge):
        def judge(self, evidence, frame):
            return self.parse({"pass": 0.8, "omega": 0.2}, frame)

    panel = LivePanel(FRAME, judges=[FakeJudge("a"), FakeJudge("b")])
    masses = panel.elicit("evidence text")
    assert len(masses) == 2
    assert all(sum(m.values()) == pytest.approx(1.0) for m in masses)


def test_responder_factory_forwards_model_id(monkeypatch):
    seen = {}

    def fake_chat(prompt, model_id=None):
        seen["prompt"] = prompt
        seen["model_id"] = model_id
        return "answer"

    monkeypatch.setattr(verifier_mod, "chat_complete", fake_chat)
    assert make_model_responder()("p", "cheap-model") == "answer"
    assert seen == {"prompt": "p", "model_id": "cheap-model"}

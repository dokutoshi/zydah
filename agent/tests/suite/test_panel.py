"""Panel v1 tests: pydantic YAML config, 2-of-3 quorum, judge sessions."""

import pytest

from zydah.fusion import make_mass
from zydah.judges import LivePanel, MetaApiJudge, load_panel_config

FRAME = frozenset({"pass", "partial", "fail"})


def a_mass():
    return make_mass(FRAME, {"pass": 0.8, FRAME: 0.2})


class OkJudge(MetaApiJudge):
    def judge(self, evidence, frame):
        return a_mass()


class BadJudge(MetaApiJudge):
    def judge(self, evidence, frame):
        raise RuntimeError("judge down")


def test_mass_prompt_anchors_label_meanings():
    from zydah.judges import mass_prompt

    prompt = mass_prompt("some evidence", FRAME)
    assert "fail = the evidence contradicts the claim" in prompt
    assert "omega = the evidence is insufficient to decide" in prompt
    assert "unproven means omega" in prompt


def test_direct_template_anchors_label_meanings():
    from zydah.verifier import load_verifier_config

    template = str(load_verifier_config().direct_mass_prompt_template)
    assert "fail = the evidence contradicts the claim" in template
    assert "unproven means omega" in template


def test_panel_config_defaults_to_spark_and_low_effort():
    cfg = load_panel_config()
    assert cfg.quorum == 2
    assert cfg.effort == "low"
    assert len(cfg.judges) == 3
    # A judge entry without an explicit model defaults to Spark.
    from zydah.judges import PanelJudge

    assert PanelJudge(name="x").model == "muse-spark"
    assert PanelJudge(name="x").effort is None


def test_panel_config_requires_judges_in_yaml(tmp_path):
    import pytest as _pytest
    from pydantic import ValidationError

    bad = tmp_path / "panel.yaml"
    bad.write_text("quorum: 2\neffort: low\n", encoding="utf-8")
    with _pytest.raises(ValidationError):
        load_panel_config(bad)


def test_live_panel_requires_explicit_judges():
    import pytest as _pytest

    with _pytest.raises(TypeError, match="judges"):
        LivePanel(FRAME)  # type: ignore[call-arg]


def test_live_panel_quorum_two_of_three_tolerates_one_failure():
    panel = LivePanel(
        FRAME,
        judges=[OkJudge("a"), BadJudge("b"), OkJudge("c")],
        quorum=2,
        session_ids=["s-1", "s-2", "s-3"],
    )
    masses = panel.elicit(item_id="evidence", run=0)
    assert len(masses) == 2
    assert panel.session_ids == ["s-1", "s-2", "s-3"]
    assert len(panel.last_failures) == 1


def test_live_panel_announces_each_judge_before_asking():
    seen = []
    panel = LivePanel(
        FRAME,
        judges=[OkJudge("a"), BadJudge("b"), OkJudge("c")],
        quorum=2,
        on_judge_start=lambda judge: seen.append(judge.model),
    )
    panel.elicit(item_id="evidence", run=0)
    assert seen == ["a", "b", "c"]


def test_failed_middle_judge_keeps_survivor_identity():
    from zydah.verifier import ZydahVerifier

    panel = LivePanel(
        FRAME,
        judges=[
            MetaApiJudge("muse-wonder", name="wonder"),
            MetaApiJudge("muse-spark", name="spark"),
            MetaApiJudge("muse-spark-open", name="spark-open"),
        ],
        quorum=2,
        session_ids=["sess-1", "sess-2", "sess-3"],
    )
    # Replace judge bodies with deterministic stubs (identity preserved).
    panel.judges[0].judge = lambda evidence, frame: a_mass()  # type: ignore[method-assign]
    panel.judges[1].judge = BadJudge("muse-spark").judge  # type: ignore[method-assign]
    panel.judges[2].judge = lambda evidence, frame: a_mass()  # type: ignore[method-assign]

    verifier = ZydahVerifier(FRAME, panel=panel)
    out = verifier.verify("claim")
    entry = out["history"][0]

    # spark failed: survivors keep their own name/model/session.
    assert [j["name"] for j in entry["judges"]] == ["wonder", "spark-open"]
    assert [j["session_id"] for j in entry["judges"]] == ["sess-1", "sess-3"]
    assert entry["judges"][1]["model"] == "muse-spark-open"
    assert len(entry["judge_masses"]) == 2
    assert out["panel_sessions"] == ["sess-1", "sess-3"]
    # The failure itself is recorded with its identity.
    assert panel.last_outcomes[1]["name"] == "spark"
    assert panel.last_outcomes[1]["mass"] is None
    assert panel.last_outcomes[1]["error"]


def test_live_panel_below_quorum_raises():
    panel = LivePanel(
        FRAME,
        judges=[BadJudge("a"), BadJudge("b"), OkJudge("c")],
        quorum=2,
    )
    with pytest.raises(RuntimeError, match="quorum"):
        panel.elicit(item_id="evidence", run=0)


def test_verify_retains_per_judge_meta_aligned_with_masses():
    from zydah.verifier import ZydahVerifier

    class StubJudge:
        def __init__(self, name, model):
            self.name = name
            self.model = model

    class StubPanel:
        def __init__(self):
            self.judges = [
                StubJudge("wonder", "muse-wonder"),
                StubJudge("spark", "muse-spark"),
                StubJudge("spark-open", "muse-spark-open"),
            ]
            self.session_ids = ["sess-1", "sess-2", "sess-3"]

        def elicit(self, item_id="", run=0):
            return [a_mass(), a_mass(), a_mass()]

    verifier = ZydahVerifier(FRAME, panel=StubPanel())
    out = verifier.verify("claim")
    entry = out["history"][0]
    assert out["panel_sessions"] == ["sess-1", "sess-2", "sess-3"]
    assert [j["name"] for j in entry["judges"]] == [
        "wonder",
        "spark",
        "spark-open",
    ]
    assert entry["judges"][1]["model"] == "muse-spark"
    assert entry["judges"][1]["session_id"] == "sess-2"
    assert len(entry["judge_masses"]) == 3
    assert "mass" in entry  # fused mass retained for frame rendering


def test_judge_sends_its_own_model_and_effort(monkeypatch):
    import zydah.judges as judges_mod

    captured = {}

    def fake_chat(prompt, model_id=None, reasoning_effort=None):
        captured["model_id"] = model_id
        captured["reasoning_effort"] = reasoning_effort
        return '{"pass": 0.8, "omega": 0.2}'

    monkeypatch.setattr(judges_mod, "chat_complete", fake_chat)
    judge = MetaApiJudge("muse-wonder", effort="low")
    judge.judge("evidence", FRAME)
    assert captured["model_id"] == "muse-wonder"
    assert captured["reasoning_effort"] == "low"

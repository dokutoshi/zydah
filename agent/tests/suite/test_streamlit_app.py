"""Streamlit app wiring tests (no UI run, no API calls)."""

import importlib.util
import sys

SPEC = importlib.util.spec_from_file_location(
    "streamlit_app", "tools/streamlit_app.py"
)
streamlit_app = importlib.util.module_from_spec(SPEC)
sys.modules["streamlit_app"] = streamlit_app
SPEC.loader.exec_module(streamlit_app)


def test_cli_panel_model_implies_panel_scoring(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["streamlit_app.py", "--panel-model", "muse-spark"]
    )

    opts = streamlit_app._cli_options()

    assert opts.panel_model == "muse-spark"
    assert opts.panel_scoring is True


def test_build_panel_yaml_model_override_preserves_names_and_effort():
    panel = streamlit_app._build_panel("yaml", "muse-spark")

    assert [j.name for j in panel.judges] == ["wonder", "spark", "spark-open"]
    assert [j.model for j in panel.judges] == ["muse-spark"] * 3
    assert [j.effort for j in panel.judges] == ["low"] * 3


def test_build_panel_base_model_override():
    panel = streamlit_app._build_panel("base", "muse-spark")

    assert [j.name for j in panel.judges] == ["wonder", "spark", "spark-open"]
    assert [j.model for j in panel.judges] == ["muse-spark"] * 3

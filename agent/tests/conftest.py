"""Shared test plumbing: every test gets outputs/{test_name}/.

The autouse fixture creates the directory; the report hook writes a
``result.txt`` with the outcome when the test finishes. Tests can use
the ``output_dir`` fixture to drop richer artifacts (JSON, transcripts).
"""

import json
from pathlib import Path

import pytest

OUTPUTS_ROOT = Path(__file__).resolve().parent.parent / "outputs"


def _safe_name(node_name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in node_name)


def _group_name(request) -> str:
    """Test group from its directory: suite, direct, levels, or misc."""
    parent = Path(request.path).parent.name
    return parent if parent in {"suite", "direct", "levels"} else "misc"


@pytest.fixture(autouse=True)
def output_dir(request):
    path = OUTPUTS_ROOT / _group_name(request) / _safe_name(request.node.name)
    path.mkdir(parents=True, exist_ok=True)
    request.node._output_dir = path
    return path


def dump_json(output_dir: Path, name: str, payload) -> Path:
    target = Path(output_dir) / name
    target.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return target


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "teardown":
        path = getattr(item, "_output_dir", None)
        if path is not None:
            status = "passed" if report.passed else ("failed" if report.failed else "skipped")
            (Path(path) / "result.txt").write_text(f"{item.name}: {status}\n")

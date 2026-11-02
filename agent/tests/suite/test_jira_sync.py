"""Jira sync tests (fake API sender, no network)."""

import importlib.util
import sys

import pytest

SPEC = importlib.util.spec_from_file_location("jira_sync", "tools/jira_sync.py")
jira_sync = importlib.util.module_from_spec(SPEC)
sys.modules["jira_sync"] = jira_sync
SPEC.loader.exec_module(jira_sync)

SAMPLE = """# Open TODOs

## ZYDAH-001 — First task

- Status: open
- Priority: high
- Area: packaging
- Source: user
- Jira: -
- Updated: 2026-11-02
- Details: do the thing
- Acceptance: done when done

## ZYDAH-002 — Second task

- Status: done
- Priority: low
- Area: docs
- Source: old
- Jira: ZYDAH-7
- Closed: 2026-11-02
- Details: already shipped
- Acceptance: none needed
"""


class FakeApi:
    def __init__(self):
        self.issues = {"ZYDAH-7": {"key": "ZYDAH-7", "status": "To Do", "fields": {}}}
        self.calls = []
        self.counter = 100

    def __call__(self, method, path, body=None):
        self.calls.append((method, path))
        if path.startswith("/rest/api/3/search/jql"):
            return {"issues": []} if "zydah-001" in path else {"issues": [{"key": "ZYDAH-7"}]}
        if method == "POST" and path == "/rest/api/3/issue":
            key = f"ZYDAH-{self.counter}"
            self.counter += 1
            self.issues[key] = {"key": key, "status": "To Do", "fields": body["fields"]}
            return {"key": key}
        if method == "PUT":
            return {}
        if path.endswith("/transitions") and method == "GET":
            return {"transitions": [{"id": "31", "to": {"name": "Done"}}, {"id": "11", "to": {"name": "To Do"}}]}
        if path.endswith("/transitions"):
            return {}
        if path.startswith("/rest/api/3/issue/") and method == "GET":
            key = path.split("/")[4].split("?")[0]
            issue = self.issues.get(key, {"key": key, "status": "To Do"})
            return {"fields": {"status": {"name": issue["status"]}}}
        raise AssertionError(f"unexpected call {method} {path}")


def test_parse_todos_multiline_details():
    items = jira_sync.parse_todos(SAMPLE, "OPEN_TODOS.md")
    assert [i.local_id for i in items] == ["ZYDAH-001", "ZYDAH-002"]
    assert items[0].priority == "high"
    assert items[1].jira_key == "ZYDAH-7"
    assert items[0].label == "zydah-001"


def test_issue_fields_carry_local_id_and_labels():
    (item,) = [i for i in jira_sync.parse_todos(SAMPLE, "OPEN_TODOS.md") if i.local_id == "ZYDAH-001"]
    fields = jira_sync.issue_fields(item, "ZYDAH")
    text = " ".join(p["content"][0]["text"] for p in fields["description"]["content"])
    assert "ZYDAH-001" in text and "do the thing" in text
    assert "zydah-001" in fields["labels"] and "packaging" in fields["labels"]
    assert fields["priority"]["name"] == "High"


def test_sync_creates_updates_transitions_and_writes_keys(tmp_path):
    fake = FakeApi()
    client = jira_sync.JiraClient("https://x.test", "e", "t", "ZYDAH", sender=fake)
    items = jira_sync.parse_todos(SAMPLE, "OPEN_TODOS.md")
    updated, plan = jira_sync.sync_items(items, client)
    by_id = {i.local_id: i for i in updated}
    assert by_id["ZYDAH-001"].jira_key.startswith("ZYDAH-")
    assert by_id["ZYDAH-002"].jira_key == "ZYDAH-7"
    actions = {(s.local_id, s.action) for s in plan}
    assert ("ZYDAH-001", "create") in actions
    assert ("ZYDAH-002", "update") in actions
    assert ("ZYDAH-002", "transition") in actions  # done -> Done

    ledger = tmp_path / "OPEN_TODOS.md"
    ledger.write_text(SAMPLE)
    assert jira_sync.write_keys_back(ledger, updated) == 1
    assert f"- Jira: {by_id['ZYDAH-001'].jira_key}" in ledger.read_text()

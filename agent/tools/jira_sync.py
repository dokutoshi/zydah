"""Jira Cloud sync for OPEN_TODOS.md / CLOSED_TODOS.md.

One issue per ``## ZYDAH-###`` section, keyed by a ``zydah-NNN`` label.
Creates missing issues, updates edited ones, transitions statuses, and
writes Jira keys back into the ``Jira:`` fields. Secrets come only from
the env file (default ``src/.env``); unit tests use a fake client.

Usage:
    uv run python tools/jira_sync.py --dry-run
    uv run python tools/jira_sync.py            # live, then writes keys back
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
OPEN_PATH = REPO_ROOT / "OPEN_TODOS.md"
CLOSED_PATH = REPO_ROOT / "CLOSED_TODOS.md"
DEFAULT_ENV = REPO_ROOT / "src" / ".env"

SECTION_RE = re.compile(r"## (ZYDAH-\d+) — (.*?)\n(.*?)(?=\n## ZYDAH-|\Z)", re.S)
FIELD_RE = re.compile(r"^- ([A-Za-z ]+): (.*)$", re.M)
JIRA_LINE_RE = re.compile(r"^(- Jira: ).*$", re.M)

STATUS_TARGET = {"open": "To Do", "blocked": "To Do", "in_progress": "In Progress", "done": "Done"}
PRIORITY_MAP = {"high": "High", "medium": "Medium", "low": "Low"}


@dataclass
class TodoItem:
    local_id: str
    title: str
    status: str
    priority: str
    area: str
    source: str
    date: str
    details: str
    acceptance: str
    verification: str
    jira_key: str
    source_file: str

    @property
    def label(self) -> str:
        return self.local_id.lower().replace("_", "-")


@dataclass
class PlanStep:
    local_id: str
    action: str  # create | update | transition | noop
    key: str
    detail: str = ""


def parse_todos(text: str, source_file: str) -> List[TodoItem]:
    items = []
    for match in SECTION_RE.finditer(text):
        local_id, title, body = match.group(1), match.group(2).strip(), match.group(3)
        fields: Dict[str, str] = {}
        current: Optional[str] = None
        for line in body.splitlines():
            field_match = re.match(r"^- ([A-Za-z ]+): (.*)$", line)
            if field_match:
                current = field_match.group(1).strip()
                fields[current] = field_match.group(2).strip()
            elif current and line.startswith("  "):
                fields[current] += " " + line.strip()
        status = fields.get("Status", "open")
        items.append(
            TodoItem(
                local_id=local_id,
                title=title,
                status=status,
                priority=fields.get("Priority", "medium"),
                area=fields.get("Area", ""),
                source=fields.get("Source", ""),
                date=fields.get("Updated", fields.get("Closed", "")),
                details=fields.get("Details", ""),
                acceptance=fields.get("Acceptance", ""),
                verification=fields.get("Verification", ""),
                jira_key=fields.get("Jira", "-"),
                source_file=source_file,
            )
        )
    return items


def adf_doc(paragraphs: List[str]) -> Dict:
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": text}]}
            for text in paragraphs
            if text
        ],
    }


def issue_fields(item: TodoItem, project_key: str) -> Dict:
    paragraphs = [
        f"Local ID: {item.local_id} (source: {item.source_file}; do not remove)",
        f"Status: {item.status} | Priority: {item.priority} | Area: {item.area}",
        f"Source: {item.source} | Updated: {item.date}",
        "",
        f"Details: {item.details}" if item.details else "",
        f"Acceptance: {item.acceptance}" if item.acceptance else "",
        f"Verification: {item.verification}" if item.verification else "",
    ]
    labels = [item.label]
    if item.area:
        labels.append(re.sub(r"[^a-z0-9]+", "-", item.area.lower()).strip("-"))
    labels.append("closed" if item.status == "done" else "open")
    return {
        "project": {"key": project_key},
        "summary": f"[{item.local_id}] {item.title}",
        "issuetype": {"name": "Task"},
        "description": adf_doc(paragraphs),
        "labels": labels,
        "priority": {"name": PRIORITY_MAP.get(item.priority, "Medium")},
    }


class JiraClient:
    """Minimal Cloud REST client; `sender` is injectable for tests."""

    def __init__(
        self,
        base_url: str,
        email: str,
        token: str,
        project_key: str,
        sender: Optional[Callable] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.project_key = project_key
        self._auth = base64.b64encode(f"{email}:{token}".encode()).decode()
        self._sender = sender or self._http

    def _http(self, method: str, path: str, body: Optional[Dict] = None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Authorization": f"Basic {self._auth}", "Accept": "application/json"},
            method=method,
        )
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read().decode()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Jira {method} {path}: HTTP {exc.code}") from exc

    def find_by_label(self, label: str) -> List[Dict]:
        jql = urllib.parse.quote(f'project = {self.project_key} AND labels = "{label}"')
        result = self._sender("GET", f"/rest/api/3/search/jql?jql={jql}&fields=key,status")
        return result.get("issues", [])

    def create(self, fields: Dict) -> Dict:
        return self._sender("POST", "/rest/api/3/issue", {"fields": fields})

    def update(self, key: str, fields: Dict) -> None:
        self._sender("PUT", f"/rest/api/3/issue/{key}", {"fields": fields})

    def transition_to(self, key: str, current_status: str, target: str) -> Optional[str]:
        if current_status == target:
            return None
        transitions = self._sender("GET", f"/rest/api/3/issue/{key}/transitions")
        for transition in transitions.get("transitions", []):
            if transition.get("to", {}).get("name") == target:
                self._sender("POST", f"/rest/api/3/issue/{key}/transitions", {"transition": {"id": transition["id"]}})
                return target
        return None


def sync_items(items: List[TodoItem], client: JiraClient) -> Tuple[List[TodoItem], List[PlanStep]]:
    """Create/update/transition each item; return items with keys + plan."""
    plan: List[PlanStep] = []
    for item in items:
        fields = issue_fields(item, client.project_key)
        target = STATUS_TARGET.get(item.status, "To Do")
        if item.jira_key and item.jira_key != "-":
            client.update(item.jira_key, fields)
            plan.append(PlanStep(item.local_id, "update", item.jira_key))
        else:
            found = client.find_by_label(item.label)
            if found:
                key = found[0]["key"]
                client.update(key, fields)
                item.jira_key = key
                plan.append(PlanStep(item.local_id, "update", key, "adopted by label"))
            else:
                created = client.create(fields)
                item.jira_key = created["key"]
                plan.append(PlanStep(item.local_id, "create", created["key"]))
        current = item.jira_key and _current_status(client, item.jira_key)
        if current is not None:
            moved = client.transition_to(item.jira_key, current, target)
            if moved:
                plan.append(PlanStep(item.local_id, "transition", item.jira_key, f"{current} -> {moved}"))
            elif current != target:
                plan.append(PlanStep(item.local_id, "noop", item.jira_key, f"no path {current} -> {target}"))
    return items, plan


def _current_status(client: JiraClient, key: str) -> Optional[str]:
    try:
        issue = client._sender("GET", f"/rest/api/3/issue/{key}?fields=status")
        return issue.get("fields", {}).get("status", {}).get("name")
    except RuntimeError:
        return None


def write_keys_back(path: Path, items: List[TodoItem]) -> int:
    """Fill `Jira:` fields for sections whose key changed; return count."""
    text = path.read_text()
    count = 0

    def replace(match):
        nonlocal count
        local_id, title, body = match.group(1), match.group(2), match.group(3)
        for item in items:
            if item.local_id == local_id and item.jira_key and item.jira_key != "-":
                new_body, n = JIRA_LINE_RE.subn(rf"\1{item.jira_key}", body, count=1)
                if n and new_body != body:
                    count += 1
                    body = new_body
        return f"## {local_id} — {title}\n{body}"

    path.write_text(SECTION_RE.sub(replace, text))
    return count


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sync todo ledgers with Jira Cloud.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--env", default=str(DEFAULT_ENV))
    args = parser.parse_args(argv)

    from dotenv import dotenv_values

    env = dotenv_values(args.env)
    missing = [k for k in ("JIRA_BASE_URL", "JIRA_EMAIL", "JIRA_API_TOKEN", "JIRA_PROJECT_KEY") if not env.get(k)]
    if missing:
        print(f"missing keys in {args.env}: {', '.join(missing)}")
        return 2

    open_items = parse_todos(OPEN_PATH.read_text(), "OPEN_TODOS.md")
    closed_items = parse_todos(CLOSED_PATH.read_text(), "CLOSED_TODOS.md")
    print(f"parsed {len(open_items)} open + {len(closed_items)} closed items")

    if args.dry_run:
        for item in open_items + closed_items:
            action = "create" if item.jira_key in ("", "-") else "update"
            target = STATUS_TARGET.get(item.status, "To Do")
            print(f"{action:6} {item.local_id} -> {target}: {item.title}")
        return 0

    client = JiraClient(env["JIRA_BASE_URL"], env["JIRA_EMAIL"], env["JIRA_API_TOKEN"], env["JIRA_PROJECT_KEY"])
    items, plan = sync_items(open_items + closed_items, client)
    for step in plan:
        print(f"{step.action:10} {step.local_id} {step.key} {step.detail}".rstrip())
    n_open = write_keys_back(OPEN_PATH, items)
    n_closed = write_keys_back(CLOSED_PATH, items)
    print(f"keys written back: {n_open} open, {n_closed} closed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

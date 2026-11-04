---
name: jira-sync
description: Sync the OPEN_TODOS/CLOSED_TODOS ledgers to Jira Cloud with branch epics, API-user assignee, and verified status mapping.
---

# Jira Sync

Use when the user asks to sync, resync, check, or repair the Jira Cloud
board for this project, or when ledger `Jira:` fields look stale.

## Source Of Truth

- Ledgers lead: `OPEN_TODOS.md` and `CLOSED_TODOS.md` in the project
  root. One issue per `## ZYDAH-###` section, keyed by a `zydah-NNN`
  label; the script writes returned keys back into `Jira:` fields.
- Never print, paste, or commit secrets. Credentials come only from
  `src/.env` (`JIRA_BASE_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN`,
  `JIRA_PROJECT_KEY`).

## Run

From the project root:

```sh
export UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python
uv run python tools/jira_sync.py --dry-run   # plan only, no writes
uv run python tools/jira_sync.py             # live sync + key writeback
```

The live run assigns every issue to the API user (account ID resolved
from `GET /rest/api/3/myself`) and parents tasks under branch epics
(`day1-foundations`, `day2-evidence-boundary`, `day2-gnn-resolver`,
`trunk-backlog`, keyed by `zydah-epic-*` labels). Re-runs are
idempotent: issues and epics are adopted by label, not recreated.

## Status Mapping — Verify, Do Not Assume

Ledger status maps to Jira status as: `open`/`blocked` -> `To Do`,
`in_progress` -> `In Progress`, `done` -> `Done`. Transitions only
fire when the workflow offers a transition whose target name matches,
so a board can silently drift from the ledger.

After every live run, verify before reporting success:

1. Confirm every ledger item has a non-`-` `Jira:` key and the plan
   output shows no unexplained `noop` (no-path) lines.
2. Spot-check at least one open, one in-progress, and one done item
   via the API (`GET /rest/api/3/issue/{key}?fields=status,assignee,parent`)
   and confirm status, assignee, and epic parent match the ledger.
3. If a status mismatches, inspect
   `GET /rest/api/3/issue/{key}/transitions` for the actual target
   names in that workflow, then fix the mapping in
   `tools/jira_sync.py` (`STATUS_TARGET` / `transition_to`) — never
   hand-edit issues to mask a mapping bug, and never claim sync from
   the script's exit code alone.

## Known Blockers

- `401` on `/myself`: email/token mismatch or revoked token in
  `src/.env`; ask the user to mint a fresh token.
- `404` project or empty createmeta: `JIRA_PROJECT_KEY` does not exist
  for this account; confirm the key in Jira before changing config.
- `400` on create: fetch `GET /rest/api/3/issue/createmeta` for the
  project and compare required fields / issue types / priorities.

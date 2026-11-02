---
name: zydah-verify
description: Run a question/claim through the Zydah belief-fusion verifier (direct elicitation + System 2 loop) and answer with [muse]/[zydah] cycle tags.
---

# Zydah Verify

Use when the user asks to verify, check, or run something "through Zydah":
a question, claim, or agent answer whose dependability matters.

## How to run

From the project root, run the verifier live (loads `.env` for the model
endpoint; direct path uses `verify_direct` with `make_model_responder()` —
the judge panel is suspended until panel routing is fixed):

```sh
source .venv/bin/activate
set -a; . ./.env; set +a
python3 - << 'PY'
from zydah.verifier import ZydahVerifier, make_model_responder, format_transcript
frame = frozenset({"pass", "partial", "fail"})
v = ZydahVerifier(frame=frame, responder=make_model_responder())
out = v.verify_direct("CLAIM-UNDER-REVIEW")
z = out["result"]["zydah"]
print(f"verdict={out['verdict']} followups={out['followups']} model={out['model_id']}")
print(f"bel={z['bel']:.4f} pl={z['pl']:.4f} width={z['width']:.4f} conflict={z['conflict']:.4f} decision={z['decision']}")
print(format_transcript(out["history"]))
PY
```

Replace `CLAIM-UNDER-REVIEW` with the claim being checked, phrased as
"Claim to evaluate: ...". Thresholds live in `src/zydah/conf/verifier.yaml`.

## How to answer

Reply with cycle attribution:

- `[muse]` — your first, direct answer (before verification).
- `[zydah]` — the verifier outcome computed from the FINAL state: final
  verdict, decision, Bel/Pl, width, conflict, follow-ups, the
  `format_transcript` prompt/response pairs, and the verified final answer.

Final-state rule: never write `[zydah]` from an intermediate incident
(such as a mid-turn blocker). The verifier only sees what it is passed —
pass it the completion state, then report its result. State plainly when
the verdict is speculative (`verdict=False`, wide interval) instead of
smoothing it over. Closed world only: no web fetch inside the cycle
unless the user asks for open-world mode.

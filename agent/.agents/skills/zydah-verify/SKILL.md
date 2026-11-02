---
name: zydah-verify
description: Run a question/claim through the Zydah belief-fusion verifier (fused panel + System 2 loop) and answer with [muse]/[zydah] cycle tags.
---

# Zydah Verify

Use when the user asks to verify, check, or run something "through Zydah":
a question, claim, or agent answer whose dependability matters.

## How to run

From the project root, run the verifier live (loads `.env` for the model
endpoint; judges come from `LivePanel`, System 2 from
`make_model_responder()`):

```sh
source .venv/bin/activate
set -a; . ./.env; set +a
python3 - << 'PY'
from zydah.judges import LivePanel
from zydah.verifier import ZydahVerifier, make_model_responder
frame = frozenset({"pass", "partial", "fail"})
v = ZydahVerifier(frame=frame, panel=LivePanel(frame), responder=make_model_responder())
out = v.verify("CLAIM-UNDER-REVIEW")
z = out["result"]["zydah"]
print(f"verdict={out['verdict']} followups={out['followups']} model={out['model_id']}")
print(f"bel={z['bel']:.4f} pl={z['pl']:.4f} width={z['width']:.4f} conflict={z['conflict']:.4f} decision={z['decision']}")
PY
```

Replace `CLAIM-UNDER-REVIEW` with the claim being checked, phrased as
"Claim to evaluate: ...". Thresholds live in `src/zydah/conf/verifier.yaml`.

## How to answer

Reply with cycle attribution:

- `[muse]` — your first, direct answer (before verification).
- `[zydah]` — the verifier outcome: final verdict, decision, Bel/Pl,
  width, conflict, follow-ups, and the verified final answer.

State plainly when the verdict is speculative (`verdict=False`, wide
interval) instead of smoothing it over. Closed world only: no web fetch
inside the cycle unless the user asks for open-world mode.

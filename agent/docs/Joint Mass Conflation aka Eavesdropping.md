# Joint Mass Conflation, aka Eavesdropping

*Why a fused verdict can look independent while quietly depending on
earlier conversation — and the defenses this codebase puts around it.*

## 1. The pathology

Dempster-Shafer fusion assumes its sources commit mass from what they
independently observed. The combination rules in `src/zydah/fusion.py`
are only as meaningful as that independence. Conflict mass `K` reads as
"independent judges disagreeing." Belief reads as corroboration.

That reading breaks when one "source" has heard the others — or has
heard the fusion process itself — before committing mass. Its mass is
then not a second observation. It is the conversation wearing a second
hat. We call this **joint mass conflation**, or, operationally,
**eavesdropping**:

> A later mass assignment is conditioned on earlier turns, prompts,
> follow-up questions, or validator output, yet is fused and reported
> as if it were an independent observation.

The damage is specific. Suppose an evaluator elicits mass `m1`, finds
the interval too wide, asks a follow-up, appends the answer to the
evidence, and elicits `m2`. If the same model produced the follow-up
and then scores the enlarged evidence, `m2` is downstream of `m1`'s
weakness and of the validator's question. Fusing them would count a
single reasoning chain twice: corroboration that never happened. Even
when nothing is explicitly fused, presenting the second pass as
"independent verification" commits the same conflation rhetorically.

Two concrete instances we hit while building this system:

1. **In-loop conflation.** The verifier appended each System 2 answer
   to the evidence string and re-elicited mass from the joined text.
   The re-elicitation was conditioned on the validator's own prompt
   (the thing asking "what would reduce uncertainty?"), so a converging
   interval reflected the prompt scaffold, not new observation.
2. **Chat-context conflation.** In this client, one session has one
   unified transcript; switching models changes only who answers later
   turns. Any `[zydah]` text emitted in a session remains in context
   for whatever model answers next. A "first response" collected after
   verification output exists is not a first response.

## 2. Everyday examples

**The dinner table.** "Who thought the soup was too salty?" asked
around the table in order. The first three say yes; everyone after is
now answering two questions — the soup, and the table. Count the
hands and you get 8/8 "independent" confirmations of oversalting,
most of which are eavesdropped. The independent version is a secret
ballot: same soup, no listening in.

**Fast follower vs first mover.** A first mover ships a product and
publishes its numbers. A fast follower studies them, copies what
worked, and launches to applause: "the market has independently
validated this design — look, two products converged on it." They did
not converge. One observed; the other overheard. Fusing their
outcomes as independent market evidence double-counts the first
mover's experiment.

**Eyewitnesses in the hallway.** Two witnesses compare stories before
giving statements, then give matching accounts. The match is now
evidence of a shared conversation at least as much as of a shared
event. Investigators separate witnesses first for exactly this
reason — the human version of session isolation.

**The anchored code review.** A reviewer opens a pull request that
already carries the author's self-review: "the risky part is fine, I
checked." Even a diligent reviewer now audits around that frame.
Their approval is real work, but it is assisted, not independent —
and a process that counts author plus reviewer as two independent
checks has conflated them.

The pattern is constant: the second judgment may still be honest,
skilled, and useful. What it is not is *another* observation.

## 3. Two kinds of dependence

Keep the failure modes separate, because the fixes differ.

**Contextual joint mass (eavesdropping proper).** The dependence runs
through shared context: a transcript, an appended follow-up, a prompt
that quotes validator metrics (`width`, `conflict`, `verdict`). The
source did not independently look at the world; it listened in.

**Model joint mass (correlation).** Two nominally distinct sources
share weights, training data, or a serving deployment, so their errors
correlate even in separate contexts. Separate prompts do not cure
shared priors.

This document's defenses target contextual joint mass. Model-level
correlation is real but separate, and is deferred to the Playgrounds
work (mixed models, separate deployments) — it must be *labeled*, not
silently upgraded.

## 4. The independence contract

A fused or verified verdict may be called independent only when all of
the following hold for the mass that produced it:

- **No eavesdropping:** the eliciting prompt contained no prior
  validator output, no `[zydah]` text, and no follow-up answers — only
  original evidence.
- **Session separation:** the first response was captured in a context
  that never contained the verification transcript. Same session ⇒
  joint, unconditionally.
- **Provenance is recorded:** every evidence chunk states kind
  (original / follow-up / external), model, session, playground, and
  turn, so dependence is auditable after the fact.
- **Immutability:** the first response text is captured before
  verification exists and sealed; later output cannot revise it.

Violating any clause does not make the verdict useless. It makes the
label wrong. The honest labels implemented in `src/zydah/evidence.py`:

| Mode | Meaning |
| --- | --- |
| `CONTEXT_INDEPENDENT` | Separate session, original-only evidence, no follow-ups consumed. |
| `ASSISTED` | Follow-up answers or unproven isolation contributed. |
| `JOINT_CONTEXT` | First response and verification shared a session: eavesdropped by construction. |

## 5. Defenses in this codebase

**Provenance ledger** (`src/zydah/evidence.py`). Evidence is chunks,
not a growing string. `original_only()` renders only caller-supplied
evidence; `assisted()` renders everything and says so. Iteration zero
always elicits from the original-only view, so the first mass of a
cycle is eavesdropping-free by construction.

**Session rule** (`classify_independence`). If the first response's
session equals the verifier's session, the mode is `JOINT_CONTEXT` —
a hard rule, not a threshold. Any follow-up use downgrades to
`ASSISTED`. Only separate sessions with zero follow-ups earn
`CONTEXT_INDEPENDENT`.

**Sealed first responses** (`src/zydah/interface.py`). The endpoint's
first text is captured with `capture_first_response` (which rejects
assistant-prose provenance) and paired with results via `seal_cycle`.
The `[muse]` block of a sealed cycle is byte-for-byte the pre-
verification text. This is what stops intermediate incidents — a
mid-turn blocker, an early speculation — from being retroactively
presented as the verified outcome, and vice versa.

**Two playgrounds.** Playground A generates first responses; it runs
no Zydah judges and never sees validator output. Playground B
verifies sealed A outputs and owns all follow-up questions. Feedback
from B may inform a later cycle only as new, labeled assisted/external
evidence — never as "original."

**Cycle logging.** Each `verify_direct` run can append JSONL records
(run ID, session IDs, independence mode, metrics) so an auditor can
reconstruct which ears heard what.

## 6. What this does not solve

- **Model-level correlation.** Playground A and B can still share one
  deployment today; contextual independence is not model independence.
  Provenance records effective model identity precisely so this is
  visible rather than laundered.
- **Manual contamination.** Nothing stops a human from pasting a
  `[zydah]` block into a fresh prompt by hand. The contract is a
  construction discipline at the API boundary, not a proof about all
  possible transcripts.
- **Honest uncertainty.** Independence hygiene does not make mass
  assignments *good*. A separately sessioned model can still park
  everything on Ω. That is a calibration problem, orthogonal to
  conflation.

## 7. How to recognize eavesdropping in the wild

- A "second opinion" whose prompt quotes the first opinion, the
  disagreement, or the question designed to resolve it.
- Agreement that arrives suspiciously fast after a follow-up that
  telegraphed the desired resolution.
- Verification labels that name the method ("independently verified")
  without naming the sessions, prompts, and models that made it
  independent.
- Verdicts whose provenance cannot be reconstructed: no recorded
  prompt, no session, no turn order.

The test to apply: *could this source have produced exactly this mass
without having heard the conversation?* If the honest answer is no —
or unknowable — it is joint mass, whatever the pipeline called it.

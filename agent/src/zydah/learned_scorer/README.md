# Learned Scorer (GNN Resolver)

Status: schema built (ZYDAH-113); baseline resolver, learned scorer, and
verifier wiring are not yet implemented (ZYDAH-114, ZYDAH-117,
ZYDAH-115). This directory will hold the PyTorch Geometric scorer. The
graph schema it consumes already lives in `src/zydah/graph.py`, with
hyperparameters in `src/zydah/conf/graph.yaml`.

## Are the graphs pre-built taxonomies?

Today, yes. As built in ZYDAH-113, the graphs are pre-built fixture
taxonomies: candidate datasets are embedded in `src/zydah/graph.py`
(`SPRINGFIELDS`, `VEHICLES`, `ANIMALS`), partition thresholds come from
`src/zydah/conf/graph.yaml`, and the verifier does not interrogate them
yet. But a taxonomy is only the skeleton; the graph adds candidate
attributes, typed relations (`faster_than`), and context affinity edges
on top of it.

Node kinds: `ambiguity` (root supernode, e.g. `Springfield (?)`),
`partition` (North/South/East/West, timezone; for comparisons: model
class, metric, fastest-vs-typical interpretation), `candidate`
(Springfield IL, BMW M4, peregrine falcon), and `context`
(session/user signals).

## How the verifier will interrogate the graph

This is the ZYDAH-114/115 design; the resolver proposes evidence and
never sets the final verdict.

1. **Trigger** — the validator flags a response (interval width and/or
   conflict above the `verifier.yaml` thresholds). The ambiguous term
   and blocking assumptions select which graph to build.
2. **Attach context** — session/user signals become `context` nodes
   with affinity edges. Original-only context keeps the evidence
   independent; follow-up-derived context is labeled `ASSISTED`.
3. **Score** — the resolver propagates context -> partitions ->
   candidates (deterministic numpy message passing in ZYDAH-114,
   learned PyG scorer in ZYDAH-117; both behind the same `Resolver`
   protocol). Output: candidate scores, partition scores, entropy
   before/after.
4. **Convert to Dempster-Shafer mass** — a confident candidate becomes
   singleton mass; a confident partition with uncertain members
   becomes subset focal mass (e.g. `{IL, MO, OH}`); residual entropy
   goes to Omega. This is where uncertainty actually shrinks: mass
   leaves Omega for structured hypotheses instead of a forced
   singleton.
5. **Test the claim over the resolved set** — if the claim's outcome
   is invariant across the high-mass candidates, the ambiguity did
   not matter and the verdict can commit. If outcomes differ, the
   graph emits a conditional answer or identifies the missing
   discriminator.
6. **Re-elicit, do not self-confirm** — the resolution enters the
   evidence ledger as a labeled graph-resolution chunk, the elicitor
   re-scores, and the validator rechecks interval width against
   `verifier.yaml`. 
7. **Escalate only on residual entropy** — if mass is still spread,
   the `[human]` question is generated at the partition with maximum
   expected information gain (a 4-way North/South/East/West split
   beats a binary "is it Illinois?" when candidate mass is spread),
   and the follow-up loop pauses as `needs_human` instead of burning
   its Hydra turn budget.

## Framework

Chosen in ZYDAH-112: PyTorch Geometric 2.8.1 on torch 2.15.0,
installed as the optional `graph` extra (`uv add --optional graph`).
PyG requires the project Python 3.12 (it crashes on 3.15). Device
selection is `device: auto` in `graph.yaml`: MPS (Metal) when
available on Apple hardware, CPU otherwise — Metal is an optional
accelerator plugin, never a requirement, and these graphs are small.

The learned scorer ships only if it beats the deterministic baseline
on held-out entities (candidate accuracy and calibration); otherwise
the baseline remains the resolver and the GNN result is recorded as a
finding (ZYDAH-117).

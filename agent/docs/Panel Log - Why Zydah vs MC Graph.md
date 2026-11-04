# Panel Log — Why Zydah vs MC Graph

Examples collected for panel discussions. Each entry is a concrete
case, what a static graph baseline does with it, and what Zydah is
expected to do instead.

## 2026-11-02 — The Celtics clue

**Prompt state.** Springfield without a state. The follow-up loop
offered direction choices (East/North/South/West/Other). The human
answered via `Other` with free text:

> "the celtics are a good team"

**What the mock artifact showed (the failure).**
`outputs/direct/test_other_clue_gets_directional_frame/other_clue.json`
is a unit-plumbing artifact: scripted responder, scripted elicitor
masses. The clue never passed through Zydah's elicitation at all, and
the transmitted context still showed every Springfield candidate at
an equal 0.125 — the *pre-clue prior* presented as if it were a
post-clue conclusion. A static MC-style graph does the same thing:
transition weights do not move because a human said something.

**Why this is the Zydah case.** The clue is indirect but informative:
Celtics -> Boston -> Massachusetts -> Springfield, MA (and East).
Zydah's job is to convert that human utterance into evidential
signal:

1. Send the clue to the model anchored by the offered choices and
   candidate frame (directionality context), not as bare text.
2. Take the model's mapping (candidate/partition) as a labeled
   human-feedback chunk — an RLHF-style signal, recorded as such.
3. Re-score the graph with that signal as context: Springfield, MA
   and the East partition gain mass; the residual stays on Omega,
   discounted by how indirect the clue is. Weights after the clue
   must not equal weights before it.
4. Re-elicit the claim under the updated candidate distribution and
   let Bel/Pl width decide the next action (commit, conditional
   answer, or the next choice round).

**Falsifiable check.** In the live variant
(`test_live_other_clue_through_zydah`), the post-clue model response
and re-elicited width are recorded verbatim. If the candidate
distribution is unchanged after the clue, Zydah has added nothing
over the static graph for this case and the entry should be read as
a failure, not a demo.

**Live outcome (2026-11-02).** Failure, as defined above. The live
model received the clue anchored by the directional frame and still
replied that with `partition = Other` there is "no additional
context", listing all 8 candidates at 0.125; width stayed 1.0 and
the loop paused for a human again
(`outputs/levels/test_live_other_clue_through_zydah/celtics_clue.json`).
Two lessons: (1) presenting pre-clue equal scores inside the
follow-up anchors the model on "nothing changed" — post-clue
re-scoring must happen in System 1 (entity-link the clue into graph
context: team -> city -> state -> candidate) before or alongside
the model call, not be delegated to the model's reading of a prior
table; (2) an `Other` partition label reads as "no partition" to
the model, so the frame must name the mapping task explicitly.

**Lesson.** Never present pre-clue priors as post-clue scores in an
artifact. Label which side of the clue a number was computed on.

**Math correction (later, 2026-11-02).** The 0.125 values are the
pignistic transform of the vacuous mass, `BetP(w) = m(Omega)/|Omega|`;
they are not beliefs. The credal state before the clue is
`m(Omega) = 1`, so every Springfield singleton has `Bel = 0` and
`Pl = 1`. The resolver now keeps those channels separate:
`candidate_scores` is recomputed as `pignistic(mass)`, while
`candidate_intervals` carries `[Bel, Pl]` per candidate. Choosing an
offered multiple-choice support conditions the prior by building a
discounted categorical source over that focal subset and combining
with Yager's rule (conflict returns to Omega), so
`Bel(A | choice)` / `Pl(A | choice)` — and only then the pignistic
distribution — redistribute. A free-text clue does not condition the
state until it has been mapped to a support focal element; the
payload therefore labels its numbers as the pre-clue pignistic and
vacuous intervals rather than as post-clue beliefs.

**Relationship update (later, 2026-11-02).** The follow-up now asks
the LLM, per candidate: "How does 'the celtics are a good team'
relate to Springfield, MA? Identify any city, state, or regional
relationship, or say there is none." The LLM is the right tool for
this — it already knows Celtics -> Boston, MA; a sentence
transformer or a hand-built entity table would be a weaker,
redundant copy of that knowledge. System 1 stays deterministic: it
matches candidate state names/labels in the model's answer and
Yager-conditions the credal state on that support, discounted by
`resolver.conditioning_alpha` (0.75) because the source is a model,
not the human. Scripted-model evidence in
`outputs/direct/test_relationship_answer_conditions_massachusetts/celtics_relationship.json`:
after "The Celtics are Boston's NBA team; Boston and Springfield
are both in Massachusetts", support = Springfield, MA, pignistic
weights move 0.125 uniform -> 0.78125 (MA) vs 0.03125 (each other),
and Bel(Springfield, MA) moves 0 -> 0.75 with Pl = 1.0. A negative
answer ("unrelated") conditions nothing.

**Live confirmation (2026-11-02, after per-candidate parsing).**
The first live rerun exposed three parser gaps — answers that both
affirm MA and end with a question were routed to the human before
extraction; negated sections for other states leaked state-name
mentions into support; enumerations of the whole candidate set
counted as evidence. After fixing extraction order, per-candidate
negation, heading detection, and slash normalization, the live run
closes `verdict: good`, `needs_human: false`, `followups: 1`, with
support = Springfield, MA alone and the same conditioned weights
(0.78125 vs 0.03125, Bel 0.75)
(`outputs/levels/test_live_other_clue_through_zydah/celtics_clue.json`).

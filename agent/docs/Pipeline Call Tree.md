# Pipeline Call Tree — LLM / Zydah

For review. Status column is literal: **Active** = runs in the live
demo today; **Inactive** = implemented and tested, not on the demo
path; **Not built** = planned only.

## Components

| Component | What it is | Status |
|---|---|---|
| LLM transport | OpenAI-compatible `chat_complete` (`src/zydah/transport.py`); endpoint/key/model from env, `MODEL_API_*` | Active |
| First response (`[muse]`) | One live model call per question, sealed via `capture_first_response` before any verifier output exists | Active |
| Judge panel | 3 judges — Muse Wonder, Muse Spark, Muse Spark Open (`PANEL` in `src/zydah/judges.py`), one mass each, fused by Yager | **Inactive on non-panel path** |
| Panel path (`verify` / `fuse_response`) | Panel masses -> `fused_verdict` [Bel, Pl] -> validator -> follow-up loop | Inactive (tests only) |
| Non-panel path (`verify_direct`) | Panel bypassed: the model scores the evidence itself, one mass per turn via `make_mass_elicitor` -> `chat_complete`. The demo driver (`tools/live_demo.py`) runs this path | **Active** |
| Validator | Width and conflict vs 0.35 thresholds from `conf/verifier.yaml`; `verdict=good` or structured follow-up | Active |
| Graph resolver | `NumpyResolver` over `CandidateGraph` (Springfield places, comparisons); partitions, Yager conditioning on support at source-dependent alpha (human 1.0, model 0.75) | **Active** (Springfield questions) |
| Learned GNN scorer | PyTorch Geometric model to replace/augment `NumpyResolver` scoring (ZYDAH-116/117); MPS verified on the Neo | **Not built** |
| MiniLM encoder | `sentence-transformers/all-MiniLM-L6-v2` cross-check: a literally-parsed relationship support survives only if the candidate's summary scores >= `similarity_threshold` (0.25) against its descriptor | Active (when graph wired) |
| Human (`[human]`) | Terminal multiple choice (direction partitions + `Other`), Pydantic `ChoiceKey`; answers recorded as RLHF `human_feedback` | Active |
| Belief frame JSON | Per turn: `{"frame": [...], "mass": {focal: w, "Omega": w}}`; per candidate: `(A, ~A, Omega)` tuples. Pignistic is derived, never stored as the value | Active |

## Call tree (live demo)

```
tools/live_demo.py main()
|-- terminal_input("[you] ")                     # controlling terminal; type-ahead flushed
|-- run_question(question)
|   |-- chat_complete(question)                  # LLM call 1: first response
|   |-- [muse] <response>
|   |-- if response asks a question (_asks_human):
|   |     [you: answer] <human text>             # e.g. "the celtics are a great team"
|   |     chat_complete(clarification + answer)  # LLM call 2: substantive response
|   |     [muse] <response>                      # loop while it keeps asking
|   |-- capture_first_response(...)              # sealed [muse], separate session
|   |-- build_verifier(...)                      # Springfield -> NumpyResolver + place graph
|   |                                              + SentenceTransformerScorer (lazy)
|   |-- ZydahVerifier.verify_direct(evidence)    # evidence = Question/Response/Human answer
|   |   |-- mass_elicitor(view) -> chat_complete # one mass per turn (NOT judges)
|   |   |-- fused_verdict([mass])                # [Bel, Pl], width, conflict
|   |   |-- validate()                           # width/conflict <= 0.35 ?
|   |   |-- if flagged:
|   |   |     followup JSON -> responder -> chat_complete   # LLM answers System 1
|   |   |     NumpyResolver.resolve(graph)       # deduction before routing
|   |   |     |-- unresolved -> needs_human + partition choices (East/North/South/West/Other)
|   |   |     |-- question-shaped -> needs_human (plain question, no graph)
|   |   |     '-- resolved -> GRAPH_RESOLUTION chunk appended; loop
|   |   '-- verdict good -> return
|   |-- render_result()                          # [zydah] <sentence> \n verdict=...
|   '-- interact() while needs_human:
|         [you: choice key] B  (or E + free text)
|         resume_direct(previous, key, other_text)
|           |-- HumanFollowup JSON -> responder -> chat_complete
|           |-- _relationship_conditioning(answer):
|           |     literal per-candidate parse (negation, enumerations, headings)
|           |     MiniLM similarities(summary, descriptor) >= 0.25 ?
|           |     Yager condition on support (alpha 0.75, model source)
|           |     -> belief_frame + (A, ~A, Omega) per candidate
|           |-- model question? -> next candidate choice round (subset shrinks)
|           '-- re-elicit -> validate -> good / next follow-up / pause
|-- after_verdict()                              # pause announced unsaved; resolved saved + follow-up invite; loop back to [you]
'-- SaveTee (--save)                             # every emitted line -> transcript in JSON
```

## Prompt / response flow

```
[you]      What is the weather in Springfield today?
   |
   v
[muse]     (live model, sealed first response)
           "Which Springfield do you mean? ..."
   |
   |  response is a question -> route to human, do NOT score it
   v
[human]    the model asked you a question; answer it here.
[you]      the celtics are a great team
   |
   v
[muse]     (model continues with the human answer in context)
           "Since the Celtics are Boston's team, Springfield, MA ..."
   |
   v
[zydah]    System 1 scores the evidence (one mass per turn):
           belief_frame turn 1: {Omega: 1.0}  ->  adjusted per turn
           width/conflict <= 0.35 ?
           |-- yes -> [zydah] the response checks out.
           |          verdict=good
           '-- no  -> follow-up JSON to the model, and/or:
                      graph deduction -> [human] A. East B. North ...
                      [you: choice key] B
                      -> HumanFollowup JSON to the model
                      -> relationship conditioning (MiniLM cross-check)
                      -> Springfield, MA (A 0.75, ~A 0.0, Omega 0.25)
                      -> re-score -> verdict

[zydah]    press Enter to ask another question (Ctrl-C to quit).
```

## Panel path (inactive in the demo)

```
ZydahVerifier.verify(evidence)
|-- fuse_response(evidence, panel)
|   |-- panel.elicit(run) -> 3 judge masses     # Wonder / Spark / Spark Open
|   |-- direct_verdict([lead mass])             # [muse] arm
|   '-- fused_verdict(masses, rule=yager)       # [zydah] arm: Bel/Pl/conflict
|-- validate() -> follow-up loop (same routing as above)
```

Panel routing issues (sequential latency, judges returning all-Omega)
are why the demo runs the non-panel path. The panel contract — 3 judges,
never fewer — is unchanged; it is simply not the path under test here.

## How Zydah steers the context (verified against the code)

There are no judges on the non-panel path, so steering is not done
by a panel vote. Zydah (System 1, deterministic) steers by controlling
**what text each model call sees** and **what message the model must
answer next**:

1. **Evidence assembly (demo).** `run_question` concatenates
   `Question / Response / Human answer` lines into one evidence
   string. Nothing else is visible to the verifier's elicitor.
2. **Ledger views (`verify_direct`).** Turn 1 is elicited on
   `ledger.original_only()` — only the sealed original evidence, so
   the first mass cannot be contaminated by verifier output
   (anti-eavesdropping). Later turns use `ledger.assisted()`: all
   chunks joined by the YAML `additional_info_prefix`
   (`"\n\nAdditional information:\n"`), labeled `ASSISTED`.
3. **The follow-up JSON is the steering message.** When the validator
   flags (width or conflict > 0.35), `serialize_followup()` output is
   transmitted verbatim to the responder model — clarifying question,
   summary, blocking assumptions. The model's answer returns as a
   `MODEL_FOLLOWUP` ledger chunk (with the verifier's session id) and
   the loop re-elicits. Repeat until width <= 0.35 or
   `max_iterations` (5) is exhausted.
4. **Graph steering.** After a follow-up answer, `NumpyResolver`
   runs before any routing: a resolved graph appends a labeled
   `GRAPH_RESOLUTION` chunk ("Graph resolution (...): best_partition=
   ...; top candidates: ...") into the assisted view, so the next
   elicitation literally reads the deduction. An unresolved graph
   instead stops the loop and generates the `[human]` multiple-choice
   question from its partitions.
5. **Human steering.** A human key press becomes a `HumanFollowup`
   JSON sent to the model: the choice value plus expanded graph
   context (candidates, `(A, ~A, Omega)` beliefs, relationship
   questions, original question). For an `Other` clue, the model's
   per-candidate relationship answer is parsed, cross-checked by
   MiniLM, and Yager-conditioned; the conditioned state re-enters as
   another `GRAPH_RESOLUTION` chunk. The human's selection is also
   recorded as `human_feedback` (the RLHF signal).

So in the demo the LLM plays three single-model roles — first
responder, mass elicitor (one mass per turn), and follow-up responder
— while Zydah owns the state: what enters the ledger, in what order,
under which session label, and when the loop stops. That is the
steering. Judge-panel steering (3 masses fused per turn) exists only
on the inactive `verify` path above.

### Is the ledger passed to the model as a prompt?

Its **text** is; its structure is not.

- The mass elicitor receives `direct_mass_prompt_template` from
  `verifier.yaml` with `{evidence}` filled by a ledger view: turn 1
  `original_only()` text, later turns the `assisted()` concatenation
  of chunk texts joined by `additional_info_prefix`
  (`"\n\nAdditional information:\n"`). The model sees the words, and
  returns one JSON mass.
- The responder never sees the ledger. It receives exactly two kinds
  of message, both JSON: the validator's follow-up
  (`serialize_followup`) and, after a human choice, the
  `HumanFollowup` (choice + expanded graph context).
- Chunk kinds, session ids, and provenance labels stay inside the
  ledger. They drive independence classification (`CONTEXT_INDEPENDENT`
  / `ASSISTED` / `JOINT_CONTEXT`) and are never shown to the model —
  that separation is the anti-eavesdropping boundary.

## Human response routing: does it reach the graph resolver?

Short answer: **not directly.** Human text is never fed to the
resolver or scored by Zydah as-is. It routes to the model first;
the resolver engages only through the structured-choice machinery
around that model call.

```
                        HUMAN RESPONSE
                              |
              +---------------+---------------+
              |                               |
     free-form text                    structured choice
  ([you: answer] / gate)            (paused [you: choice key])
              |                               |
        to the MODEL               GATE: paused result with
      (continuation /              `choices` + key validates?
       direction)                   (Other needs free text)
              |                          |           |
           [muse]                       no          yes
                                          |           |
                                     ValueError   +---+-------------+
                                     (fail closed)|                 |
                                          partition/candidate   Other + free text
                                          key                   (clue)
                                                  |                 |
                                          _choice_context    _other_context
                                          graph rebuilt      candidates +
                                          resolve() +        relationship_questions
                                          Yager condition          |
                                          (human alpha)       HumanFollowup JSON
                                                  |                 |
                                                  |            [muse] answers
                                                  |                 |
                                                  |        _relationship_conditioning
                                                  |        resolver configured?
                                                  |        per-candidate parse
                                                  |        + MiniLM cross-check
                                                  |           |            |
                                                  |        support      none + answer
                                                  |        found        ends in `?`
                                                  |           |            |
                                                  +-----+-----+      back to HUMAN
                                                        |         (candidate subset
                                              candidate_beliefs    as next choices)
                                                (A, ~A, Omega)
                                                        |
                                              HumanFollowup JSON -> [muse]
                                                        |
                                              revised answer re-elicited
                                              -> Zydah scores (Bel/Pl)
```

Two qualifications keep the diagram honest:

- A partition/candidate **key selection does reach the resolver**:
  `_choice_context` rebuilds the graph and conditions the chosen
  members (human alpha) before the model is called. The human's
  *text* still never goes to the resolver — the conditioned
  `(A, ~A, Omega)` beliefs ride to the model inside the
  `HumanFollowup` JSON, and Zydah scores the model's revised
  answer, not the human's words.
- Precondition above all of it: a resolver exists only if one was
  injected. In the demo, `build_verifier` adds `NumpyResolver` +
  place graph + MiniLM only when "springfield" appears in the
  question or first response. No graph configured, no right side.

For contrast, model answers face the graph on every flagged
follow-up (deduction before routing, `src/zydah/verifier.py`):
even a question-shaped answer first runs `_resolve_with_graph`;
unresolved generates the `[human]` multiple-choice question
above, resolved appends a `GRAPH_RESOLUTION` evidence chunk and
the agent continues without the human.

## Negation: Zydah before MiniLM (veto semantics)

Correct — with one precise condition on what "before" means.

MiniLM is a bi-encoder: "Celtics are in Boston" and "Celtics are
**not** in Boston" embed almost identically. No amount of belief
JSON prepended to its input text fixes that — the encoder has no
polarity to reason with, and JSON tokens just pollute the
similarity.

What solves it is veto semantics, which is what your ordering
gives if implemented this way:

```
human free text
  -> Zydah System 1: belief JSON context (live candidates, partitions, (A, ~A, Ω))
  -> polarity parse per candidate clause (verifier.yaml +/- markers)
       negative marker in the candidate's clause  =>  NO support. Final.
  -> MiniLM cross-check ONLY inside the affirmed set (topicality, threshold)
  -> Yager condition candidate_beliefs (human alpha)
```

- Negation is decided by symbolic code that can actually see the
  word "not" / "no relationship" in the right clause.
- MiniLM never gets a vote on polarity. It can only confirm an
  already-affirmed mention is topical, or withhold support below
  threshold.
- The belief JSON context is what makes the parse per-candidate
  instead of global: System 1 knows which candidates are live and
  what mass they carry before any similarity is computed.

That's also the safe failure mode: unrecognized negation
(sarcasm, "wrong coast") matches no affirmative — conditioning
simply doesn't fire and beliefs stay uniform, rather than
conditioning on the wrong candidate.

So: yes, Zydah-first solves the challenge iff MiniLM is
downstream of the polarity gate with no override. If MiniLM's
score can resurrect a negated candidate, the bug is still there
regardless of order.

## Invariants

- `[muse]` is always the sealed first response; `[zydah]` is always
  the final verifier state, never a mid-turn blocker.
- First-response session != verifier session, or the cycle is labeled
  `JOINT_CONTEXT`; follow-up-assisted cycles are labeled `ASSISTED`.
- A scalar is never the stored value: belief frames and
  `(A, ~A, Omega)` tuples are; pignistic/Bel/Pl derive from them.
- Every threshold, marker list, and model name lives in
  `src/zydah/conf/*.yaml`; no hyperparameters in code.

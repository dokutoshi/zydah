"""Zydah verifier: System 2 escalation loop for speculative fused verdicts.

The validator treats a wide [Bel, Pl] interval and/or high conflict mass as
speculative: the fused answer is not dependable yet. When flagged, System 1
answers the LLM with a structured follow-up (clarifying question, summary,
blocking assumptions) and that JSON itself is the transmitted message: a
model responder answers it with a better response, the answer is appended
as evidence, and fusion runs again until the interval/conflict contract
from the OmegaConf YAML is satisfied or the iteration budget ends.

This is a closed-world verifier: it only reasons over supplied evidence.
Open-world evidence gathering (web fetch) is a later extension point.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from zydah.choices import (
    OTHER_VALUE,
    Choice,
    ChoiceKey,
    HumanChoice,
    HumanChoiceContext,
    HumanFollowup,
)
from zydah.evidence import (
    GRAPH_RESOLUTION,
    MODEL_FOLLOWUP,
    ORIGINAL,
    EvidenceChunk,
    EvidenceLedger,
    classify_independence,
)
from zydah.fusion import Mass
from zydah.interface import FUSED_TAG, direct_verdict, fuse_response, fused_verdict
from zydah.judges import extract_json_object, parse_mass_json
from zydah.transport import chat_complete

ModelResponder = Callable[[str, str], str]
MassElicitor = Callable[[str, FrozenSet[str], str], Mass]


def make_model_responder() -> ModelResponder:
    """Live responder: sends the prompt via zydah.transport.

    The verifier supplies the resolved model ID (cheaper model by
    default); transport reads endpoint/key from the environment.
    """

    def responder(prompt: str, model_id: str) -> str:
        return chat_complete(prompt, model_id=model_id)

    return responder


def make_mass_elicitor(template: str) -> MassElicitor:
    """Direct elicitor: one model call -> one validated mass (no panel).

    Used by the transport+verifier path, which bypasses the judge panel:
    the model scores the evidence itself, and the verifier judges the
    resulting interval.
    """

    def elicitor(evidence: str, frame: FrozenSet[str], model_id: str) -> Mass:
        prompt = template.format(labels=", ".join(sorted(frame)), evidence=evidence)
        return parse_mass_json(extract_json_object(chat_complete(prompt, model_id=model_id)), frame)

    return elicitor


def serialize_followup(followup: Mapping[str, object]) -> str:
    """Serialize a follow-up payload exactly as transmitted to the LLM.

    The JSON block is the message: a responder receives these bytes, so
    artifacts and transcripts that show them show a sent call, not a
    summary of one.
    """
    return json.dumps(followup, indent=2)


def format_needs_human(result: Mapping[str, object]) -> str:
    """Render a paused follow-up loop for the terminal: [human] / [zydah]."""
    if not result.get("needs_human"):
        return ""
    return (
        f"[human]\n{result.get('human_question', '')}\n"
        f"\n"
        f"[zydah]\n"
        f"pending: follow-up paused for human input "
        f"(followups used: {result.get('followups', 0)})\n"
    )


def format_transcript(history: Sequence[Mapping[str, object]]) -> str:
    """Render follow-up cycles as Prompt_1/Response_1 pairs for streaming.

    Only flagged iterations appear: each carries the transmitted follow-up
    JSON and, when a responder answered, its response.
    """
    blocks = []
    n = 0
    for entry in history:
        if "prompt" not in entry:
            continue
        n += 1
        blocks.append(f"[Prompt {n}]\n{entry['prompt']}")
        if entry.get("response"):
            blocks.append(f"[Response {n}]\n{entry['response']}")
    return "\n\n".join(blocks)


def load_verifier_config(overrides: Optional[Sequence[str]] = None) -> DictConfig:
    """Load verifier hyperparameters from ``conf/verifier.yaml`` via Hydra.

    Standing rule: thresholds, tau, iteration budgets, prompt text, and the
    default model live in YAML, not as literals in code.
    """
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="conf"):
        return compose(config_name="verifier", overrides=list(overrides or []))


def _coerce_config(config: Optional[object]) -> DictConfig:
    if config is None:
        return load_verifier_config()
    if isinstance(config, DictConfig):
        return config
    if isinstance(config, Mapping):
        return OmegaConf.create(config)
    raise TypeError(f"config must be a DictConfig or mapping, got {type(config)!r}")


def _require(cfg: DictConfig, key: str) -> object:
    value = OmegaConf.select(cfg, key)
    if value is None:
        raise ValueError(f"verifier config is missing required key {key!r}")
    return value


class ZydahVerifier:
    """Verify a fused verdict; ask System 2 for help while it is speculative.

    Parameters come from OmegaConf YAML by default. ``model_id`` resolution
    order: explicit argument, then the environment variable named by
    ``model.env_var`` in YAML, then ``model.id`` in YAML (the cheaper model).
    """

    def __init__(
        self,
        frame: FrozenSet[str],
        panel: Optional[object] = None,
        responder: Optional[ModelResponder] = None,
        config: Optional[object] = None,
        model_id: Optional[str] = None,
        mass_elicitor: Optional[MassElicitor] = None,
        session_id: Optional[str] = None,
        log_path: Optional[object] = None,
        graph_resolver: Optional[object] = None,
        graph_factory: Optional[Callable[[str], object]] = None,
    ) -> None:
        self.frame = frozenset(frame)
        self.panel = panel
        self.responder = responder
        self.graph_resolver = graph_resolver
        self.graph_factory = graph_factory
        self.cfg = _coerce_config(config)
        self.model_id = self._resolve_model_id(model_id)
        self.session_id = session_id
        self._log_path = Path(log_path) if log_path else None
        self._run_id = uuid.uuid4().hex
        self.mass_elicitor = mass_elicitor or make_mass_elicitor(
            str(_require(self.cfg, "direct_mass_prompt_template"))
        )
        self._question = str(_require(self.cfg, "question"))

    def _log(self, event: Dict) -> None:
        """Append one JSONL cycle record (no secrets; ids/metrics only)."""
        if self._log_path is None:
            return
        record = {"run_id": self._run_id, "ts": time.time(), **event}
        with open(self._log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _resolve_model_id(self, model_id: Optional[str]) -> str:
        if model_id:
            return model_id
        env_var = str(_require(self.cfg, "model.env_var"))
        from_env = os.environ.get(env_var)
        if from_env:
            return from_env
        return str(_require(self.cfg, "model.id"))

    def validate(self, zydah_verdict: Mapping[str, object]) -> Dict:
        """Judge one fused verdict on interval width and conflict mass.

        Returns ``{"verdict": <good value from YAML>}`` when both metrics are
        within their YAML thresholds; otherwise ``{"verdict": False, ...}``
        with the follow-up question the prompt injector will send onward.
        """
        width = float(zydah_verdict.get("width", 0.0))
        conflict = float(zydah_verdict.get("conflict", 0.0))
        width_threshold = float(_require(self.cfg, "width_threshold"))
        conflict_threshold = float(_require(self.cfg, "conflict_threshold"))
        result: Dict = {
            "width": width,
            "conflict": conflict,
            "width_threshold": width_threshold,
            "conflict_threshold": conflict_threshold,
        }
        if zydah_verdict.get("verdict") is None:
            return {"verdict": False, "question": self._question, **result}
        if width <= width_threshold and conflict <= conflict_threshold:
            return {"verdict": str(_require(self.cfg, "good_verdict")), **result}
        return {"verdict": False, "question": self._question, **result}

    def followup_payload(
        self,
        validation: Mapping[str, object],
        assumptions: Optional[Sequence[str]] = None,
    ) -> Dict:
        """System 1's structured response to the LLM when flagged.

        This is the contract the LLM answers with a better response:
        the clarifying ``question``, a ``summary`` of what to fix, and
        the blocking ``assumptions``. Assumptions are caller-annotated
        (e.g. "Springfield has no state qualifier") — System 1 surfaces
        them, it does not invent them. No delivery status is attached:
        whether it was answered is shown by the next turn, not a flag.
        """
        return {
            "question": validation.get("question") or self._question,
            "summary": str(_require(self.cfg, "followup_summary")),
            "assumptions": [str(item) for item in (assumptions or [])],
        }

    def _is_human_question(self, answer: object) -> bool:
        """True when the responder asks the human for missing input.

        Such an answer is a routing signal, not evidence: it must not
        enter the ledger as a follow-up chunk.
        """
        text = str(answer).strip()
        if "?" not in text:
            return False
        markers = _require(self.cfg, "human_question_markers")
        lowered = text.lower()
        return any(str(marker).lower() in lowered for marker in markers)

    def _partition_choices(self, graph: object) -> List[Choice]:
        """Structured choices for a partition question (Other last)."""
        labels = [
            node.label
            for node in graph.nodes_of_kind("partition")  # type: ignore[attr-defined]
            if node.features.get("axis") == "direction"
        ]
        values = (sorted(labels) if labels else ["More detail"]) + [OTHER_VALUE]
        keys = list(ChoiceKey)
        if len(values) > len(keys):
            raise ValueError("too many partition choices for ChoiceKey enum")
        return [Choice(key=keys[i], value=value) for i, value in enumerate(values)]

    def _format_choices(self, choices: List[Choice]) -> str:
        options = "\n".join(
            f"{choice.key.value}. {choice.value}" for choice in choices
        )
        template = str(_require(self.cfg, "human_question_template"))
        return template.format(options=options)

    def _partition_question(self, graph: object) -> str:
        """Multiple-choice human question at the direction partitions.

        Choices are lettered in sorted order with ``Other`` always
        appended last, so a human can answer outside the graph's
        taxonomy instead of being forced into it.
        """
        return self._format_choices(self._partition_choices(graph))

    @staticmethod
    def _candidate_choices(candidates: List[str]) -> List[Choice]:
        """Next-round choices from a remaining candidate subset."""
        keys = list(ChoiceKey)
        values = list(candidates) + [OTHER_VALUE]
        if len(values) > len(keys):
            raise ValueError("too many candidates for ChoiceKey enum")
        return [Choice(key=keys[i], value=value) for i, value in enumerate(values)]

    def _resolve_with_graph(self, evidence_text: str):
        """Run the injected graph resolver, if configured.

        Returns ``(resolution, graph)`` or ``(None, None)``.
        """
        if self.graph_resolver is None or self.graph_factory is None:
            return None, None
        graph = self.graph_factory(evidence_text)
        return self.graph_resolver.resolve(graph), graph

    @staticmethod
    def _resolution_text(resolution: object) -> str:
        top = sorted(
            resolution.candidate_scores.items(),  # type: ignore[attr-defined]
            key=lambda kv: kv[1],
            reverse=True,
        )[:3]
        summary = ", ".join(f"{cid}={score:.4f}" for cid, score in top)
        return (
            f"Graph resolution ({resolution.status}): "  # type: ignore[attr-defined]
            f"best_partition={resolution.best_partition}; top candidates: {summary}"  # type: ignore[attr-defined]
        )

    def verify(
        self, evidence: str, run: int = 0, assumptions: Optional[Sequence[str]] = None
    ) -> Dict:
        """Fuse, validate, and iterate with System 2 while speculative.

        Each iteration appends the model answer to the evidence and re-runs
        fusion. Without a responder, the first flagged prompt is returned so
        a caller can send it to a model itself.
        """
        world = str(_require(self.cfg, "world"))
        if world != "closed":
            raise NotImplementedError(
                "open-world verification is not implemented; web fetch is a follow-up"
            )
        max_iterations = int(_require(self.cfg, "max_iterations"))
        current_evidence = evidence
        history = []
        followups = 0
        while True:
            result = fuse_response(
                current_evidence,
                self.frame,
                panel=self.panel,
                tau=float(_require(self.cfg, "tau")),
                rule=str(_require(self.cfg, "rule")),
                run=run + followups,
            )
            zydah_verdict = result[FUSED_TAG]
            validation = self.validate(zydah_verdict)
            history.append({"validation": validation, "zydah": zydah_verdict})
            if validation["verdict"] is not False:
                return {
                    "verdict": validation["verdict"],
                    "result": result,
                    "history": history,
                    "followups": followups,
                    "model_id": self.model_id,
                }
            followup = self.followup_payload(validation, assumptions)
            prompt = serialize_followup(followup)  # transmitted to the LLM
            history[-1]["prompt"] = prompt
            history[-1]["followup"] = followup
            if self.responder is None or followups >= max_iterations:
                return {
                    "verdict": False,
                    "question": validation["question"],
                    "followup": followup,
                    "prompt": prompt,
                    "result": result,
                    "history": history,
                    "followups": followups,
                    "model_id": self.model_id,
                }
            answer = self.responder(prompt, self.model_id)
            if not answer or not str(answer).strip():
                return {
                    "verdict": False,
                    "question": validation["question"],
                    "followup": followup,
                    "prompt": prompt,
                    "result": result,
                    "history": history,
                    "followups": followups,
                    "model_id": self.model_id,
                }
            history[-1]["response"] = answer
            if self._is_human_question(answer):
                history[-1]["route"] = "human"
                return {
                    "verdict": False,
                    "needs_human": True,
                    "human_question": str(answer),
                    "question": validation["question"],
                    "followup": followup,
                    "prompt": prompt,
                    "result": result,
                    "history": history,
                    "followups": followups,
                    "model_id": self.model_id,
                }
            resolution, graph = self._resolve_with_graph(current_evidence)
            if resolution is not None and resolution.status == "unresolved":
                history[-1]["route"] = "agent_unresolved"
                return {
                    "verdict": False,
                    "needs_human": True,
                    "human_question": self._partition_question(graph),
                    "choices": [c.model_dump(mode="json") for c in self._partition_choices(graph)],
                    "evidence": current_evidence,
                    "original_question": current_evidence,
                    "question": validation["question"],
                    "followup": followup,
                    "prompt": prompt,
                    "result": result,
                    "history": history,
                    "followups": followups,
                    "model_id": self.model_id,
                }
            prefix = str(_require(self.cfg, "additional_info_prefix"))
            current_evidence = f"{current_evidence}{prefix}{answer}"
            if resolution is not None:
                history[-1]["route"] = "agent"
                current_evidence = (
                    f"{current_evidence}{prefix}{self._resolution_text(resolution)}"
                )
            followups += 1

    def _independence(self, first_session_id, followups: int) -> dict:
        return classify_independence(first_session_id, self.session_id, followups)

    def _resolve_human_choice(
        self,
        previous: Mapping[str, object],
        choice_key: object,
        other_text: Optional[str] = None,
    ) -> HumanFollowup:
        """Resolve a paused result's choice key to a HumanFollowup.

        Shared by both resume paths. ``Other`` takes free text;
        anything unoffered or textless fails closed with ValueError.
        """
        if not previous.get("needs_human"):
            raise ValueError("previous result is not paused for human input")
        raw_choices = previous.get("choices")
        if not raw_choices:
            raise ValueError("previous result carries no structured choices")
        choices = [Choice.model_validate(item) for item in raw_choices]  # type: ignore[union-attr]
        key = ChoiceKey(str(choice_key).strip().upper())
        selected = next((c for c in choices if c.key == key), None)
        if selected is None:
            raise ValueError(f"choice {key.value!r} was not offered")
        value = selected.value
        if value == OTHER_VALUE:
            if not other_text or not str(other_text).strip():
                raise ValueError("the Other choice requires free text")
            value = str(other_text).strip()
        return HumanFollowup(human_choice=HumanChoice(key=key, value=value))

    def _condition_resolution(
        self,
        resolution: object,
        support_ids,
        alpha_key: str = "human_choice_alpha",
    ) -> object:
        """Condition a graph resolution on chosen support, if supported.

        Uses the resolver's Yager conditioning with the reliability
        named by ``alpha_key`` in its graph config (human choices are
        trusted more than model-supplied relationships); resolvers
        without a ``condition`` method return their input unchanged.
        """
        condition = getattr(self.graph_resolver, "condition", None)
        if not callable(condition):
            return resolution
        cfg = getattr(self.graph_resolver, "cfg", None)
        alpha = OmegaConf.select(cfg, f"resolver.{alpha_key}") if cfg is not None else None
        return condition(
            resolution,
            frozenset(support_ids),
            alpha=float(alpha) if alpha is not None else 1.0,
        )

    def _choice_context(
        self, previous: Mapping[str, object], partition_value: str
    ) -> HumanChoiceContext:
        """Expand a chosen partition against the candidate graph.

        Returns the partition's member candidate labels, their
        re-scored shares within the subset (resolver scores
        renormalized over members), and the original question — the
        context a stateless model call needs to act on the choice.
        """
        original = str(previous.get("original_question") or "")
        context = HumanChoiceContext(
            partition=partition_value, question=original or None
        )
        if self.graph_resolver is None or self.graph_factory is None:
            return context
        graph = self.graph_factory(original)
        partition_node = next(
            (
                node
                for node in graph.nodes_of_kind("partition")  # type: ignore[attr-defined]
                if node.label == partition_value
                and node.features.get("axis") == "direction"
            ),
            None,
        )
        if partition_node is None:
            # Second-round choice: the value is a single candidate.
            candidate_node = next(
                (
                    node
                    for node in graph.nodes_of_kind("candidate")  # type: ignore[attr-defined]
                    if node.label == partition_value
                ),
                None,
            )
            if candidate_node is not None:
                context.candidates = [candidate_node.label]
                resolution = self.graph_resolver.resolve(graph)  # type: ignore[attr-defined]
                conditioned = self._condition_resolution(
                    resolution, {candidate_node.id}
                )
                context.candidate_scores = {
                    candidate_node.label: float(
                        conditioned.candidate_scores[candidate_node.id]
                    )
                }
                context.candidate_intervals = {
                    candidate_node.label: list(
                        conditioned.candidate_intervals[candidate_node.id]
                    )
                }
            return context
        member_ids = [
            edge.source
            for edge in graph.edges_of_kind("member_of")  # type: ignore[attr-defined]
            if edge.target == partition_node.id
        ]
        labels = {node.id: node.label for node in graph.nodes}  # type: ignore[attr-defined]
        context.candidates = [labels[cid] for cid in member_ids if cid in labels]
        resolution = self.graph_resolver.resolve(graph)  # type: ignore[attr-defined]
        conditioned = self._condition_resolution(resolution, set(member_ids))
        context.candidate_scores = {
            labels[cid]: float(conditioned.candidate_scores.get(cid, 0.0))
            for cid in member_ids
            if cid in labels
        }
        context.candidate_intervals = {
            labels[cid]: list(conditioned.candidate_intervals.get(cid, (0.0, 1.0)))
            for cid in member_ids
            if cid in labels
        }
        return context

    def _other_context(
        self, previous: Mapping[str, object], clue_text: str = ""
    ) -> HumanChoiceContext:
        """Directional frame for a free-text (Other) human answer.

        The offered choices tell the model what the clue must be
        mapped onto: a direction (Celtics -> East / Massachusetts)
        or one of the remaining candidates. Without this frame an
        Other answer arrives unanchored and cannot be resolved.
        """
        original = str(previous.get("original_question") or "")
        offered = [
            str(item.get("value"))
            for item in (previous.get("choices") or [])  # type: ignore[union-attr]
            if isinstance(item, Mapping) and item.get("value") != OTHER_VALUE
        ]
        context = HumanChoiceContext(
            partition=OTHER_VALUE,
            offered_choices=offered + [OTHER_VALUE],
            question=original or None,
        )
        candidate_like = [value for value in offered if ", " in value]
        if candidate_like:
            context.candidates = candidate_like
            share = 1.0 / len(candidate_like)
            context.candidate_scores = {
                label: share for label in candidate_like
            }
            context.candidate_intervals = {
                label: [0.0, 1.0] for label in candidate_like
            }
            self._add_relationship_questions(context, clue_text)
            return context
        if self.graph_resolver is not None and self.graph_factory is not None:
            graph = self.graph_factory(original)
            labels = {
                node.id: node.label for node in graph.nodes  # type: ignore[attr-defined]
            }
            context.candidates = [
                node.label for node in graph.nodes_of_kind("candidate")  # type: ignore[attr-defined]
            ]
            resolution = self.graph_resolver.resolve(graph)  # type: ignore[attr-defined]
            context.candidate_scores = {
                labels[cid]: float(score)
                for cid, score in resolution.candidate_scores.items()
                if cid in labels
            }
            context.candidate_intervals = {
                labels[cid]: list(interval)
                for cid, interval in resolution.candidate_intervals.items()
                if cid in labels
            }
        self._add_relationship_questions(context, clue_text)
        return context

    def _add_relationship_questions(
        self, context: HumanChoiceContext, clue_text: str
    ) -> None:
        """Ask how the clue relates to each candidate (attribution)."""
        if not clue_text or not context.candidates:
            return
        template = str(_require(self.cfg, "relationship_question_template"))
        context.relationship_questions = {
            label: template.format(clue=clue_text, candidate=label)
            for label in context.candidates
        }

    def _relationship_conditioning(
        self, answer: object, followup: HumanFollowup
    ) -> Optional[Dict]:
        """Condition the graph on the model's relationship answer.

        The LLM supplies the world knowledge (``Celtics -> Boston,
        Massachusetts``); System 1 adds no entity table of its own. It
        only matches candidate state names/labels mentioned in the
        model's answer and Yager-conditions the credal state on that
        support, so attribution weights move to the related candidate
        instead of staying uniform. Negative answers condition nothing.
        """
        if (
            followup.context is None
            or not followup.context.relationship_questions
        ):
            return None
        if self.graph_resolver is None or self.graph_factory is None:
            return None
        text = str(answer)
        negatives = [
            str(marker).lower()
            for marker in (
                OmegaConf.select(
                    self.cfg, "relationship_negative_markers"
                )
                or []
            )
        ]
        affirmatives = [
            str(marker).lower()
            for marker in (
                OmegaConf.select(
                    self.cfg, "relationship_affirmative_markers"
                )
                or []
            )
        ]

        def mentions(node: object, haystack: str) -> bool:
            lowered = haystack.lower()
            state_name = str(node.features.get("state_name", "")).lower()  # type: ignore[attr-defined]
            state = str(node.features.get("state", ""))  # type: ignore[attr-defined]
            return bool(
                (state_name and state_name in lowered)
                or node.label.lower() in lowered  # type: ignore[attr-defined]
                or (
                    state
                    and re.search(rf"\b{re.escape(state)}\b", haystack)
                    is not None
                )
            )

        def _normalized(haystack: str) -> str:
            # "city / state" and "city/state" are the same phrase.
            return re.sub(r"\s*/\s*", "/", haystack.lower())

        def is_negative(haystack: str) -> bool:
            lowered = _normalized(haystack)
            return any(
                _normalized(marker) in lowered for marker in negatives
            )

        def is_affirmative(haystack: str) -> bool:
            lowered = _normalized(haystack)
            return any(
                _normalized(marker) in lowered for marker in affirmatives
            )

        original = ""
        if followup.context is not None:
            original = followup.context.question or ""
        graph = self.graph_factory(original)
        nodes = list(graph.nodes_of_kind("candidate"))  # type: ignore[attr-defined]
        labels: Dict[str, str] = {}
        support: List[str] = []
        # Per-candidate sections (the live answer shape): each
        # candidate's own section decides its support, so negated
        # mentions of other states never leak into this candidate.
        headings = []
        for node in nodes:
            # A section heading is the label followed by a separator;
            # a bare mention in an enumeration is not a section.
            match = re.search(
                re.escape(node.label) + r"\s*[:\-—?]",
                text,
                re.IGNORECASE,
            )
            if match is None:
                match = re.search(
                    re.escape(node.label), text, re.IGNORECASE
                )
            if match is not None:
                headings.append((match.start(), match.end(), node))
        headings.sort(key=lambda item: item[0])
        if len({node.id for _, _, node in headings}) >= 2:
            for index, (_, end, node) in enumerate(headings):
                seg_end = (
                    headings[index + 1][0]
                    if index + 1 < len(headings)
                    else len(text)
                )
                body = text[end:seg_end]
                if is_negative(body) or not (
                    is_affirmative(body) or mentions(node, body)
                ):
                    continue
                support.append(node.id)
                labels[node.id] = node.label
        else:
            # Free-form answer: judge each clause separately.
            clauses = [
                clause
                for clause in re.split(r"[.!?;\n]+", text)
                if clause.strip()
            ]
            # A clause naming several candidates is an enumeration of
            # the set, not evidence for any one of them.
            evidence_clauses = [
                clause
                for clause in clauses
                if sum(mentions(node, clause) for node in nodes) <= 1
            ]
            for node in nodes:
                state_name = str(
                    node.features.get("state_name", "")
                ).lower()

                def spelled_mention(clause: str) -> bool:
                    lowered = clause.lower()
                    return bool(
                        (state_name and state_name in lowered)
                        or node.label.lower() in lowered
                    )

                if any(
                    mentions(node, clause)
                    and not is_negative(clause)
                    and (is_affirmative(clause) or spelled_mention(clause))
                    for clause in evidence_clauses
                ):
                    support.append(node.id)
                    labels[node.id] = node.label
        if not support:
            return None
        all_labels = {
            node.id: node.label
            for node in graph.nodes_of_kind("candidate")  # type: ignore[attr-defined]
        }
        resolution = self.graph_resolver.resolve(graph)  # type: ignore[attr-defined]
        conditioned = self._condition_resolution(
            resolution, set(support), alpha_key="conditioning_alpha"
        )
        return {
            "support": [labels[cid] for cid in support],
            "candidate_scores": {
                all_labels[cid]: float(
                    conditioned.candidate_scores.get(cid, 0.0)
                )
                for cid in all_labels
            },
            "candidate_intervals": {
                all_labels[cid]: list(
                    conditioned.candidate_intervals.get(cid, (0.0, 1.0))
                )
                for cid in all_labels
            },
            "text": (
                "Conditioned on model relationship answer: "
                f"{self._resolution_text(conditioned)}"
            ),
        }

    def resume_direct(
        self,
        previous: Mapping[str, object],
        choice_key: object,
        other_text: Optional[str] = None,
    ) -> Dict:
        """Resume a paused direct loop from the human's choice.

        The key (e.g. ``B``) is resolved through the Pydantic
        ``ChoiceKey`` enum to its value (``North``; ``Other`` takes
        free text). That value is sent to the model as a
        ``HumanFollowup`` JSON — a human-feedback signal — and the
        model's revised response is appended as evidence and
        re-elicited. The selection is recorded on the paused history
        entry as ``human_feedback`` for later analysis/training.
        """
        followup = self._resolve_human_choice(previous, choice_key, other_text)
        if self.responder is None:
            raise RuntimeError("resuming a human choice requires a responder")
        offered = {
            str(item.get("key")): str(item.get("value"))
            for item in (previous.get("choices") or [])  # type: ignore[union-attr]
            if isinstance(item, Mapping)
        }
        if offered.get(followup.human_choice.key.value) != OTHER_VALUE:
            followup = HumanFollowup(
                human_choice=followup.human_choice,
                context=self._choice_context(
                    previous, followup.human_choice.value
                ),
            )
        else:
            followup = HumanFollowup(
                human_choice=followup.human_choice,
                context=self._other_context(
                    previous, followup.human_choice.value
                ),
            )
        prompt = followup.model_dump_json(indent=2)
        ledger = previous["ledger"]  # type: ignore[assignment]
        history = previous["history"]  # type: ignore[assignment]
        followups = int(previous.get("followups", 0))  # type: ignore[arg-type]
        first_session = next(
            (
                chunk.session_id
                for chunk in ledger.chunks  # type: ignore[attr-defined]
                if chunk.kind == ORIGINAL
            ),
            None,
        )
        history[-1]["human_feedback"] = followup.model_dump(mode="json")
        history[-1]["human_followup_prompt"] = prompt
        self._log(
            {
                "event": "human_feedback",
                "choice": followup.model_dump(mode="json"),
                "verifier_session": self.session_id,
            }
        )
        answer = self.responder(prompt, self.model_id)
        if not answer or not str(answer).strip():
            return {**dict(previous), "verdict": False, "human_followup": followup.model_dump(mode="json")}
        history[-1]["response"] = answer
        # Relationship evidence is extracted before routing: an answer
        # can both identify the support and end with a question, and
        # the identified support conditions the state either way.
        conditioning = self._relationship_conditioning(answer, followup)
        if conditioning is None and self._is_human_question(answer):
            result = {
                **dict(previous),
                "verdict": False,
                "needs_human": True,
                "human_question": str(answer),
                "human_followup": followup.model_dump(mode="json"),
                "history": history,
                "followups": followups,
            }
            if followup.context and len(followup.context.candidates) > 1:
                # Continue the loop: the remaining candidate subset is
                # the next state; offer it as the next choice round.
                next_choices = self._candidate_choices(
                    followup.context.candidates
                )
                result["choices"] = [
                    choice.model_dump(mode="json") for choice in next_choices
                ]
                result["human_question"] = (
                    f"{answer}\n\n{self._format_choices(next_choices)}"
                )
            return result
        conditioning_extra: Dict = {}
        if conditioning is not None:
            history[-1]["graph_conditioning"] = {
                key: value
                for key, value in conditioning.items()
                if key != "text"
            }
            conditioning_extra = {
                "graph_conditioning": history[-1]["graph_conditioning"]
            }
        prefix = str(_require(self.cfg, "additional_info_prefix"))
        ledger.add(  # type: ignore[attr-defined]
            EvidenceChunk(
                MODEL_FOLLOWUP,
                str(answer),
                session_id=self.session_id,
                model_id=self.model_id,
                playground="B",
                turn=followups + 1,
            )
        )
        if conditioning is not None:
            ledger.add(  # type: ignore[attr-defined]
                EvidenceChunk(
                    GRAPH_RESOLUTION,
                    conditioning["text"],
                    session_id=self.session_id,
                    model_id=self.model_id,
                    playground="B",
                    turn=followups + 1,
                )
            )
        followups += 1
        max_iterations = int(_require(self.cfg, "max_iterations"))
        while True:
            view_text = ledger.assisted(separator=prefix)  # type: ignore[attr-defined]
            mass = self.mass_elicitor(view_text, self.frame, self.model_id)
            zydah_verdict = fused_verdict(
                [mass],
                self.frame,
                tau=float(_require(self.cfg, "tau")),
                rule=str(_require(self.cfg, "rule")),
            )
            result = {
                "muse": direct_verdict([mass], self.frame),
                "zydah": zydah_verdict,
            }
            validation = self.validate(zydah_verdict)
            history.append(
                {"validation": validation, "zydah": zydah_verdict, "mass": mass}
            )
            independence = self._independence(first_session, followups)
            if validation["verdict"] is not False:
                return {
                    **dict(previous),
                    **conditioning_extra,
                    "verdict": validation["verdict"],
                    "needs_human": False,
                    "result": result,
                    "history": history,
                    "followups": followups,
                    "independence": independence,
                    "ledger": ledger,
                    "human_followup": followup.model_dump(mode="json"),
                }
            next_followup = self.followup_payload(validation)
            next_prompt = serialize_followup(next_followup)
            history[-1]["prompt"] = next_prompt
            history[-1]["followup"] = next_followup
            base = {
                **dict(previous),
                **conditioning_extra,
                "verdict": False,
                "question": validation["question"],
                "followup": next_followup,
                "prompt": next_prompt,
                "result": result,
                "history": history,
                "followups": followups,
                "independence": independence,
                "ledger": ledger,
                "human_followup": followup.model_dump(mode="json"),
            }
            if followups >= max_iterations:
                return base
            next_answer = self.responder(next_prompt, self.model_id)
            if not next_answer or not str(next_answer).strip():
                return base
            history[-1]["response"] = next_answer
            if self._is_human_question(next_answer):
                return {
                    **base,
                    "needs_human": True,
                    "human_question": str(next_answer),
                }
            ledger.add(  # type: ignore[attr-defined]
                EvidenceChunk(
                    MODEL_FOLLOWUP,
                    str(next_answer),
                    session_id=self.session_id,
                    model_id=self.model_id,
                    playground="B",
                    turn=followups + 1,
                )
            )
            followups += 1

    def verify_direct(
        self,
        evidence: str,
        first_response=None,
        assumptions: Optional[Sequence[str]] = None,
    ) -> Dict:
        """Transport + verifier only: bypass the judge panel.

        The model scores the evidence itself (one mass per call via
        ``mass_elicitor``); the validator judges that mass's interval.
        When flagged, System 1 returns a structured ``followup`` for the
        LLM to answer with a better response (see ``followup_payload``);
        with a responder wired, that answer is appended as evidence and
        re-judged. A single source has no inter-judge conflict, so
        conflict stays 0.0 and width carries the speculation signal.
        """
        world = str(_require(self.cfg, "world"))
        if world != "closed":
            raise NotImplementedError(
                "open-world verification is not implemented; web fetch is a follow-up"
            )
        max_iterations = int(_require(self.cfg, "max_iterations"))
        prefix = str(_require(self.cfg, "additional_info_prefix"))
        first_session = getattr(first_response, "session_id", None)
        first_playground = getattr(first_response, "playground", None)
        ledger = EvidenceLedger()
        ledger.add(
            EvidenceChunk(ORIGINAL, evidence, session_id=first_session, playground=first_playground)
        )
        history = []
        followups = 0
        while True:
            view_text = (
                ledger.original_only() if followups == 0 else ledger.assisted(separator=prefix)
            )
            mass = self.mass_elicitor(view_text, self.frame, self.model_id)
            zydah_verdict = fused_verdict(
                [mass],
                self.frame,
                tau=float(_require(self.cfg, "tau")),
                rule=str(_require(self.cfg, "rule")),
            )
            result = {
                "muse": direct_verdict([mass], self.frame),
                "zydah": zydah_verdict,
            }
            validation = self.validate(zydah_verdict)
            history.append({"validation": validation, "zydah": zydah_verdict, "mass": mass})
            if validation["verdict"] is not False:
                independence = self._independence(first_session, followups)
                self._log({"event": "verify_direct", "verdict": validation["verdict"], "followups": followups, "independence": independence, "sessions": ledger.session_ids(), "verifier_session": self.session_id, "width": zydah_verdict.get("width"), "conflict": zydah_verdict.get("conflict")})
                return {
                    "verdict": validation["verdict"],
                    "result": result,
                    "history": history,
                    "followups": followups,
                    "model_id": self.model_id,
                    "mode": "direct",
                    "independence": independence,
                    "ledger": ledger,
                }
            followup = self.followup_payload(validation, assumptions)
            prompt = serialize_followup(followup)  # transmitted to the LLM
            history[-1]["prompt"] = prompt
            history[-1]["followup"] = followup
            if self.responder is None or followups >= max_iterations:
                independence = self._independence(first_session, followups)
                self._log({"event": "verify_direct", "verdict": False, "followups": followups, "independence": independence, "sessions": ledger.session_ids(), "verifier_session": self.session_id, "width": zydah_verdict.get("width"), "conflict": zydah_verdict.get("conflict")})
                return {
                    "verdict": False,
                    "question": validation["question"],
                    "followup": followup,
                    "prompt": prompt,
                    "result": result,
                    "history": history,
                    "followups": followups,
                    "model_id": self.model_id,
                    "mode": "direct",
                    "independence": independence,
                    "ledger": ledger,
                }
            answer = self.responder(prompt, self.model_id)
            if not answer or not str(answer).strip():
                independence = self._independence(first_session, followups)
                self._log({"event": "verify_direct", "verdict": False, "followups": followups, "independence": independence, "sessions": ledger.session_ids(), "verifier_session": self.session_id, "width": zydah_verdict.get("width"), "conflict": zydah_verdict.get("conflict")})
                return {
                    "verdict": False,
                    "question": validation["question"],
                    "followup": followup,
                    "prompt": prompt,
                    "result": result,
                    "history": history,
                    "followups": followups,
                    "model_id": self.model_id,
                    "mode": "direct",
                    "independence": independence,
                    "ledger": ledger,
                }
            history[-1]["response"] = answer
            if self._is_human_question(answer):
                history[-1]["route"] = "human"
                independence = self._independence(first_session, followups)
                self._log({"event": "verify_direct", "verdict": False, "needs_human": True, "followups": followups, "independence": independence, "sessions": ledger.session_ids(), "verifier_session": self.session_id, "width": zydah_verdict.get("width"), "conflict": zydah_verdict.get("conflict")})
                return {
                    "verdict": False,
                    "needs_human": True,
                    "human_question": str(answer),
                    "question": validation["question"],
                    "followup": followup,
                    "prompt": prompt,
                    "result": result,
                    "history": history,
                    "followups": followups,
                    "model_id": self.model_id,
                    "mode": "direct",
                    "independence": independence,
                    "ledger": ledger,
                }
            resolution, graph = self._resolve_with_graph(view_text)
            if resolution is not None and resolution.status == "unresolved":
                history[-1]["route"] = "agent_unresolved"
                independence = self._independence(first_session, followups)
                self._log({"event": "verify_direct", "verdict": False, "needs_human": True, "followups": followups, "independence": independence, "sessions": ledger.session_ids(), "verifier_session": self.session_id, "width": zydah_verdict.get("width"), "conflict": zydah_verdict.get("conflict")})
                return {
                    "verdict": False,
                    "needs_human": True,
                    "human_question": self._partition_question(graph),
                    "choices": [c.model_dump(mode="json") for c in self._partition_choices(graph)],
                    "question": validation["question"],
                    "original_question": evidence,
                    "followup": followup,
                    "prompt": prompt,
                    "result": result,
                    "history": history,
                    "followups": followups,
                    "model_id": self.model_id,
                    "mode": "direct",
                    "independence": independence,
                    "ledger": ledger,
                }
            ledger.add(
                EvidenceChunk(
                    MODEL_FOLLOWUP,
                    str(answer),
                    session_id=self.session_id,
                    model_id=self.model_id,
                    playground="B",
                    turn=followups + 1,
                )
            )
            if resolution is not None:
                history[-1]["route"] = "agent"
                ledger.add(
                    EvidenceChunk(
                        GRAPH_RESOLUTION,
                        self._resolution_text(resolution),
                        session_id=self.session_id,
                        model_id=self.model_id,
                        playground="B",
                        turn=followups + 1,
                    )
                )
            followups += 1

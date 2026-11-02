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
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, FrozenSet, Mapping, Optional, Sequence

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from zydah.evidence import (
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
    ) -> None:
        self.frame = frozenset(frame)
        self.panel = panel
        self.responder = responder
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
            prefix = str(_require(self.cfg, "additional_info_prefix"))
            current_evidence = f"{current_evidence}{prefix}{answer}"
            followups += 1

    def _independence(self, first_session_id, followups: int) -> dict:
        return classify_independence(first_session_id, self.session_id, followups)

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
            followups += 1

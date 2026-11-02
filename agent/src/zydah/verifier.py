"""Zydah verifier: System 2 escalation loop for speculative fused verdicts.

The validator treats a wide [Bel, Pl] interval and/or high conflict mass as
speculative: the fused answer is not dependable yet. When flagged, the
prompt injector builds a System 2 prompt asking what additional information
would most reduce uncertainty. A model responder answers, the answer is
appended as evidence, and fusion runs again until the interval/conflict
contract from the OmegaConf YAML is satisfied or the iteration budget ends.

This is a closed-world verifier: it only reasons over supplied evidence.
Open-world evidence gathering (web fetch) is a later extension point.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, FrozenSet, Mapping, Optional, Sequence

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

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


def format_transcript(history: Sequence[Mapping[str, object]]) -> str:
    """Render follow-up cycles as Prompt_1/Response_1 pairs for streaming.

    Only flagged iterations appear: each carries the System 2 prompt and,
    when a responder answered, its response.
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


class PromptInjector:
    """Builds the System 2 prompt when the validator flags uncertainty."""

    def __init__(self, template: str, question: str, metric_format: str) -> None:
        self.template = template
        self.question = question
        self.metric_format = metric_format

    def build(self, *, evidence: str, zydah_verdict: Mapping[str, object]) -> str:
        fmt = self.metric_format
        return self.template.format(
            evidence=evidence,
            question=self.question,
            verdict=zydah_verdict.get("verdict"),
            decision=zydah_verdict.get("decision"),
            bel=format(float(zydah_verdict.get("bel", 0.0)), fmt),
            pl=format(float(zydah_verdict.get("pl", 0.0)), fmt),
            width=format(float(zydah_verdict.get("width", 0.0)), fmt),
            conflict=format(float(zydah_verdict.get("conflict", 0.0)), fmt),
        )


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
    ) -> None:
        self.frame = frozenset(frame)
        self.panel = panel
        self.responder = responder
        self.cfg = _coerce_config(config)
        self.model_id = self._resolve_model_id(model_id)
        self.mass_elicitor = mass_elicitor or make_mass_elicitor(
            str(_require(self.cfg, "direct_mass_prompt_template"))
        )
        self.prompt_injector = PromptInjector(
            template=str(_require(self.cfg, "prompt_template")),
            question=str(_require(self.cfg, "question")),
            metric_format=str(_require(self.cfg, "metric_format")),
        )

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
            return {"verdict": False, "question": self.prompt_injector.question, **result}
        if width <= width_threshold and conflict <= conflict_threshold:
            return {"verdict": str(_require(self.cfg, "good_verdict")), **result}
        return {"verdict": False, "question": self.prompt_injector.question, **result}

    def verify(self, evidence: str, run: int = 0) -> Dict:
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
            prompt = self.prompt_injector.build(
                evidence=current_evidence, zydah_verdict=zydah_verdict
            )
            history[-1]["prompt"] = prompt
            if self.responder is None or followups >= max_iterations:
                return {
                    "verdict": False,
                    "question": validation["question"],
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

    def verify_direct(self, evidence: str) -> Dict:
        """Transport + verifier only: bypass the judge panel.

        The model scores the evidence itself (one mass per call via
        ``mass_elicitor``); the validator judges that mass's interval.
        Follow-ups use the same prompt-injector loop as ``verify``.
        A single source has no inter-judge conflict, so conflict stays
        0.0 and width carries the speculation signal.
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
            mass = self.mass_elicitor(current_evidence, self.frame, self.model_id)
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
                return {
                    "verdict": validation["verdict"],
                    "result": result,
                    "history": history,
                    "followups": followups,
                    "model_id": self.model_id,
                    "mode": "direct",
                }
            prompt = self.prompt_injector.build(
                evidence=current_evidence, zydah_verdict=zydah_verdict
            )
            history[-1]["prompt"] = prompt
            if self.responder is None or followups >= max_iterations:
                return {
                    "verdict": False,
                    "question": validation["question"],
                    "prompt": prompt,
                    "result": result,
                    "history": history,
                    "followups": followups,
                    "model_id": self.model_id,
                    "mode": "direct",
                }
            answer = self.responder(prompt, self.model_id)
            if not answer or not str(answer).strip():
                return {
                    "verdict": False,
                    "question": validation["question"],
                    "prompt": prompt,
                    "result": result,
                    "history": history,
                    "followups": followups,
                    "model_id": self.model_id,
                    "mode": "direct",
                }
            history[-1]["response"] = answer
            prefix = str(_require(self.cfg, "additional_info_prefix"))
            current_evidence = f"{current_evidence}{prefix}{answer}"
            followups += 1

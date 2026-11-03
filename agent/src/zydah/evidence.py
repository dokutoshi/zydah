"""Evidence provenance: the anti-eavesdropping boundary.

Every piece of evidence is a chunk with provenance — kind, model,
session, playground, turn. Filtered views decide what an elicitation
prompt may see:

- ``original_only()``: only caller-supplied original evidence. This is
  the independent view; follow-up answers and validator output are
  invisible here.
- ``assisted()``: everything, follow-ups included, explicitly labeled.

Playground A generates first responses (no Zydah judges); Playground B
verifies sealed outputs. A chunk's playground records which side made
it, so joint mass is detectable instead of silent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

ORIGINAL = "original"
MODEL_FOLLOWUP = "model_followup"
GRAPH_RESOLUTION = "graph_resolution"
EXTERNAL = "external"

PLAYGROUND_A = "A"  # generation, no Zydah judges
PLAYGROUND_B = "B"  # verification, Zydah judges

MODE_CONTEXT_INDEPENDENT = "CONTEXT_INDEPENDENT"
MODE_ASSISTED = "ASSISTED"
MODE_JOINT_CONTEXT = "JOINT_CONTEXT"


@dataclass(frozen=True)
class EvidenceChunk:
    kind: str
    text: str
    session_id: Optional[str] = None
    model_id: Optional[str] = None
    playground: Optional[str] = None
    turn: int = 0


@dataclass
class EvidenceLedger:
    chunks: List[EvidenceChunk] = field(default_factory=list)

    def add(self, chunk: EvidenceChunk) -> EvidenceChunk:
        self.chunks.append(chunk)
        return chunk

    def of_kind(self, kind: str) -> List[EvidenceChunk]:
        return [c for c in self.chunks if c.kind == kind]

    def original_only(self, separator: str = "\n") -> str:
        return separator.join(c.text for c in self.of_kind(ORIGINAL))

    def assisted(self, separator: str = "\n") -> str:
        return separator.join(c.text for c in self.chunks)

    def session_ids(self) -> List[str]:
        seen: List[str] = []
        for chunk in self.chunks:
            if chunk.session_id and chunk.session_id not in seen:
                seen.append(chunk.session_id)
        return seen


def classify_independence(
    first_session_id: Optional[str],
    verifier_session_id: Optional[str],
    followups: int,
) -> dict:
    """Apply the anti-eavesdropping session rule.

    Same session as verification => JOINT_CONTEXT (hard rule, no
    exceptions). Any follow-up use => ASSISTED. Only a separately
    sessioned first response with zero follow-ups is contextually
    independent; without a proven session it stays ASSISTED.
    """
    if (
        first_session_id is not None
        and verifier_session_id is not None
        and first_session_id == verifier_session_id
    ):
        return {"mode": MODE_JOINT_CONTEXT, "reasons": ["same session as verifier"]}
    if followups > 0:
        return {"mode": MODE_ASSISTED, "reasons": ["follow-up answers used"]}
    if first_session_id is None or verifier_session_id is None:
        return {"mode": MODE_ASSISTED, "reasons": ["unproven session isolation"]}
    return {"mode": MODE_CONTEXT_INDEPENDENT, "reasons": ["separate sessions, original-only"]}

"""Structured human-choice follow-ups (Pydantic + Enum).

When the follow-up loop pauses for a human (`needs_human`), the
question is offered as lettered choices. The human's keypress is a
feedback signal: System 1 resolves the key to its value, sends that
value to the model as the next follow-up JSON, and the model's
revised response is appended and re-elicited.

Pydantic models own the wire shapes so invalid keys/values fail at
construction, not inside the verifier loop.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ChoiceKey(str, Enum):
    """Letter keys for multiple-choice human questions."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    H = "H"
    I = "I"
    J = "J"
    K = "K"
    L = "L"
    M = "M"
    N = "N"
    O = "O"
    P = "P"
    Q = "Q"
    R = "R"
    S = "S"
    T = "T"
    U = "U"
    V = "V"
    W = "W"
    X = "X"
    Y = "Y"
    Z = "Z"


class HumanRoute(str, Enum):
    """How a responder answer was routed by the verifier."""

    HUMAN = "human"
    AGENT = "agent"
    AGENT_UNRESOLVED = "agent_unresolved"


class Choice(BaseModel):
    """One selectable option in a human question."""

    key: ChoiceKey
    value: str = Field(min_length=1)


class HumanChoice(BaseModel):
    """The human's resolved selection (key plus mapped value)."""

    key: ChoiceKey
    value: str = Field(min_length=1)


class HumanChoiceContext(BaseModel):
    """Graph context attached to a human choice.

    The selected partition is expanded against the candidate graph:
    its member candidates (labels), their re-scored shares within the
    subset, and the original question under review — everything a
    stateless model call needs to act on the choice.
    """

    partition: str
    candidates: List[str] = Field(default_factory=list)
    # Per candidate the (A, ~A, Omega) tuple rendered as labeled keys:
    # mass committed to A, mass committed against A, uncommitted mass.
    candidate_beliefs: Dict[str, Dict[str, float]] = Field(
        default_factory=dict
    )
    offered_choices: List[str] = Field(default_factory=list)
    relationship_questions: Dict[str, str] = Field(default_factory=dict)
    question: Optional[str] = None


class HumanFollowup(BaseModel):
    """Follow-up JSON sent to the model carrying the human's choice.

    This is the transmitted message after a human answers a
    multiple-choice question: ``{"human_choice": {"key": "B",
    "value": "North"}, "context": {...}}``. For the ``Other`` choice
    the value is the human's free text and context is empty.
    """

    human_choice: HumanChoice
    context: Optional[HumanChoiceContext] = None


OTHER_VALUE = "Other"

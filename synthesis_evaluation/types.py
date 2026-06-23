"""Canonical dataclasses used across the synthesis-evaluation suite.

Five entity labels in scope: NAME_GIVEN, NAME_FAMILY, EMAIL_ADDRESS,
USERNAME, ORGANIZATION. ORGANIZATION spans are grouped by the employer
organization (`org_group`) rather than by character — org scoring is
org-group-level and character-independent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

LABELS: Tuple[str, ...] = (
    "NAME_GIVEN", "NAME_FAMILY", "EMAIL_ADDRESS", "USERNAME", "ORGANIZATION",
)
TIER_MINIMAL  = "minimal"
TIER_ENTITY   = "entity"
TIER_COMPLETE = "complete"
TIERS = (TIER_MINIMAL, TIER_ENTITY, TIER_COMPLETE)


@dataclass(frozen=True)
class GroundTruthSpan:
    """Ground-truth annotation with character assignment.

    `characters` may contain >1 id when a surface form is genuinely
    shared (e.g. "Donovan" → both Megan Donovan and Brian Donovan).
    Scorers treat each character listed as an owner of the span.

    ORGANIZATION spans carry `org_group` (the canonical employer-org
    name) instead of character attribution; `characters` is empty for
    them.
    """
    text: str
    start: int
    end: int
    label: str
    characters: Tuple[str, ...]
    disambiguation_source: Optional[str] = None
    disambiguation_confidence: Optional[str] = None
    org_group: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "GroundTruthSpan":
        return cls(
            text=d["text"],
            start=d["start"],
            end=d["end"],
            label=d["label"],
            characters=tuple(d.get("characters") or ()),
            disambiguation_source=d.get("disambiguation_source"),
            disambiguation_confidence=d.get("disambiguation_confidence"),
            org_group=d.get("org_group"),
        )


@dataclass(frozen=True)
class SynthEntity:
    """One detected-and-synthesized entity in a synthesizer's output.

    Offsets refer to the original text. `new_text` is the synthetic
    replacement. `group_id` is set only when the synthesizer emits
    explicit groupings (tier == 'complete'); else None.
    """
    start: int
    end: int
    label: str
    text: str            # original surface form (text[start:end] in the original)
    new_text: str        # synthetic replacement
    group_id: Optional[str] = None
    score: Optional[float] = None

    @classmethod
    def from_dict(cls, d: dict) -> "SynthEntity":
        return cls(
            start=d["start"],
            end=d["end"],
            label=d["label"],
            text=d["text"],
            new_text=d["new_text"],
            group_id=d.get("group_id"),
            score=d.get("score"),
        )


@dataclass
class SynthOutput:
    """The synthesizer's output for one row."""
    synthetic_text: str
    entities: Optional[List[SynthEntity]]   # None ⇒ tier == minimal
    tier: str                               # one of TIERS

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"unknown tier: {self.tier!r}")


@dataclass
class EvalRow:
    """One row of input to the scorers (ground truth + synthesizer output)."""
    meta: dict
    text: str
    ground_truth_spans: List[GroundTruthSpan]
    synthesis: SynthOutput

    @classmethod
    def from_dict(cls, d: dict) -> "EvalRow":
        gt = [GroundTruthSpan.from_dict(s) for s in (d.get("ground_truth_spans") or [])]
        sy = d["synthesis"]
        ents = sy.get("entities")
        synth = SynthOutput(
            synthetic_text=sy["synthetic_text"],
            entities=[SynthEntity.from_dict(e) for e in ents] if ents is not None else None,
            tier=sy["tier"],
        )
        return cls(
            meta=d.get("meta") or {},
            text=d.get("text") or "",
            ground_truth_spans=gt,
            synthesis=synth,
        )


@dataclass
class Character:
    """Character metadata loaded from `pii_per_character.json`."""
    character_id: str
    canonical_name: Optional[str]
    first_names: List[str] = field(default_factory=list)
    last_names: List[str] = field(default_factory=list)
    nicknames: List[str] = field(default_factory=list)
    emails: List[str] = field(default_factory=list)
    slack_handles: List[str] = field(default_factory=list)
    alt_slack_handles: List[str] = field(default_factory=list)
    organizations: List[str] = field(default_factory=list)
    role_raw: Optional[str] = None
    job_title: Optional[str] = None

    @classmethod
    def from_dict(cls, cid: str, d: dict) -> "Character":
        return cls(
            character_id=cid,
            canonical_name=d.get("canonical_name"),
            first_names=list(d.get("first_names") or []),
            last_names=list(d.get("last_names") or []),
            nicknames=list(d.get("nicknames") or []),
            emails=[e["value"] if isinstance(e, dict) else e
                    for e in (d.get("emails") or [])],
            slack_handles=list(d.get("slack_handles") or []),
            alt_slack_handles=list(d.get("alt_slack_handles") or []),
            organizations=list(d.get("organizations") or []),
            role_raw=d.get("role_raw"),
            job_title=d.get("job_title"),
        )

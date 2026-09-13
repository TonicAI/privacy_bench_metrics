"""Canonical dataclasses used across the synthesis-evaluation suite.

Ten entity labels in scope: the five original ones (NAME_GIVEN, NAME_FAMILY,
EMAIL_ADDRESS, USERNAME, ORGANIZATION) plus PHONE_NUMBER, LOCATION_ADDRESS,
EMPLOYEE_ID, ACCOUNT_NUMBER and URL.

Ownership is decided per span, not per label. A span whose ``characters``
list is non-empty belongs to those characters; a span carrying ``org_group``
and no characters belongs to that organization group; ORGANIZATION spans
always belong to their org group (their ``characters`` list only enumerates
the members). Everything a character or org group owns — old labels and new —
is judged together for that owner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

LABELS: Tuple[str, ...] = (
    "NAME_GIVEN", "NAME_FAMILY", "EMAIL_ADDRESS", "USERNAME", "ORGANIZATION",
    "PHONE_NUMBER", "LOCATION_ADDRESS", "EMPLOYEE_ID", "ACCOUNT_NUMBER", "URL",
)
ORIGINAL_LABELS: Tuple[str, ...] = LABELS[:5]
# labels a character can own / an org group can own
PERSON_LABELS: Tuple[str, ...] = tuple(l for l in LABELS if l != "ORGANIZATION")
ORG_LABELS: Tuple[str, ...] = ("ORGANIZATION", "LOCATION_ADDRESS", "URL", "ACCOUNT_NUMBER", "PHONE_NUMBER")

# Predicted-label vocabularies differ across NER engines for the newer types (Textual says
# NUMERIC_PII, Presidio says LOCATION or US_BANK_NUMBER ...). A prediction under one of these
# labels counts as label-matched detection of the gold labels listed. The five original labels
# stay strict so historical scores are unchanged.
LABEL_ALIASES: Dict[str, Tuple[str, ...]] = {
    "PHONE": ("PHONE_NUMBER",), "TELEPHONE": ("PHONE_NUMBER",), "PHONE_NUMBER": ("PHONE_NUMBER",),
    "LOCATION": ("LOCATION_ADDRESS",), "ADDRESS": ("LOCATION_ADDRESS",), "STREET_ADDRESS": ("LOCATION_ADDRESS",),
    "LOCATION_ADDRESS": ("LOCATION_ADDRESS",), "GPE": ("LOCATION_ADDRESS",),
    "URL": ("URL",), "DOMAIN_NAME": ("URL",), "WEBSITE": ("URL",), "LINK": ("URL",),
    "EMPLOYEE_ID": ("EMPLOYEE_ID",), "ACCOUNT_NUMBER": ("ACCOUNT_NUMBER",),
    "NUMERIC_PII": ("EMPLOYEE_ID", "ACCOUNT_NUMBER"), "HEALTHCARE_ID": ("EMPLOYEE_ID", "ACCOUNT_NUMBER"),
    "ID": ("EMPLOYEE_ID", "ACCOUNT_NUMBER"), "ID_NUMBER": ("EMPLOYEE_ID", "ACCOUNT_NUMBER"),
    "IDENTIFIER": ("EMPLOYEE_ID", "ACCOUNT_NUMBER"), "US_BANK_NUMBER": ("ACCOUNT_NUMBER",),
    "IBAN_CODE": ("ACCOUNT_NUMBER",), "CREDIT_CARD": ("ACCOUNT_NUMBER",), "BANK_ACCOUNT": ("ACCOUNT_NUMBER",),
}


def labels_match(gold_label: str, pred_label: Optional[str]) -> bool:
    """Label-matched detection: equal labels, or a predicted label that is an alias of the gold label."""
    if pred_label is None:
        return False
    return pred_label == gold_label or gold_label in LABEL_ALIASES.get(str(pred_label).upper(), ())


def in_scope(label: Optional[str]) -> bool:
    """A predicted label the eval can use: one of ours or an alias of one of ours."""
    return label is not None and (label in LABELS or str(label).upper() in LABEL_ALIASES)


def owner_kind(label: str, characters, org_group: Optional[str]) -> Optional[str]:
    """'person' | 'org' | None for a gold span."""
    if label == "ORGANIZATION":
        return "org" if org_group else None
    if characters:
        return "person"
    if org_group:
        return "org"
    return None
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

    @property
    def owner(self) -> Optional[str]:
        return owner_kind(self.label, self.characters, self.org_group)

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

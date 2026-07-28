"""The three headline PrivacyBench metrics, with fixed denominators.

Combines the ground truth, the synthesizer's entities, and the LLM
judge's verdicts into one classification per gold span:

  detected   ⇔ a predicted entity with the gold span's label overlaps
               the gold span (label-matched, largest overlap wins — the
               same matching rule the judge's mapping join uses; a span
               found only under a different label is an NER miss)
  replaced   ⇔ the label-matched prediction's new_text != text
  coherent   ⇔ replaced AND the judge marked that (surface → value)
               mapping coherent

  NER recall           = detected / gold      (denominator: all gold spans)
  synthesis accuracy   = coherent / detected  (denominator: all detected
                         gold spans — an identity mapping, where the
                         synthetic value equals the original, always
                         counts as incoherent; spans the judge failed
                         to evaluate stay in the denominator but can't
                         score)
  combined accuracy    = coherent / gold      (end-to-end; equals
                         NER recall × synthesis accuracy)

Each stage's failures land in exactly one metric: a span the NER step
missed only hurts recall; a detected span left unchanged or replaced
incoherently only hurts synthesis accuracy; the combined score is
their product, so a pipeline cannot inflate any column by detecting
less or by echoing values back unchanged.

Also collects identity-mapping diagnostics (which detected spans were
left unchanged) for the report and viewer, mirroring the recall
scorer's false-negative examples.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from .score_realism_llm import CHAR_LABELS
from .score_recall import _match_pred
from .types import EvalRow, LABELS

_COUNT_KEYS = ("gold", "detected", "coherent", "incoherent", "skipped")
TOP_K_IDENTITY = 100


def build_verdict_lookups(realism_llm: dict) -> Tuple[dict, dict]:
    """Index the judge's verdicts.

    Returns ``(char_lookup, org_lookup)``:
      char_lookup: {(character_id, label, surface) → verdict entry}
      org_lookup:  {(org_group, surface) → verdict entry}
    """
    char_lookup: Dict[Tuple[str, str, str], dict] = {}
    for cid, blk in (realism_llm.get("per_character") or {}).items():
        parsed = blk.get("parsed") or {}
        for entry in (parsed.get("verdicts") or []) if isinstance(parsed, dict) else []:
            if (isinstance(entry, dict) and isinstance(entry.get("surface"), str)
                    and entry.get("label") in CHAR_LABELS):
                char_lookup[(cid, entry["label"], entry["surface"])] = entry
    org_lookup: Dict[Tuple[str, str], dict] = {}
    for grp, blk in (realism_llm.get("per_org_group") or {}).items():
        parsed = blk.get("parsed") or {}
        for entry in (parsed.get("verdicts") or []) if isinstance(parsed, dict) else []:
            if isinstance(entry, dict) and isinstance(entry.get("surface"), str):
                org_lookup[(grp, entry["surface"])] = entry
    return char_lookup, org_lookup


def _entry_value_flag(entry: dict, value: str) -> bool:
    """Per-value verdict; a value the judge didn't list individually
    inherits the surface-level verdict."""
    for vv in (entry.get("values") or []):
        if isinstance(vv, dict) and vv.get("value") == value:
            return bool(vv.get("coherent"))
    return bool(entry.get("coherent"))


def _flag_for_span(g, value: str, char_lookup: dict,
                   org_lookup: dict) -> Optional[bool]:
    """Judge verdict for one gold span's synthetic value.
    True/False = judged; None = no verdict available (skipped)."""
    if g.label == "ORGANIZATION":
        if not g.org_group:
            return None
        entry = org_lookup.get((g.org_group, g.text))
        return _entry_value_flag(entry, value) if entry else None
    flags = [
        _entry_value_flag(entry, value)
        for cid in g.characters
        if (entry := char_lookup.get((cid, g.label, g.text))) is not None
    ]
    if not flags:
        return None
    return any(flags)


def compute_metrics(rows: List[EvalRow], realism_llm: dict) -> dict:
    """Compute the headline metric block from eval rows + judge results."""
    realism_llm = realism_llm or {}
    judge_skipped = bool(realism_llm.get("skipped_reason"))
    char_lookup, org_lookup = build_verdict_lookups(realism_llm)

    overall = {lab: dict.fromkeys(_COUNT_KEYS, 0) for lab in LABELS}
    # Per-character, character labels only. A gold span shared by N
    # characters counts under each owner (per-character sums can exceed
    # the global totals), matching the recall scorer's convention.
    per_char: Dict[str, Dict[str, dict]] = defaultdict(
        lambda: {lab: dict.fromkeys(_COUNT_KEYS, 0) for lab in CHAR_LABELS})

    identity_text_label: Counter = Counter()
    identity_examples: List[dict] = []

    for row_idx, row in enumerate(rows):
        preds = list(row.synthesis.entities or [])
        for g in row.ground_truth_spans:
            if g.label not in LABELS:
                continue
            char_slots = ([per_char[cid][g.label] for cid in g.characters]
                          if g.label in CHAR_LABELS else [])

            def bump(key: str) -> None:
                overall[g.label][key] += 1
                for slot in char_slots:
                    slot[key] += 1

            bump("gold")
            p = _match_pred(g, preds)          # label-matched overlap;
            if p is None:                      # what the judge saw
                continue                       # NER miss (not found, or
            bump("detected")                   # found under another label)
            if p.new_text == p.text:
                bump("incoherent")             # identity mapping: always a miss
                identity_text_label[(g.text, g.label)] += 1
                if len(identity_examples) < TOP_K_IDENTITY:
                    meta = row.meta or {}
                    identity_examples.append({
                        "row_idx": row_idx,
                        "row_id": meta.get("row_id") or meta.get("cell_id"),
                        "gold": {"text": g.text, "label": g.label,
                                 "start": g.start, "end": g.end,
                                 "characters": list(g.characters)},
                        "snippet": row.text[max(0, g.start - 60):g.end + 60],
                    })
            else:
                flag = _flag_for_span(g, p.new_text, char_lookup, org_lookup)
                bump("coherent" if flag is True
                     else "incoherent" if flag is False else "skipped")

    def _rates(c: dict) -> dict:
        out = dict(c)
        out["ner_recall"] = (c["detected"] / c["gold"]) if c["gold"] else None
        if judge_skipped:
            out["synthesis_accuracy"] = out["combined_accuracy"] = None
        else:
            out["synthesis_accuracy"] = (
                c["coherent"] / c["detected"] if c["detected"] else None)
            out["combined_accuracy"] = (
                c["coherent"] / c["gold"] if c["gold"] else None)
        return out

    totals = dict.fromkeys(_COUNT_KEYS, 0)
    for lab in LABELS:
        for k in _COUNT_KEYS:
            totals[k] += overall[lab][k]

    return {
        "judge_skipped": judge_skipped,
        "overall": _rates(totals),
        "by_label": {lab: _rates(overall[lab]) for lab in LABELS},
        "per_character": {
            cid: {"by_label": {lab: dict(blk[lab]) for lab in CHAR_LABELS}}
            for cid, blk in sorted(per_char.items())
        },
        "identity_examples": identity_examples,
        "identity_by_surface": [
            {"text": t, "label": l, "count": n}
            for (t, l), n in identity_text_label.most_common()
        ],
    }

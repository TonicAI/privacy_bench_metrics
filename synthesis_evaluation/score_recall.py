"""NER recall (detection) against the ground truth.

We don't compute precision or F1: the ground-truth spans aren't an
exhaustive listing of every PII string the synthesizer might also have
masked, so "predicted-but-not-in-gold" is not a meaningful error.

Recall definition
-----------------
Detection only — whether the value was actually *changed* is scored
separately, as synthesis accuracy (see metrics.py). For every
ground-truth span in scope:
  - TRUE POSITIVE  ⇔ a predicted entity with the gold span's label
                     overlaps the gold span.
  - FALSE NEGATIVE ⇔ no label-matched overlapping prediction: the gold
                     PII was never detected, or was detected only under
                     a different label.

Recall = tp / (tp + fn), reported overall and per label; the
denominator is always the total gold-span count.

``_match_pred`` — the label-matched largest-overlap matcher — is the
single span-matching rule shared by every scorer (detection here, the
judge's mapping join in score_realism_llm, and metrics.py).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional

from .types import EvalRow, LABELS, GroundTruthSpan, SynthEntity

TOP_K_FN = 100


def _overlaps(a_s: int, a_e: int, b_s: int, b_e: int) -> bool:
    return a_s < b_e and b_s < a_e


def _match_pred(g: GroundTruthSpan, preds: List[SynthEntity]) -> Optional[SynthEntity]:
    """The predicted entity with the gold span's label and the largest
    overlap onto `g`; None when no same-label prediction overlaps."""
    best = None
    for p in preds:
        if p.label != g.label:
            continue
        if _overlaps(g.start, g.end, p.start, p.end):
            ov = min(g.end, p.end) - max(g.start, p.start)
            if best is None or ov > best[0]:
                best = (ov, p)
    return best[1] if best else None


def score(rows: List[EvalRow]) -> dict:
    tp = Counter()  # label → count
    fn = Counter()
    # Per-character per-label counters. A gold span shared by N
    # characters counts under each character (so per-character sums
    # exceed the global totals when multi-character spans exist).
    per_char_tp: Dict[str, Counter] = defaultdict(Counter)
    per_char_fn: Dict[str, Counter] = defaultdict(Counter)
    # Every (text, label) FN occurrence, deduped across the corpus.
    fn_text_label: Counter = Counter()
    fn_examples = []

    for row_idx, row in enumerate(rows):
        gts = [g for g in row.ground_truth_spans if g.label in LABELS]
        preds = list(row.synthesis.entities or [])
        for g in gts:
            detected = _match_pred(g, preds) is not None
            if detected:
                tp[g.label] += 1
                for cid in g.characters:
                    per_char_tp[cid][g.label] += 1
            else:
                fn[g.label] += 1
                for cid in g.characters:
                    per_char_fn[cid][g.label] += 1
                fn_text_label[(g.text, g.label)] += 1
                if len(fn_examples) < TOP_K_FN:
                    fn_examples.append({
                        "row_idx": row_idx,
                        "cell_id": row.meta.get("row_id") or row.meta.get("cell_id"),
                        "gold": {"text": g.text, "label": g.label,
                                 "start": g.start, "end": g.end,
                                 "characters": list(g.characters)},
                        "snippet": row.text[max(0, g.start - 60):g.end + 60],
                    })

    def _recall(t: int, f: int) -> Optional[float]:
        return t / (t + f) if (t + f) else None

    def _block(tp_c: Counter, fn_c: Counter) -> dict:
        by_label = {
            lab: {
                "tp":     tp_c[lab],
                "fn":     fn_c[lab],
                "total":  tp_c[lab] + fn_c[lab],
                "recall": _recall(tp_c[lab], fn_c[lab]),
            }
            for lab in LABELS
        }
        t = sum(tp_c.values())
        f = sum(fn_c.values())
        overall = {
            "tp":     t,
            "fn":     f,
            "total":  t + f,
            "recall": _recall(t, f),
        }
        return {"overall": overall, "by_label": by_label}

    global_block = _block(tp, fn)
    per_character: Dict[str, dict] = {}
    for cid in sorted(set(per_char_tp) | set(per_char_fn)):
        per_character[cid] = _block(per_char_tp[cid], per_char_fn[cid])

    # Build the (text, label) → count list for the FN-by-surface table.
    fn_by_surface: List[dict] = [
        {"text": t, "label": l, "count": n}
        for (t, l), n in fn_text_label.most_common()
    ]

    return {
        "overall":       global_block["overall"],
        "by_label":      global_block["by_label"],
        "per_character": per_character,
        "fn_examples":   fn_examples,
        "fn_by_surface": fn_by_surface,
    }

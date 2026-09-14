"""Span-level NER precision / recall / F1 against a human-annotated gold file.

The synthesis evaluation (score_recall.py) reports recall only, because the
generated ground truth is not an exhaustive listing of every string an NER
engine might defensibly flag. The human-annotated gold released on the
dataset under `human_annotations/<set>.jsonl` *is* an exhaustive annotation
of the five message PII labels, so against it precision and F1 are
meaningful. This module scores raw NER predictions against that gold; it is
the scoring used for the dataset card's "NER engines scored on the human
gold" table.

Matching rules
--------------
Two variants are reported side by side:

- **overlap** — a predicted span matches a gold span when the labels are
  equal and the character ranges overlap (half-open ranges; merely touching
  does not count). Matching is not one-to-one: each side is tested
  independently, so one prediction may detect several gold spans and vice
  versa.
    recall    = gold spans with a label-matched overlapping prediction / gold spans
    precision = predicted spans with a label-matched overlapping gold span / predicted spans
- **exact** — the (start, end, label) triples must be identical.

F1 is the harmonic mean of the corresponding precision and recall. Rows are
joined by row id; predicted rows with no gold row are ignored (reported by
the CLI, never scored). Spans whose label is outside LABELS are dropped from
both sides before scoring.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

from .types import LABELS, in_scope, labels_match


def _row_id(row: dict) -> str:
    rid = row.get("row_id")
    if rid is None:
        meta = row.get("meta") or {}
        rid = meta.get("row_id") or meta.get("cell_id")
    if rid is None:
        raise KeyError(f"row has no row_id (top-level, meta.row_id, or meta.cell_id): {list(row)}")
    return str(rid)


def _row_spans(row: dict, field: Optional[str]) -> List[dict]:
    if field is not None:
        return row[field]
    for key in ("entities", "spans", "ground_truth_spans"):
        if key in row:
            return row[key]
    raise KeyError(f"row has no span field (entities, spans, or ground_truth_spans): {list(row)}")


def spans_by_row(rows: List[dict], field: Optional[str] = None) -> Dict[str, List[dict]]:
    """row id -> spans with a benchmark label. `field` forces the span key;
    by default the first of entities / spans / ground_truth_spans is used."""
    out: Dict[str, List[dict]] = {}
    for row in rows:
        # gold rows carry benchmark labels; prediction rows may use an engine's own vocabulary
        # (NUMERIC_PII, US_BANK_NUMBER, LOCATION, ...) which types.LABEL_ALIASES maps onto ours
        spans = [s for s in _row_spans(row, field) if in_scope(s["label"])]
        out[_row_id(row)] = spans
    return out


def _overlaps(a: dict, b: dict) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def score_counts(gold: Dict[str, List[dict]], pred: Dict[str, List[dict]]) -> Counter:
    """Raw match counts of one predictions file against one gold file."""
    c: Counter = Counter()
    for rid, g_spans in gold.items():
        p_spans = pred.get(rid, [])
        c["gold"] += len(g_spans)
        c["pred"] += len(p_spans)
        c["detected"] += sum(
            1 for g in g_spans
            if any(labels_match(g["label"], p["label"]) and _overlaps(p, g) for p in p_spans))
        c["matched"] += sum(
            1 for p in p_spans
            if any(labels_match(g["label"], p["label"]) and _overlaps(g, p) for g in g_spans))
        c["exact_tp"] += sum(
            1 for g in g_spans
            if any(labels_match(g["label"], p["label"]) and p["start"] == g["start"] and p["end"] == g["end"] for p in p_spans))
        for g in g_spans:
            c[f"gold_{g['label']}"] += 1
            if any(labels_match(g["label"], p["label"]) and _overlaps(p, g) for p in p_spans):
                c[f"detected_{g['label']}"] += 1
    return c


def prf(c: Counter) -> Dict[str, Optional[float]]:
    """Precision / recall / F1 (overlap and exact) from raw counts."""
    def f1(p: Optional[float], r: Optional[float]) -> Optional[float]:
        return 2 * p * r / (p + r) if (p is not None and r is not None and p + r) else None
    p_o = c["matched"] / c["pred"] if c["pred"] else None
    r_o = c["detected"] / c["gold"] if c["gold"] else None
    p_e = c["exact_tp"] / c["pred"] if c["pred"] else None
    r_e = c["exact_tp"] / c["gold"] if c["gold"] else None
    return {"P": p_o, "R": r_o, "F1": f1(p_o, r_o),
            "P_exact": p_e, "R_exact": r_e, "F1_exact": f1(p_e, r_e)}


def label_recall(c: Counter) -> Dict[str, Optional[float]]:
    """Overlap recall per label from raw counts."""
    return {
        label: (c[f"detected_{label}"] / c[f"gold_{label}"] if c[f"gold_{label}"] else None)
        for label in LABELS
    }

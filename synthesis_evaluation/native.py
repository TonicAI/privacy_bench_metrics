"""Join native-coordinate predictions to the ground truth.

The published ground truth (`ground_truth/<set>/ground_truth.jsonl`) locates every gold span in the
containing file's own coordinates: character/byte offsets into the `.eml`, a JSON pointer plus offsets
for a Slack message, per-glyph boxes for a PDF, `<w:t>` XPath fragments for a DOCX, a cell for an XLSX,
a row/column for a CSV. A pipeline that works on the raw export reports its detected spans and
replacements in the same shape (see the dataset card, "Synthesizer output"):

    {"file": {"kind": "pdf", "path": "email/<id>.eml", "container": {...}},
     "spans": [{"label": "NAME_GIVEN", "text": "Megan", "new_text": "Alicia", "location": {...}}]}

This module matches each gold span to the predicted span of the same file unit that overlaps it most
in native coordinates (at least `MATCH_THRESHOLD` of the union, Jaccard-style), and turns the result
into the offset-based `EvalRow` the scorers already understand: the matched prediction is placed at
the gold span's own `start`/`end` in the row text, carrying the pipeline's label and replacement. From
there NER recall, the LLM judge and the headline metrics run unchanged. Nothing is read from the
export files themselves: the join is purely on the coordinates both sides report.

Gold spans whose `location.status` is `not_rendered` (footer text the PDF renderer clipped; the
characters are not in the file) are left out of the denominator. Predictions that overlap no gold span
are not scored: the gold covers only the seed characters and their organizations, so an unmatched
prediction is not necessarily wrong.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from .types import EvalRow, TIER_ENTITY, in_scope, labels_match

MATCH_THRESHOLD = 0.5
KINDS = ("eml", "slack", "pdf", "docx", "xlsx", "csv")


# ---------------------------------------------------------------------------
# keys and geometry
# ---------------------------------------------------------------------------

def rel_path(path: str) -> str:
    """Paths relative to the export root; tolerate a leading `input/` or `/work/input/`."""
    p = (path or "").strip().replace("\\", "/")
    for pre in ("/work/input/", "input/", "/work/", "./"):
        if p.startswith(pre):
            p = p[len(pre):]
    return p


def norm_kind(kind: Optional[str]) -> str:
    k = (kind or "").lower()
    return "eml" if k == "email" else k


def unit_key(kind: str, path: str, container: Optional[dict]) -> Tuple:
    """Identity of a file unit: the file, plus the attachment part for documents embedded in an email."""
    part = None
    if container and container.get("kind") == "eml_attachment":
        part = container.get("part_index")
        if part is None:
            part = ("name", container.get("filename"))
    return (norm_kind(kind), rel_path(path), part)


def iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def _norm_xpath(x: str) -> str:
    return re.sub(r"\[1\]", "", x or "")


def _gold_parts(loc: dict) -> List[dict]:
    """A gold location may be split over several structural parts (`crosses_parts`)."""
    return loc["fragments"] if loc.get("crosses_parts") else [loc]


def _gold_docx_fragments(loc: dict) -> List[Tuple[str, int, int]]:
    out = []
    for p in _gold_parts(loc):
        for fr in p.get("fragments") or []:
            out.append((_norm_xpath(fr["xpath"]), fr["start"], fr["end"]))
    return out


def _pred_docx_fragments(loc: dict) -> List[Tuple[str, int, int]]:
    out = []
    for fr in loc.get("fragments") or []:
        out.append((_norm_xpath(fr.get("xpath", "")), int(fr.get("start", 0)), int(fr.get("end", 0))))
    if not out and loc.get("xpath"):
        out.append((_norm_xpath(loc["xpath"]), int(loc.get("start", loc.get("char_start", 0))),
                    int(loc.get("end", loc.get("char_end", 0)))))
    return out


def _gold_pdf_chars(loc: dict) -> List[list]:
    out: List[list] = []
    for p in _gold_parts(loc):
        out.extend(p.get("chars") or [])
    return out


def overlap(kind: str, pred: dict, gold: dict) -> float:
    """Fraction in [0, 1] of the gold location covered by the predicted location, in native coordinates."""
    try:
        if kind == "eml":
            if "file_char_start" in pred:
                return iou(pred["file_char_start"], pred["file_char_end"], gold["file_char_start"], gold["file_char_end"])
            if "file_byte_start" in pred:
                return iou(pred["file_byte_start"], pred["file_byte_end"], gold["file_byte_start"], gold["file_byte_end"])
            return 0.0
        if kind == "slack":
            pp = (pred.get("json_pointer") or "").rstrip("/")
            if not pp.endswith("/text"):
                pp += "/text"
            if pp != gold["json_pointer"]:
                return 0.0
            return iou(pred["start"], pred["end"], gold["start"], gold["end"])
        if kind == "pdf":
            gchars = _gold_pdf_chars(gold)
            gset = {(c[1], round(c[2], 1), round(c[3], 1)) for c in gchars}
            if pred.get("chars"):
                pset = {(c[1], round(float(c[2]), 1), round(float(c[3]), 1)) for c in pred["chars"]}
                return len(gset & pset) / len(gset | pset) if gset | pset else 0.0
            boxes = pred.get("boxes") or []
            if not boxes or not gchars:
                return 0.0
            covered = 0
            for c in gchars:
                cx, cy = (c[2] + c[4]) / 2, (c[3] + c[5]) / 2
                if any(b.get("page") == c[1] and b["x0"] - 0.5 <= cx <= b["x1"] + 0.5 and b["top"] - 0.5 <= cy <= b["bottom"] + 0.5
                       for b in boxes):
                    covered += 1
            return covered / len(gchars)
        if kind == "docx":
            gf, pf = _gold_docx_fragments(gold), _pred_docx_fragments(pred)
            total = sum(e - s for _, s, e in gf)
            if not total:
                return 0.0
            inter = 0
            for gx, gs, ge in gf:
                for px, ps, pe in pf:
                    if px == gx:
                        inter += max(0, min(ge, pe) - max(gs, ps))
            ptotal = sum(e - s for _, s, e in pf)
            return inter / max(total, ptotal) if max(total, ptotal) else 0.0
        if kind == "xlsx":
            best = 0.0
            for gg in _gold_parts(gold):
                if (pred.get("sheet") or "").lower() != (gg.get("sheet") or "").lower() \
                        or (pred.get("cell") or "").upper() != (gg.get("cell") or "").upper():
                    continue
                if "char_start" in pred:
                    best = max(best, iou(pred["char_start"], pred["char_end"], gg["char_start"], gg["char_end"]))
                else:
                    best = max(best, 1.0)
            return best
        if kind == "csv":
            best = 0.0
            for gg in _gold_parts(gold):
                if "file_char_start" in pred and "file_char_start" in gg and pred.get("row") is None:
                    best = max(best, iou(pred["file_char_start"], pred["file_char_end"], gg["file_char_start"], gg["file_char_end"]))
                    continue
                if int(pred.get("row", -1)) != gg.get("row") or int(pred.get("col", -1)) != gg.get("col"):
                    continue
                if "value_char_start" in pred:
                    best = max(best, iou(pred["value_char_start"], pred["value_char_end"], gg["value_char_start"], gg["value_char_end"]))
                elif "file_char_start" in pred:
                    best = max(best, iou(pred["file_char_start"], pred["file_char_end"], gg["file_char_start"], gg["file_char_end"]))
                else:
                    best = max(best, 1.0)
            return best
    except (KeyError, TypeError, ValueError):
        return 0.0
    return 0.0


# ---------------------------------------------------------------------------
# loading and joining
# ---------------------------------------------------------------------------

def load_predictions(rows: Iterable[dict]) -> Dict[Tuple, List[dict]]:
    """`{unit_key: [span, ...]}` from prediction rows (`file` + `spans`), in-scope labels only."""
    out: Dict[Tuple, List[dict]] = defaultdict(list)
    for r in rows:
        f = r.get("file") or {}
        kind = norm_kind(f.get("kind"))
        key = unit_key(kind, f.get("path", ""), f.get("container") or None)
        for s in r.get("spans") or r.get("predicted_spans") or []:
            if not in_scope(s.get("label")):
                continue
            out[key].append({"label": s.get("label"), "text": s.get("text", ""), "new_text": s.get("new_text", s.get("text", "")),
                             "location": s.get("location") or {}})
    return out


def gold_key(gt_row: dict) -> Tuple:
    f = gt_row.get("file") or {}
    return unit_key(f.get("kind"), f.get("path", ""), f.get("container") or None)


def join(gt_rows: List[dict], preds: Dict[Tuple, List[dict]], threshold: float = MATCH_THRESHOLD
         ) -> Tuple[List[EvalRow], List[str], dict]:
    """Build one EvalRow per ground-truth row.

    For every scorable gold span, the same-unit prediction with the largest native overlap (>= threshold)
    and a matching label becomes an entity at the gold span's offsets, so the offset-based scorers see
    exactly what the native match found. A gold span whose only overlapping prediction carries another
    label is left undetected (the published recall definition is label-matched).

    Returns (rows, kind per row, diagnostics).
    """
    rows: List[EvalRow] = []
    kinds: List[str] = []
    diag: Dict[str, Counter] = {"gold": Counter(), "excluded_not_rendered": Counter(), "predicted": Counter(),
                                "matched": Counter(), "matched_other_label": Counter(), "units_without_predictions": Counter()}
    # Slack day files hold many messages: predictions may be keyed per day file while gold rows are per message
    by_file: Dict[Tuple, List[dict]] = defaultdict(list)
    for k, lst in preds.items():
        by_file[(k[0], k[1])].extend(lst)
        diag["predicted"][k[0]] += len(lst)
    for gt in gt_rows:
        f = gt.get("file") or {}
        kind = norm_kind(f.get("kind"))
        key = gold_key(gt)
        cands = preds.get(key) or by_file.get((key[0], key[1])) or []
        if not cands:
            diag["units_without_predictions"][kind] += 1
        spans_in, entities = [], []
        for s in gt.get("ground_truth_spans") or []:
            loc = s.get("location") or {}
            if loc.get("status") == "not_rendered":
                diag["excluded_not_rendered"][kind] += 1
                continue
            spans_in.append({k2: v for k2, v in s.items() if k2 != "location"})
            diag["gold"][kind] += 1
            best, best_p = 0.0, None
            for p in cands:
                sc = overlap(kind, p["location"], loc)
                if sc > best:
                    best, best_p = sc, p
            if best_p is None or best < threshold:
                continue
            if not labels_match(s["label"], best_p["label"]):
                diag["matched_other_label"][kind] += 1
                # a same-label prediction may still exist with a smaller (but sufficient) overlap
                alt = [(overlap(kind, p["location"], loc), p) for p in cands if labels_match(s["label"], p["label"])]
                alt = [(sc, p) for sc, p in alt if sc >= threshold]
                if not alt:
                    continue
                best, best_p = max(alt, key=lambda t: t[0])
            diag["matched"][kind] += 1
            entities.append({"start": s["start"], "end": s["end"], "label": s["label"],
                             "text": s["text"], "new_text": best_p["new_text"] if best_p["new_text"] is not None else s["text"],
                             "group_id": None, "score": round(best, 4)})
        text = gt.get("text") or ""
        rows.append(EvalRow.from_dict({"meta": {**(gt.get("meta") or {}), "kind": kind, "file": f.get("path")},
                                       "text": text, "ground_truth_spans": spans_in,
                                       "synthesis": {"synthetic_text": _reconstruct(text, entities), "entities": entities,
                                                     "tier": TIER_ENTITY}}))
        kinds.append(kind)
    d = {name: dict(c) for name, c in diag.items()}
    d["threshold"] = threshold
    return rows, kinds, d


def _reconstruct(text: str, entities: List[dict]) -> str:
    for e in sorted(entities, key=lambda e: e["start"], reverse=True):
        text = text[:e["start"]] + (e.get("new_text") or e["text"]) + text[e["end"]:]
    return text

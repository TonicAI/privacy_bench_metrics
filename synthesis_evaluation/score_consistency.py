"""Per-(character, label, original surface) consistency check.

For every (character, label, original_surface) bucket, look at the
distinct synthetic values the synthesizer emitted for that bucket. A
perfectly consistent synthesizer emits exactly one synthetic value per
bucket. More than one ⇒ inconsistency (the same name, email, etc.
mapped to different synthetic forms across rows).

We also report a second, looser metric: per-(character, label) distinct
synthetic surface forms. This is the user's "number of email addresses,
number of first names" per group check — for each character group, how
many distinct synthetic surface forms appeared for each entity type?
A character with 1 distinct synthetic NAME_GIVEN across the corpus is
consistent at the identity level; >1 suggests cross-row identity drift.

Matching: a predicted entity matches a gold span when they overlap and
share the same label (same lenient rule used by the recall scorer).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from .types import EvalRow, GroundTruthSpan, LABELS, SynthEntity, in_scope, labels_match


def _overlaps(a_s: int, a_e: int, b_s: int, b_e: int) -> bool:
    return a_s < b_e and b_s < a_e


def _match_pred(g: GroundTruthSpan, preds: List[SynthEntity]) -> SynthEntity | None:
    """Find the predicted entity that lenient-matches a gold span."""
    best = None
    for p in preds:
        if not labels_match(g.label, p.label):
            continue
        if _overlaps(g.start, g.end, p.start, p.end):
            # Prefer the prediction with the most offset overlap to
            # avoid grabbing an unrelated adjacent prediction.
            ov = min(g.end, p.end) - max(g.start, p.start)
            if best is None or ov > best[0]:
                best = (ov, p)
    return best[1] if best else None


def _best_overlap_any_label(g: GroundTruthSpan, preds: List[SynthEntity]) -> SynthEntity | None:
    """Same as score_recall._best_overlap — label-agnostic overlap.

    Used to detect "passthrough" cases (gold span survived into the
    synthetic text unchanged) for the per-character mapping table.
    """
    best = None
    for p in preds:
        if _overlaps(g.start, g.end, p.start, p.end):
            ov = min(g.end, p.end) - max(g.start, p.start)
            if best is None or ov > best[0]:
                best = (ov, p)
    return best[1] if best else None


def score(rows: List[EvalRow]) -> dict:
    # (character, label, original) → set of synthetic surface forms.
    # Case is preserved on the original surface form so 'Megan' and
    # 'MEGAN' are tracked as distinct buckets.
    bucket: Dict[Tuple[str, str, str], set] = defaultdict(set)
    # (character, label) → set of unique synthetic surface forms
    per_char_label_synth: Dict[Tuple[str, str], set] = defaultdict(set)
    # (character, label) → set of unique original surface forms (for context)
    per_char_label_orig: Dict[Tuple[str, str], set] = defaultdict(set)

    # Record per-(character, label, orig) the synthetic forms seen with
    # sample row pointers — used to surface offenders.
    offender_samples: Dict[Tuple[str, str, str], Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))

    # Passthrough samples: per (cid, lab, orig), row_idxs where the
    # gold span survived into the synthetic text unchanged
    # (matches score_recall's FN definition — no overlapping prediction
    # or matched prediction has new_text == text). Tracked separately
    # so the per-character mapping table can show these as identity
    # `(orig → orig)` entries without disturbing the bucket-consistency
    # math (which only considers synthesizer-produced values).
    passthrough_samples: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)

    # Organization-group buckets (character-independent):
    # (org_group, orig) → synth set, with counts + passthroughs.
    org_bucket: Dict[Tuple[str, str], set] = defaultdict(set)
    org_samples: Dict[Tuple[str, str], Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    org_passthrough: Dict[Tuple[str, str], List[int]] = defaultdict(list)

    for row_idx, row in enumerate(rows):
        gts = [g for g in row.ground_truth_spans if g.label in LABELS]
        preds_all = list(row.synthesis.entities or [])
        preds = [p for p in preds_all if in_scope(p.label)]
        for g in gts:
            p = _match_pred(g, preds)
            is_org_grouped = g.owner == "org" and g.org_group
            if p is None:
                # No same-label prediction → check passthrough via
                # label-agnostic overlap.
                pa = _best_overlap_any_label(g, preds_all)
                if pa is None or pa.new_text == pa.text:
                    for cid in g.characters:
                        passthrough_samples[(cid, g.label, g.text)].append(row_idx)
                    if is_org_grouped:
                        org_passthrough[(g.org_group, g.text)].append(row_idx)
                continue
            if p.new_text == p.text:
                # Same-label prediction but synthesizer didn't change
                # the text → passthrough.
                for cid in g.characters:
                    passthrough_samples[(cid, g.label, g.text)].append(row_idx)
                if is_org_grouped:
                    org_passthrough[(g.org_group, g.text)].append(row_idx)
                continue
            orig_key = g.text
            for cid in g.characters:
                bucket[(cid, g.label, orig_key)].add(p.new_text)
                per_char_label_synth[(cid, g.label)].add(p.new_text)
                per_char_label_orig[(cid, g.label)].add(g.text)
                offender_samples[(cid, g.label, orig_key)][p.new_text].append(row_idx)
            if is_org_grouped:
                org_bucket[(g.org_group, orig_key)].add(p.new_text)
                org_samples[(g.org_group, orig_key)][p.new_text].append(row_idx)

    # Compute the bucket-level metric.
    total_buckets = len(bucket)
    consistent_buckets = sum(1 for s in bucket.values() if len(s) == 1)
    bucket_fraction = consistent_buckets / total_buckets if total_buckets else 0.0

    # Per-character per-label aggregates + the full original→synthetic
    # mapping (every distinct original surface form seen for this
    # character at this label, with its deduped list of synthetic
    # values). The mapping makes it easy to inspect what the
    # synthesizer produced for each ground-truth surface form.
    per_character: Dict[str, dict] = {}
    chars = sorted(
        {cid for (cid, _, _) in bucket}
        | {cid for (cid, _, _) in passthrough_samples}
    )
    for cid in chars:
        per_label_block: Dict[str, dict] = {}
        for lab in LABELS:
            orig = per_char_label_orig.get((cid, lab), set())
            synth = per_char_label_synth.get((cid, lab), set())
            # Bucket-level fraction restricted to this (cid, lab).
            keys = [k for k in bucket if k[0] == cid and k[1] == lab]
            n = len(keys)
            ok = sum(1 for k in keys if len(bucket[k]) == 1)
            # Bucket key already preserves the original casing, so each
            # case-distinct surface form has its own bucket. Build the
            # original → synthetic mapping directly.
            mapping_by_orig: Dict[str, set] = {k[2]: bucket[k] for k in keys}
            # Same mapping with per-synthetic occurrence counts, sorted
            # by count desc, plus passthrough entries showing how often
            # the synthesizer left the original surface in place.
            mapping_with_counts: Dict[str, List[dict]] = {}
            for k in keys:
                orig_text = k[2]
                synths = bucket[k]
                samples = offender_samples.get(k) or {}
                items = [
                    {"value": sv, "count": len(samples.get(sv) or [])}
                    for sv in synths
                ]
                items.sort(key=lambda d: (-d["count"], d["value"]))
                mapping_with_counts[orig_text] = items
            # Add passthrough entries — `orig → orig` with a flag —
            # for this (cid, lab). These are gold spans the synthesizer
            # did not change (either undetected or detected-but-equal).
            pt_keys = [k for k in passthrough_samples
                       if k[0] == cid and k[1] == lab]
            for k in pt_keys:
                orig_text = k[2]
                n = len(passthrough_samples[k])
                if not n:
                    continue
                entry = {"value": orig_text, "count": n, "passthrough": True}
                mapping_with_counts.setdefault(orig_text, []).append(entry)
            per_label_block[lab] = {
                "unique_orig":  len(orig),
                "unique_synth": len(synth),
                "buckets": n,
                "consistent_buckets": ok,
                "consistent_fraction": ok / n if n else None,
                "original_to_synthetic": {
                    o: sorted(s) for o, s in sorted(mapping_by_orig.items())
                },
                "original_to_synthetic_counts": {
                    o: mapping_with_counts[o]
                    for o in sorted(mapping_with_counts.keys())
                },
            }
        per_character[cid] = per_label_block

    # Surface offenders: buckets with >1 synthetic surface form.
    # Inconsistent buckets, sorted by severity (most distinct synthetic
    # values first). For each bucket we record every synthetic value
    # with its occurrence count so the renderer can show the most
    # common N and elide the rest.
    inconsistent_buckets = []
    for (cid, lab, orig), synth_set in bucket.items():
        if len(synth_set) <= 1:
            continue
        samples = offender_samples[(cid, lab, orig)]  # synth → [row_idxs]
        counts = [{"value": sv, "count": len(samples[sv])} for sv in synth_set]
        counts.sort(key=lambda d: (-d["count"], d["value"]))
        inconsistent_buckets.append({
            "character": cid,
            "label": lab,
            "original": orig,
            "n_distinct_synthetic": len(synth_set),
            "n_total_occurrences": sum(c["count"] for c in counts),
            "synthetic_value_counts": counts,
        })
    inconsistent_buckets.sort(
        key=lambda d: (-d["n_distinct_synthetic"], d["character"], d["label"], d["original"])
    )

    # ---- Organization-group consistency (character-independent) ----
    org_total = len(org_bucket)
    org_consistent = sum(1 for s in org_bucket.values() if len(s) == 1)
    per_org_group: Dict[str, dict] = {}
    org_keys_by_group: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for k in org_bucket:
        org_keys_by_group[k[0]].append(k)
    for k in org_passthrough:
        if k not in org_bucket:
            org_keys_by_group[k[0]].append(k)
    for grp, keys in sorted(org_keys_by_group.items()):
        mapping_with_counts: Dict[str, List[dict]] = {}
        n = ok = 0
        for k in sorted(set(keys), key=lambda x: x[1]):
            orig_text = k[1]
            items: List[dict] = []
            if k in org_bucket:
                n += 1
                if len(org_bucket[k]) == 1:
                    ok += 1
                samples = org_samples.get(k) or {}
                items = [{"value": sv, "count": len(samples.get(sv) or [])}
                         for sv in org_bucket[k]]
                items.sort(key=lambda d: (-d["count"], d["value"]))
            if k in org_passthrough and org_passthrough[k]:
                items.append({"value": orig_text,
                              "count": len(org_passthrough[k]),
                              "passthrough": True})
            if items:
                mapping_with_counts[orig_text] = items
        per_org_group[grp] = {
            "buckets": n,
            "consistent_buckets": ok,
            "consistent_fraction": ok / n if n else None,
            "original_to_synthetic_counts": mapping_with_counts,
        }

    return {
        "overall": {
            "buckets": total_buckets,
            "consistent_buckets": consistent_buckets,
            "consistent_fraction": bucket_fraction,
        },
        "per_character": per_character,
        "inconsistent_buckets": inconsistent_buckets,
        "total_inconsistent_buckets": len(inconsistent_buckets),
        "org_overall": {
            "buckets": org_total,
            "consistent_buckets": org_consistent,
            "consistent_fraction": org_consistent / org_total if org_total else None,
        },
        "per_org_group": per_org_group,
    }

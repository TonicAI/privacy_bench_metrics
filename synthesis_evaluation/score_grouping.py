"""Pairwise + B³ grouping metrics.

Only runs when the synthesizer emits `tier == "complete"` (every
predicted entity has a `group_id`). For `tier in {"minimal", "entity"}`
the function returns a stub `{"skipped_reason": ...}` dict and the
orchestrator omits this scorer from the summary.

For evaluation we restrict to entities that successfully matched a
ground-truth span (otherwise we can't compare against gold groups
anyway). The ground-truth group label is the span's
`characters` field — if a span is shared by multiple characters we use
the lexicographically-smallest id (so the gold grouping is a partition,
not a cover).

Metrics emitted:

  pairwise.precision / recall / f1
      Compute the set of "same-group" entity-pairs in the prediction
      and in the gold, then take the standard P/R/F over those sets.

  b_cubed.precision / recall / f1
      For each entity i, let g(i) be its gold cluster and p(i) its
      predicted cluster. B³ precision_i = |g(i) ∩ p(i)| / |p(i)|,
      recall_i = |g(i) ∩ p(i)| / |g(i)|. Macro-averaged over entities.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .types import EvalRow, GroundTruthSpan, LABELS, SynthEntity, TIER_COMPLETE


def _overlaps(a_s: int, a_e: int, b_s: int, b_e: int) -> bool:
    return a_s < b_e and b_s < a_e


def _match_pred(g: GroundTruthSpan, preds: List[SynthEntity]) -> Optional[SynthEntity]:
    best = None
    for p in preds:
        if p.label != g.label:
            continue
        if _overlaps(g.start, g.end, p.start, p.end):
            ov = min(g.end, p.end) - max(g.start, p.start)
            if best is None or ov > best[0]:
                best = (ov, p)
    return best[1] if best else None


def _gold_cluster_id(g: GroundTruthSpan) -> Optional[str]:
    """Pick a deterministic single gold cluster id for an entity."""
    if not g.characters:
        return None
    return sorted(g.characters)[0]


def _pairwise(pairs_gold: set, pairs_pred: set) -> dict:
    tp = len(pairs_gold & pairs_pred)
    fp = len(pairs_pred - pairs_gold)
    fn = len(pairs_gold - pairs_pred)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def _b_cubed(items: List[Tuple[str, str]]) -> dict:
    """items: list of (gold_cluster, pred_cluster) per entity."""
    if not items:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0}
    gold_groups: Dict[str, List[int]] = defaultdict(list)
    pred_groups: Dict[str, List[int]] = defaultdict(list)
    for i, (g, p) in enumerate(items):
        gold_groups[g].append(i)
        pred_groups[p].append(i)

    p_sum = r_sum = 0.0
    for i, (g, p) in enumerate(items):
        g_set = set(gold_groups[g])
        p_set = set(pred_groups[p])
        inter = len(g_set & p_set)
        p_sum += inter / len(p_set) if p_set else 0.0
        r_sum += inter / len(g_set) if g_set else 0.0
    n = len(items)
    p = p_sum / n
    r = r_sum / n
    f = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f, "n": n}


def score(rows: List[EvalRow]) -> dict:
    """Run the grouping metric. No-op for non-'complete' synthesizers."""
    # Determine tier from the first row that has a synthesis (they all
    # should share a tier; we sample for safety).
    tier = None
    for row in rows:
        if row.synthesis and row.synthesis.tier:
            tier = row.synthesis.tier
            break
    if tier != TIER_COMPLETE:
        return {
            "skipped": True,
            "skipped_reason": f"synthesizer tier is {tier!r}, not 'complete'",
        }

    # Collect (gold_cluster, pred_cluster) pairs for every matched entity.
    items: List[Tuple[str, str]] = []
    fragmentation: Dict[str, set] = defaultdict(set)  # gold → predicted clusters covering it
    merge: Dict[str, set] = defaultdict(set)          # predicted → gold clusters present in it

    for row in rows:
        gts = [g for g in row.ground_truth_spans if g.label in LABELS]
        preds = [p for p in (row.synthesis.entities or []) if p.label in LABELS]
        for g in gts:
            p = _match_pred(g, preds)
            if p is None or p.group_id is None:
                continue
            gold_cid = _gold_cluster_id(g)
            if gold_cid is None:
                continue
            items.append((gold_cid, p.group_id))
            fragmentation[gold_cid].add(p.group_id)
            merge[p.group_id].add(gold_cid)

    # Pairwise: build sets of within-group entity-index pairs.
    gold_by_cluster: Dict[str, List[int]] = defaultdict(list)
    pred_by_cluster: Dict[str, List[int]] = defaultdict(list)
    for i, (g, p) in enumerate(items):
        gold_by_cluster[g].append(i)
        pred_by_cluster[p].append(i)

    def _pairs_in(groups: Dict[str, List[int]]) -> set:
        out = set()
        for members in groups.values():
            m = sorted(members)
            for i in range(len(m)):
                for j in range(i + 1, len(m)):
                    out.add((m[i], m[j]))
        return out

    pairs_gold = _pairs_in(gold_by_cluster)
    pairs_pred = _pairs_in(pred_by_cluster)

    pairwise = _pairwise(pairs_gold, pairs_pred)
    b3 = _b_cubed(items)

    worst_fragmentation = sorted(
        ({"gold_character": c, "n_predicted_clusters_covering": len(s)}
         for c, s in fragmentation.items() if len(s) > 1),
        key=lambda d: -d["n_predicted_clusters_covering"],
    )[:20]
    worst_merge = sorted(
        ({"predicted_cluster": pc, "n_gold_characters_inside": len(s)}
         for pc, s in merge.items() if len(s) > 1),
        key=lambda d: -d["n_gold_characters_inside"],
    )[:20]

    return {
        "skipped": False,
        "n_entities": len(items),
        "pairwise": pairwise,
        "b_cubed": b3,
        "worst_fragmentation": worst_fragmentation,
        "worst_merge": worst_merge,
    }

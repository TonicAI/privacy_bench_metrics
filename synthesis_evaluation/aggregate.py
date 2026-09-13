"""Aggregate several evaluation runs (one per dataset) into one table.

PrivacyBench has 21 datasets; a pipeline is scored once per dataset with `run_eval` or
`run_eval_native`. This pools the resulting `results.json` files by summing the gold / detected /
coherent counts, so the pooled recall, synthesis accuracy and combined accuracy have the same fixed
denominators as the per-dataset scores (a micro average; the dataset card's baseline table used the
macro average over datasets, reported here as well).

    python -m synthesis_evaluation.aggregate synthesis_evaluation/runs/my_pipeline_* --out my_pipeline_summary.md
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

COUNT_KEYS = ("gold", "detected", "coherent", "incoherent", "skipped")


def _rates(c: Dict[str, int], judged: bool) -> dict:
    out = dict(c)
    out["ner_recall"] = c["detected"] / c["gold"] if c["gold"] else None
    out["synthesis_accuracy"] = (c["coherent"] / c["detected"] if c["detected"] else None) if judged else None
    out["combined_accuracy"] = (c["coherent"] / c["gold"] if c["gold"] else None) if judged else None
    return out


def _add(dst: Counter, block: Optional[dict]) -> None:
    for k in COUNT_KEYS:
        dst[k] += int((block or {}).get(k, 0) or 0)


def aggregate(run_dirs: List[Path]) -> dict:
    overall, by_label, by_kind = Counter(), defaultdict(Counter), defaultdict(Counter)
    per_dataset = {}
    judged = True
    for d in run_dirs:
        res = json.load(open(d / "results.json", encoding="utf-8"))
        m = res["metrics"]
        judged = judged and not m.get("judge_skipped")
        _add(overall, m["overall"])
        for lab, b in (m.get("by_label") or {}).items():
            _add(by_label[lab], b)
        for kind, b in (m.get("by_kind") or {}).items():
            _add(by_kind[kind], b)
        per_dataset[d.name] = {k: m["overall"].get(k) for k in ("gold", "ner_recall", "synthesis_accuracy", "combined_accuracy")}
    n = len(per_dataset)

    def macro(key: str) -> Optional[float]:
        vals = [v[key] for v in per_dataset.values() if v.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    return {"runs": [str(d) for d in run_dirs], "datasets": n, "judged": judged,
            "overall": _rates(overall, judged),
            "macro": {k: macro(k) for k in ("ner_recall", "synthesis_accuracy", "combined_accuracy")},
            "by_label": {lab: _rates(c, judged) for lab, c in sorted(by_label.items())},
            "by_kind": {kind: _rates(c, judged) for kind, c in by_kind.items()},
            "per_dataset": per_dataset}


def _p(x: Optional[float]) -> str:
    return f"{x:.3f}" if x is not None else "n/a"


def render(agg: dict) -> str:
    o = agg["overall"]
    lines = [f"# Pooled scores over {agg['datasets']} datasets", "",
             "| | gold | NER recall | synthesis accuracy | combined |", "|---|---|---|---|---|",
             f"| pooled (micro) | {o['gold']:,} | {_p(o['ner_recall'])} | {_p(o['synthesis_accuracy'])} | {_p(o['combined_accuracy'])} |",
             f"| macro over datasets | | {_p(agg['macro']['ner_recall'])} | {_p(agg['macro']['synthesis_accuracy'])} | {_p(agg['macro']['combined_accuracy'])} |", ""]
    if agg["by_kind"]:
        lines += ["## By file kind", "", "| kind | gold | NER recall | synthesis accuracy | combined |", "|---|---|---|---|---|"]
        for kind in ("eml", "slack", "pdf", "docx", "xlsx", "csv", "messages", "documents"):
            b = agg["by_kind"].get(kind)
            if b:
                lines.append(f"| {kind} | {b['gold']:,} | {_p(b['ner_recall'])} | {_p(b['synthesis_accuracy'])} | {_p(b['combined_accuracy'])} |")
        lines.append("")
    lines += ["## By label", "", "| label | gold | NER recall | synthesis accuracy | combined |", "|---|---|---|---|---|"]
    for lab, b in agg["by_label"].items():
        if b["gold"]:
            lines.append(f"| {lab} | {b['gold']:,} | {_p(b['ner_recall'])} | {_p(b['synthesis_accuracy'])} | {_p(b['combined_accuracy'])} |")
    lines += ["", "## Per dataset", "", "| dataset | gold | NER recall | synthesis accuracy | combined |", "|---|---|---|---|---|"]
    for name, v in sorted(agg["per_dataset"].items()):
        lines.append(f"| {name} | {v['gold']:,} | {_p(v['ner_recall'])} | {_p(v['synthesis_accuracy'])} | {_p(v['combined_accuracy'])} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path, help="run directories containing results.json")
    ap.add_argument("--out", type=Path, default=None, help="write the markdown table here (also prints it)")
    ap.add_argument("--json", type=Path, default=None, help="write the aggregate as JSON")
    a = ap.parse_args()
    dirs = sorted(d for d in a.run_dirs if (d / "results.json").exists())
    agg = aggregate(dirs)
    text = render(agg)
    print(text)
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    if a.json:
        a.json.write_text(json.dumps(agg, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()

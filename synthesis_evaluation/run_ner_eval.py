"""Score NER predictions against the human-annotated gold.

Evaluates one or more (gold, predictions) file pairs — typically the six
`human_annotations/<set>.jsonl` files from the dataset paired with your NER
engine's output on the matching `tasks/<set>_messages.jsonl` — and reports
precision, recall, and F1 under the overlap and exact rules of score_ner.py,
per dataset, pooled (micro: counts summed across datasets, the dataset
card's headline numbers), and macro (unweighted mean over datasets).

Predictions are one JSON line per message. Each row needs a row id
(`row_id`, `meta.row_id`, or `meta.cell_id` — must match the gold rows) and
its predicted spans under `entities` or `spans`, each span carrying
`start` / `end` (character offsets into the message text) and `label`.
Spans with labels outside the benchmark's five are ignored.

Usage:
  python -m synthesis_evaluation.run_ner_eval \
    --gold "$DATA/human_annotations/*.jsonl" \
    --predictions-dir my_ner_output \
    --run-name my_engine_human_gold

`--predictions-dir` pairs each gold file with `<dir>/<same basename>`;
alternatively pass explicit pairs by repeating `--gold F --predictions F`.
Writes runs/<run-name>/{results.json, summary.md}.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .load import load_jsonl
from .score_ner import LABELS, label_recall, prf, score_counts, spans_by_row


def _resolve_pairs(args: argparse.Namespace) -> List[Tuple[Path, Path]]:
    golds: List[Path] = []
    for pattern in args.gold:
        hits = sorted(glob.glob(pattern))
        if not hits:
            sys.exit(f"--gold matches nothing: {pattern}")
        golds.extend(Path(h) for h in hits)

    if args.predictions_dir is not None:
        if args.predictions:
            sys.exit("pass either --predictions-dir or explicit --predictions, not both")
        return [(g, args.predictions_dir / g.name) for g in golds]

    if len(args.predictions) != len(golds):
        sys.exit(f"{len(golds)} gold files but {len(args.predictions)} --predictions; "
                 "pass one --predictions per --gold (in the same order) or use --predictions-dir")
    return list(zip(golds, (Path(p) for p in args.predictions)))


def _score_pair(gold_path: Path, pred_path: Path) -> Counter:
    if not pred_path.exists():
        sys.exit(f"predictions file not found: {pred_path}")
    gold = spans_by_row(list(load_jsonl(gold_path)), field="ground_truth_spans")
    pred = spans_by_row(list(load_jsonl(pred_path)))
    unmatched = len(set(pred) - set(gold))
    joined = len(set(pred) & set(gold))
    print(f"{gold_path.stem}: {len(gold)} gold rows, joined {joined} prediction rows"
          + (f" ({unmatched} prediction rows have no gold row and are ignored)" if unmatched else ""))
    return score_counts(gold, pred)


def _pct(x: Optional[float]) -> str:
    return f"{100 * x:.1f}%" if x is not None else "n/a"


def _metric_row(name: str, m: Dict[str, Optional[float]]) -> str:
    return (f"| {name} | {_pct(m['P'])} | {_pct(m['R'])} | {_pct(m['F1'])} "
            f"| {_pct(m['P_exact'])} | {_pct(m['R_exact'])} | {_pct(m['F1_exact'])} |")


def _macro(per_task: Dict[str, Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for key in ("P", "R", "F1", "P_exact", "R_exact", "F1_exact"):
        vals = [m[key] for m in per_task.values() if m[key] is not None]
        out[key] = sum(vals) / len(vals) if vals else None
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gold", action="append", required=True, metavar="FILE_OR_GLOB",
                        help="human-annotated gold JSONL (repeatable; globs allowed)")
    parser.add_argument("--predictions", action="append", default=[], metavar="FILE",
                        help="predictions JSONL for the corresponding --gold (repeat per pair)")
    parser.add_argument("--predictions-dir", type=Path,
                        help="directory holding one predictions file per gold file, same basename")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("synthesis_evaluation/runs"))
    args = parser.parse_args()

    run_dir = args.runs_dir / args.run_name
    if run_dir.exists():
        sys.exit(f"run dir already exists: {run_dir}")

    pairs = _resolve_pairs(args)
    counts: Dict[str, Counter] = {}
    for gold_path, pred_path in pairs:
        counts[gold_path.stem] = _score_pair(gold_path, pred_path)

    pooled = sum(counts.values(), Counter())
    per_task = {task: prf(c) for task, c in counts.items()}
    pooled_metrics = prf(pooled)
    macro_metrics = _macro(per_task)

    lines = [
        "# NER metrics against the human-annotated gold",
        "",
        f"Run: `{args.run_name}`. Matching rules: *overlap* = any character overlap with the",
        "same label; *exact* = identical (start, end, label). See `score_ner.py`.",
        "",
        "| Dataset | P | R | F1 | P (exact) | R (exact) | F1 (exact) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines += [_metric_row(task, m) for task, m in per_task.items()]
    lines += [_metric_row("**pooled**", pooled_metrics),
              _metric_row("**macro**", macro_metrics),
              "",
              "Recall (overlap) by label, pooled:",
              "",
              "| " + " | ".join(LABELS) + " |",
              "|" + "---|" * len(LABELS)]
    pooled_labels = label_recall(pooled)
    lines += ["| " + " | ".join(_pct(pooled_labels[label]) for label in LABELS) + " |", ""]
    summary = "\n".join(lines)

    run_dir.mkdir(parents=True)
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    (run_dir / "results.json").write_text(json.dumps({
        "run_name": args.run_name,
        "pairs": {g.stem: {"gold": str(g), "predictions": str(p)} for g, p in pairs},
        "counts": {task: dict(c) for task, c in counts.items()},
        "per_dataset": per_task,
        "pooled": pooled_metrics,
        "macro": macro_metrics,
        "labels_recall_pooled": pooled_labels,
    }, indent=2), encoding="utf-8")

    print(summary)
    print(f"wrote {run_dir}/summary.md and results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

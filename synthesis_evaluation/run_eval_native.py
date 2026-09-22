"""Score a raw-export pipeline's output against the published PrivacyBench ground truth.

The pipeline reads `tasks/<set>/` (the raw `.eml` files, Slack export and Drive listing) and reports,
for each file unit it de-identified, the spans it detected and their replacements in the file's native
coordinates — the same shape as `ground_truth/<set>/ground_truth.jsonl` (see the dataset card,
"Synthesizer output"):

    {"file": {"kind": "eml", "path": "email/<id>.eml", "container": null},
     "spans": [{"label": "NAME_GIVEN", "text": "Megan", "new_text": "Alicia",
                "location": {"kind": "eml", "file_char_start": 553, "file_char_end": 558, ...}}]}

Each gold span is matched to the predicted span of the same file unit that overlaps it most in native
coordinates (>= 0.5 of the union) and carries its label; the matched replacement is then scored by the
same recall, LLM-judge and metric code as the message-level evaluator, so the three headline numbers
have the same definitions:

    NER recall          = detected / gold
    synthesis accuracy  = coherent / detected      (identity replacements count as misses)
    combined accuracy   = coherent / gold          (= their product)

reported overall, per entity label, per file kind (eml, slack, pdf, docx, xlsx, csv), per character and
per organization group. Every gold span in the file counts whether or not the pipeline produced output
for that file; the only exclusion is the handful of PDF spans the renderer clipped (`not_rendered`).

    python -m synthesis_evaluation.run_eval_native \\
        --predictions  my_pipeline/aaron_pfizer/predictions.jsonl \\
        --ground-truth $DATA/ground_truth/aaron_pfizer/ground_truth.jsonl \\
        --characters   $DATA/ground_truth/aaron_pfizer/characters.json \\
        --run-name     aaron_pfizer_my_pipeline [--judge-model claude-opus-5] [--skip-llm-judge]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from . import native, score_consistency, score_grouping, score_realism_llm, score_realism_rule, score_recall
from .load import load_characters, load_jsonl
from .metrics import compute_metrics
from .render import render
from .types import LABELS

RUNS_DIR = Path(__file__).resolve().parent / "runs"


def _p(x: Optional[float]) -> str:
    return f"{x:.3f}" if x is not None else "n/a"


def by_kind_metrics(rows, kinds: List[str], realism_llm: dict) -> Dict[str, dict]:
    """The headline block per file kind. The judge's verdicts are keyed by owner/label/surface, not by
    row, so scoring a subset of rows against the same verdicts is exact."""
    out: Dict[str, dict] = {}
    for kind in native.KINDS:
        sub = [r for r, k in zip(rows, kinds) if k == kind]
        if not sub:
            continue
        m = compute_metrics(sub, realism_llm)
        out[kind] = {k: v for k, v in m["overall"].items()}
    msgs = [r for r, k in zip(rows, kinds) if k in ("eml", "slack")]
    docs = [r for r, k in zip(rows, kinds) if k not in ("eml", "slack")]
    if msgs:
        out["messages"] = dict(compute_metrics(msgs, realism_llm)["overall"])
    if docs:
        out["documents"] = dict(compute_metrics(docs, realism_llm)["overall"])
    return out


def render_by_kind(by_kind: Dict[str, dict], diag: dict) -> str:
    lines = ["", "## By file kind", "",
             "Every gold span of every file counts; a file the pipeline never emitted contributes misses. "
             f"Spans excluded as `not_rendered`: {sum(diag.get('excluded_not_rendered', {}).values())}.", "",
             "| kind | gold | detected | NER recall | synthesis accuracy | combined | predicted spans |", "|---|---|---|---|---|---|---|"]
    for kind in list(native.KINDS) + ["messages", "documents"]:
        b = by_kind.get(kind)
        if not b:
            continue
        pred = diag.get("predicted", {}).get(kind, "") if kind in native.KINDS else ""
        lines.append(f"| {kind} | {b['gold']:,} | {b['detected']:,} | {_p(b['ner_recall'])} | {_p(b['synthesis_accuracy'])} | "
                     f"{_p(b['combined_accuracy'])} | {pred} |")
    other = diag.get("matched_other_label", {})
    if other:
        lines += ["", f"Gold spans overlapped only by a prediction with a different label (counted as NER misses): "
                  + ", ".join(f"{k} {v}" for k, v in sorted(other.items()))]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True, type=Path, help="pipeline output JSONL: {file, spans[...]} per file unit")
    ap.add_argument("--ground-truth", required=True, type=Path, help="ground_truth/<set>/ground_truth.jsonl")
    ap.add_argument("--characters", type=Path, default=None, help="ground_truth/<set>/characters.json (needed for the judge)")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--judge-model", default=None, help=f"LLM judge model id (default {score_realism_llm.MODEL})")
    ap.add_argument("--skip-llm-judge", action="store_true", help="NER recall only (offline, no API key)")
    ap.add_argument("--overlap", type=float, default=native.MATCH_THRESHOLD, help="minimum native-coordinate overlap for a match")
    args = ap.parse_args()
    if args.judge_model:
        score_realism_llm.MODEL = args.judge_model
    for p in (args.predictions, args.ground_truth):
        if not p.is_file():
            sys.exit(f"missing file: {p}")
    if not args.skip_llm_judge and not args.characters:
        sys.exit("--characters is required for synthesis accuracy; pass --skip-llm-judge for recall only")
    out_dir = args.runs_dir / args.run_name
    if out_dir.exists():
        sys.exit(f"refusing to overwrite existing run dir: {out_dir}")
    out_dir.mkdir(parents=True)

    characters = load_characters(args.characters) if args.characters else None
    print(f"loading predictions from {args.predictions} ...")
    preds = native.load_predictions(load_jsonl(args.predictions))
    print(f"  {sum(len(v) for v in preds.values()):,} in-scope predicted spans in {len(preds):,} file units")
    print(f"loading ground truth from {args.ground_truth} ...")
    gt_rows = list(load_jsonl(args.ground_truth))
    rows, kinds, diag = native.join(gt_rows, preds, threshold=args.overlap)
    print(f"  {len(rows):,} ground-truth rows; gold spans per kind {diag['gold']}; matched {diag['matched']}")

    timings: Dict[str, float] = {}
    t = time.monotonic()
    recall = score_recall.score(rows)
    timings["recall"] = time.monotonic() - t
    print(f"recall: {_p(recall['overall']['recall'])}  tp={recall['overall']['tp']}  fn={recall['overall']['fn']}")
    t = time.monotonic()
    consistency = score_consistency.score(rows)
    timings["consistency"] = time.monotonic() - t
    t = time.monotonic()
    realism_rule = score_realism_rule.score(rows)
    timings["realism_rule"] = time.monotonic() - t
    if args.skip_llm_judge:
        realism_llm = {"skipped_reason": "--skip-llm-judge", "per_character": {}, "overall": {"coherent": 0, "incoherent": 0, "skipped": 0}}
    else:
        print(f"LLM judge ({score_realism_llm.MODEL}) ...")
        t = time.monotonic()
        realism_llm = score_realism_llm.score(rows, characters=characters, workers=args.workers)
        timings["realism_llm"] = time.monotonic() - t
    t = time.monotonic()
    grouping = score_grouping.score(rows)
    timings["grouping"] = time.monotonic() - t

    metrics = compute_metrics(rows, realism_llm)
    metrics["by_kind"] = by_kind_metrics(rows, kinds, realism_llm)
    o = metrics["overall"]
    print(f"\nscores — ner_recall={_p(o['ner_recall'])}  synthesis_accuracy={_p(o['synthesis_accuracy'])}  combined={_p(o['combined_accuracy'])}")
    for kind, b in metrics["by_kind"].items():
        print(f"  {kind:10s} recall={_p(b['ner_recall'])}  synthesis={_p(b['synthesis_accuracy'])}  combined={_p(b['combined_accuracy'])}  (gold {b['gold']:,})")

    results = {
        "config": {"predictions": str(args.predictions), "ground_truth": str(args.ground_truth),
                   "characters": str(args.characters) if args.characters else None, "run_name": args.run_name,
                   "format": "native", "overlap_threshold": args.overlap, "judge_model": None if args.skip_llm_judge else score_realism_llm.MODEL,
                   "n_rows": len(rows), "tier": "entity", "labels": list(LABELS)},
        "timings_sec": timings,
        "metrics": metrics,
        "native_join": diag,
        "detail": {"recall": recall, "consistency": consistency, "realism_rule": realism_rule, "judge": realism_llm, "grouping": grouping},
    }
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    render(results, out_dir / "summary.md", out_dir / "viewer.html")
    with open(out_dir / "summary.md", "a", encoding="utf-8") as f:
        f.write(render_by_kind(metrics["by_kind"], diag))
    print(f"\nwrote {out_dir / 'results.json'}\nwrote {out_dir / 'summary.md'}\nwrote {out_dir / 'viewer.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""CLI orchestrator for the PrivacyBench synthesis evaluation.

Scores a synthesizer's **output file** against a **ground-truth file**.
The output file is minimal — one JSON object per message carrying only
the detected PII entities and their synthetic replacements — and is
joined to the ground truth by ``row_id``. Everything the scorers need
(the original text, the gold PII spans, and the predicted entities) is
assembled here; no per-pipeline adapter is required.

Output-file schema (one JSON object per line)::

    {
      "row_id": "<id matching meta.row_id in the ground-truth file>",
      "entities": [
        {"start": 46, "end": 58, "label": "USERNAME",
         "text": "<@U02CARLOS>", "new_text": "<@U02MIGUEL_SANTOS>"},
        ...
      ]
    }

``label`` is one of NAME_GIVEN, NAME_FAMILY, EMAIL_ADDRESS, USERNAME,
ORGANIZATION. ``start``/``end`` are character offsets into the original
message text (from the ground-truth file). ``text`` is the original PII
surface and ``new_text`` its synthetic replacement; an entity counts as
"detected and replaced" only when ``new_text`` differs from ``text``.

Writes ``results.json``, ``summary.md``, and ``viewer.html`` into the
run directory.

Usage
-----

    python -m synthesis_evaluation.run_eval \\
        --predictions  example_output/megan_donovan_eli_lilly_xml_haiku_output.jsonl \\
        --ground-truth ground_truth/megan_donovan_eli_lilly_ground_truth_spans.jsonl \\
        --characters   ground_truth/megan_donovan_eli_lilly_characters_ground_truth.json \\
        --run-name     megan_donovan_eli_lilly_xml_haiku

The eval reports three metrics: NER recall, synthesis accuracy, and
synthesis + NER accuracy. The latter two come from an LLM judge that
scores each synthetic replacement against the character's PII (from the
``--characters`` roster), so ``--characters`` is required unless you
pass ``--skip-llm-judge`` to compute NER recall offline.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from . import score_recall, score_realism_llm
from .load import load_characters, load_jsonl
from .render import render
from .types import LABELS, TIER_ENTITY, EvalRow

RUNS_DIR = Path(__file__).resolve().parent / "runs"


def _load_predictions(path: Path) -> Dict[str, List[dict]]:
    """``{row_id: [entity, ...]}`` from the output file. Entities are
    filtered to the in-scope labels and normalized to the eval shape."""
    preds: Dict[str, List[dict]] = {}
    for row in load_jsonl(path):
        rid = row.get("row_id")
        if rid is None:
            raise SystemExit(
                "every prediction row must carry a 'row_id' that matches "
                "meta.row_id in the ground-truth file")
        ents: List[dict] = []
        for e in (row.get("entities") or []):
            if e.get("label") not in LABELS:
                continue
            ents.append({
                "start":    e["start"],
                "end":      e["end"],
                "label":    e["label"],
                "text":     e["text"],
                "new_text": e["new_text"],
                "group_id": e.get("group_id"),
                "score":    e.get("score"),
            })
        preds[rid] = ents
    return preds


def _reconstruct(text: str, entities: List[dict]) -> str:
    """Splice replacements into the original text (right-to-left so
    offsets stay valid). Used only for the human-facing report/viewer;
    the numeric scores depend on the entities, not this string."""
    for e in sorted(entities, key=lambda e: e["start"], reverse=True):
        text = text[:e["start"]] + (e.get("new_text") or e["text"]) + text[e["end"]:]
    return text


def _build_rows(predictions: Dict[str, List[dict]],
                ground_truth_path: Path) -> List[EvalRow]:
    """Join predictions to ground-truth rows by row_id and build EvalRows."""
    rows: List[EvalRow] = []
    matched = 0
    for gt in load_jsonl(ground_truth_path):
        rid = (gt.get("meta") or {}).get("row_id")
        ents = predictions.get(rid, [])
        if rid in predictions:
            matched += 1
        text = gt.get("text") or ""
        rows.append(EvalRow.from_dict({
            "meta": gt.get("meta") or {},
            "text": text,
            "ground_truth_spans": gt.get("ground_truth_spans") or [],
            "synthesis": {
                "synthetic_text": _reconstruct(text, ents),
                "entities":       ents,
                "tier":           TIER_ENTITY,
            },
        }))
    unmatched_preds = set(predictions) - {
        (r.meta or {}).get("row_id") for r in rows}
    print(f"  joined {matched}/{len(rows)} ground-truth rows to predictions"
          + (f"  ({len(unmatched_preds)} prediction row_ids had no "
             f"matching ground-truth row)" if unmatched_preds else ""))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True, type=Path,
                    help="Synthesizer output JSONL (row_id + entities per line).")
    ap.add_argument("--ground-truth", required=True, type=Path,
                    help="Ground-truth JSONL (meta + text + ground_truth_spans).")
    ap.add_argument("--characters", type=Path, default=None,
                    help="Character roster JSON (<set>_characters_ground_truth.json). "
                         "Required for synthesis accuracy — the LLM judge uses each "
                         "character's PII to score the synthetic replacements. "
                         "Only optional with --skip-llm-judge.")
    ap.add_argument("--run-name", required=True,
                    help="Output dir name under the runs dir.")
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR,
                    help="Directory to write the run dir into (default: "
                         "synthesis_evaluation/runs/).")
    ap.add_argument("--workers", type=int, default=8,
                    help="Workers for the LLM judge (default 8).")
    ap.add_argument("--skip-llm-judge", action="store_true",
                    help="Skip the LLM-as-judge scorer (offline; NER recall only).")
    ap.add_argument("--limit-rows", type=int, default=None,
                    help="Truncate to N rows (debug only).")
    args = ap.parse_args()

    if not args.predictions.is_file():
        sys.exit(f"missing predictions: {args.predictions}")
    if not args.ground_truth.is_file():
        sys.exit(f"missing ground truth: {args.ground_truth}")

    if not args.skip_llm_judge and not args.characters:
        sys.exit("--characters is required for synthesis accuracy (the LLM "
                 "judge scores replacements against each character's PII). "
                 "Pass the dataset's <set>_characters_ground_truth.json, or "
                 "use --skip-llm-judge to compute NER recall only.")

    out_dir = args.runs_dir / args.run_name
    if out_dir.exists():
        sys.exit(f"refusing to overwrite existing run dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)

    characters: Optional[dict] = None
    if args.characters:
        if not args.characters.is_file():
            sys.exit(f"missing characters: {args.characters}")
        print(f"loading characters from {args.characters} ...")
        characters = load_characters(args.characters)
        print(f"  {len(characters)} characters loaded")

    print(f"loading predictions from {args.predictions} ...")
    predictions = _load_predictions(args.predictions)
    print(f"  {len(predictions)} prediction rows")

    print(f"loading ground truth from {args.ground_truth} ...")
    rows = _build_rows(predictions, args.ground_truth)
    if args.limit_rows is not None:
        rows = rows[: args.limit_rows]
    print(f"  {len(rows)} rows built")

    tier = rows[0].synthesis.tier if rows else None
    timings = {}

    print("\nscoring recall ...")
    t = time.monotonic()
    recall = score_recall.score(rows)
    timings["recall"] = time.monotonic() - t
    r = recall["overall"]["recall"]
    r_str = f"{r:.3f}" if r is not None else "n/a"
    print(f"  done in {timings['recall']:.1f}s — overall recall={r_str}  "
          f"tp={recall['overall']['tp']}  fn={recall['overall']['fn']}")

    if args.skip_llm_judge:
        print("\nskipping LLM judge (--skip-llm-judge)")
        realism_llm = {
            "skipped_reason": "--skip-llm-judge",
            "per_character": {},
            "overall": {"coherent": 0, "incoherent": 0, "skipped": 0},
        }
    else:
        print("\nscoring realism (LLM judge) ...")
        t = time.monotonic()
        realism_llm = score_realism_llm.score(
            rows, characters=characters, workers=args.workers)
        timings["realism_llm"] = time.monotonic() - t
        plt = realism_llm.get("per_label_totals") or {}
        bits = [f"{lab}: {plt.get(lab, {}).get('coherent', 0)}/"
                f"{plt.get(lab, {}).get('coherent', 0) + plt.get(lab, {}).get('incoherent', 0)}"
                f" coherent"
                for lab in ("NAME_GIVEN", "NAME_FAMILY", "EMAIL_ADDRESS", "USERNAME")]
        ogt = realism_llm.get("org_group_totals") or {}
        if ogt:
            bits.append(f"ORG surfaces: {ogt.get('coherent', 0)}/"
                        f"{ogt.get('coherent', 0) + ogt.get('incoherent', 0)} coherent")
        print(f"  done in {timings.get('realism_llm', 0):.1f}s — " + "; ".join(bits))

    results = {
        "config": {
            "predictions": str(args.predictions),
            "ground_truth": str(args.ground_truth),
            "characters": str(args.characters) if args.characters else None,
            "run_name": args.run_name,
            "n_rows": len(rows),
            "tier": tier,
        },
        "timings_sec": timings,
        "metrics": {
            "recall":      recall,
            "realism_llm": realism_llm,
        },
    }

    (out_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2))
    render(results, out_dir / "summary.md", out_dir / "viewer.html")
    print(f"\nwrote {out_dir / 'results.json'}")
    print(f"wrote {out_dir / 'summary.md'}")
    print(f"wrote {out_dir / 'viewer.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

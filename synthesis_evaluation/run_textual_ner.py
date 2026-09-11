"""Run Tonic Textual NER over the dataset's messages to produce predictions
that run_ner_eval can score — reproduces the dataset card's Textual row.

Reads one or more JSONL files whose rows carry a row id (`row_id`,
`meta.row_id`, or `meta.cell_id`) and a `text` field — the dataset's
`human_annotations/<set>.jsonl` files work directly — and writes, per input
file, a predictions file of the same basename with rows
`{"row_id": ..., "spans": [{label, start, end, text, score}]}`.

Textual is called with a USERNAME allow-list regex so Slack mentions like
`<@U02CARLOS>` are detected as single bracket-inclusive spans (the published
scores are produced this way), detections are kept at score >= 0.5 and
filtered to the benchmark's five labels. The server version is printed at
the start — the dataset card states which version its scores came from.

Requires `pip install tonic-textual` and `TONIC_TEXTUAL_API_KEY`.

Usage:
  python -m synthesis_evaluation.run_textual_ner \
    --input "$DATA/human_annotations/*.jsonl" \
    --out textual_predictions

then score with:
  python -m synthesis_evaluation.run_ner_eval \
    --gold "$DATA/human_annotations/*.jsonl" \
    --predictions-dir textual_predictions \
    --run-name textual_human_gold
"""
from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional

from .load import load_jsonl
from .types import LABELS

BASE_URL = "https://textual.tonic.ai"
USERNAME_ALLOW_LIST = {"USERNAME": [r"<@[^>|]+(?:\|[^>]*)?>"]}
MIN_SCORE = 0.5
BATCH_SIZE = 100
MAX_WORKERS = 4


def _row_id(row: dict) -> str:
    rid = row.get("row_id")
    if rid is None:
        meta = row.get("meta") or {}
        rid = meta.get("row_id") or meta.get("cell_id")
    if rid is None:
        raise KeyError(f"row has no row_id (top-level, meta.row_id, or meta.cell_id): {list(row)}")
    return str(rid)


def predict(texts: List[str]) -> List[List[dict]]:
    """One list of `{label, start, end, text, score}` spans per input text."""
    from tonic_textual.redact_api import TextualNer

    client = TextualNer()
    batches = [list(range(i, min(i + BATCH_SIZE, len(texts))))
               for i in range(0, len(texts), BATCH_SIZE)]
    results: List[Optional[List[dict]]] = [None] * len(texts)

    def _process(batch_idxs: List[int]) -> None:
        resp = client.redact_bulk([texts[i] for i in batch_idxs],
                                  label_allow_lists=USERNAME_ALLOW_LIST)
        for idx, dets in zip(batch_idxs, resp.de_identify_results):
            spans = []
            for det in dets or []:
                score = float(getattr(det, "score", 1.0) or 1.0)
                if score < MIN_SCORE or det.label not in LABELS:
                    continue
                spans.append({"label": det.label, "start": int(det.start),
                              "end": int(det.end), "text": det.text, "score": score})
            spans.sort(key=lambda s: (s["start"], s["end"], s["label"]))
            results[idx] = spans

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        list(exe.map(_process, batches))
    return [r if r is not None else [] for r in results]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", action="append", required=True, metavar="FILE_OR_GLOB",
                        help="JSONL with a row id and a text field per row "
                             "(the dataset's human_annotations/<set>.jsonl files work directly)")
    parser.add_argument("--out", type=Path, required=True,
                        help="directory for the predictions files (one per input, same basename)")
    args = parser.parse_args()

    if not os.environ.get("TONIC_TEXTUAL_API_KEY"):
        sys.exit("TONIC_TEXTUAL_API_KEY not set")

    inputs: List[Path] = []
    for pattern in args.input:
        hits = sorted(glob.glob(pattern))
        if not hits:
            sys.exit(f"--input matches nothing: {pattern}")
        inputs.extend(Path(h) for h in hits)

    with urllib.request.urlopen(f"{BASE_URL}/api/version", timeout=30) as resp:
        print(f"Tonic Textual server version: {resp.read().decode().strip()}")

    args.out.mkdir(parents=True, exist_ok=True)
    for path in inputs:
        rows = list(load_jsonl(path))
        spans = predict([r.get("text") or "" for r in rows])
        dest = args.out / path.name
        with dest.open("w", encoding="utf-8") as fh:
            for row, row_spans in zip(rows, spans):
                fh.write(json.dumps({"row_id": _row_id(row), "spans": row_spans},
                                    ensure_ascii=False) + "\n")
        print(f"{path.stem}: {len(rows)} rows -> {dest} ({sum(len(s) for s in spans)} spans)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

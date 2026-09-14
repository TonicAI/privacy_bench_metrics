"""Run Tonic Textual NER over the dataset's human-annotated gold texts to
produce predictions that run_ner_eval can score — reproduces the dataset
card's Textual rows for both the messages and the documents.

Reads one or more JSONL files whose rows carry a row id (`row_id`,
`meta.row_id`, or `meta.cell_id`, plus `meta.page` for document pages) and a
`text` field — the dataset's `human_annotations/<set>.jsonl` and
`human_annotations/<set>_documents.jsonl` files work directly — and writes,
per input file, a predictions file of the same basename with rows
`{"row_id": ..., "spans": [{label, start, end, text, score}]}`.

Two modes:

* Default (no --config): the card's **message** rows — plain `redact_bulk`
  with a USERNAME allow-list regex so Slack mentions like `<@U02CARLOS>`
  come back as single bracket-inclusive spans.
* `--config synthesis_evaluation/textual_config.json`: the card's
  **document** rows ("Textual (SDK, graph config)") — the graph pipeline's
  configuration expressed in SDK terms: its ACCOUNT_NUMBER /
  LOCATION_ADDRESS allow lists, EMAIL_ADDRESS block list, and username
  regexes are applied server side; its employee-id regexes client side
  (EMPLOYEE_ID is not a Textual label, so a server-side allow list under
  that name is ignored); US_BANK_NUMBER folds to ACCOUNT_NUMBER and
  LOCATION_COMPLETE_ADDRESS to LOCATION_ADDRESS. Needs `pip install regex`
  (the config uses variable-width look-behinds).

Label scope follows the card's convention automatically: message files are
filtered to the five original labels, document-page files (rows with
`meta.page`) keep all detections the config emits — the scorer's label
aliases handle engine vocabulary such as NUMERIC_PII.

Detections below --min-score are dropped (default 0.5, the published
threshold); allow-list matches come back with score 1. The Textual server
version is printed at the start — the card states which version its scores
came from. Requires `pip install tonic-textual` and `TONIC_TEXTUAL_API_KEY`.

Usage (messages — document-page files in the glob are skipped without --config):
  python -m synthesis_evaluation.run_textual_ner \
    --input "$DATA/human_annotations/*.jsonl" --out textual_msgs
Usage (documents):
  python -m synthesis_evaluation.run_textual_ner \
    --input "$DATA/human_annotations/*_documents.jsonl" \
    --config synthesis_evaluation/textual_config.json --out textual_docs

then score with run_ner_eval (--gold the same files, --predictions-dir the
output folder).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

from .load import load_jsonl
from .score_ner import _row_id
from .types import ORIGINAL_LABELS

BASE_URL = "https://textual.tonic.ai"
USERNAME_ALLOW_LIST = {"USERNAME": [r"<@[^>|]+(?:\|[^>]*)?>"]}
BATCH_SIZE = 50
MAX_WORKERS = 4
LABEL_FOLDS = {"CUSTOM_USERNAME": "USERNAME", "CUSTOM_EMPLOYEE_ID": "EMPLOYEE_ID",
               "US_BANK_NUMBER": "ACCOUNT_NUMBER", "LOCATION_COMPLETE_ADDRESS": "LOCATION_ADDRESS"}


def _bmp_safe(text: str) -> str:
    """Textual's API rejects astral-plane characters; replace, keeping offsets."""
    return "".join(ch if ord(ch) <= 0xFFFF else "�" for ch in text)


def _dedupe(spans: List[dict]) -> List[dict]:
    """Drop exact duplicates and spans fully inside a longer same-label span."""
    spans = sorted({(s["start"], s["end"], s["label"]): s for s in spans}.values(),
                   key=lambda s: (s["start"], -(s["end"] - s["start"])))
    out: List[dict] = []
    for s in spans:
        if any(o["label"] == s["label"] and o["start"] <= s["start"] and s["end"] <= o["end"]
               and (o["start"], o["end"]) != (s["start"], s["end"]) for o in out):
            continue
        out.append(s)
    return out


class TextualRunner:
    def __init__(self, config: Optional[dict], min_score: float):
        from tonic_textual.redact_api import TextualNer

        self.client = TextualNer()
        self.min_score = min_score
        if config is None:
            self.allow = USERNAME_ALLOW_LIST
            self.block = None
            self.keep = None          # per-file scope decides
            self.employee_id = []
        else:
            import regex              # variable-width look-behinds in the config
            self.allow = {"USERNAME": config["custom_entities"]["username"]["regexes"],
                          "ACCOUNT_NUMBER": config["allow_lists"]["ACCOUNT_NUMBER"]["regexes"],
                          "LOCATION_ADDRESS": config["allow_lists"]["LOCATION_ADDRESS"]["regexes"]}
            self.block = {"EMAIL_ADDRESS": config["block_lists"]["EMAIL_ADDRESS"]["regexes"]}
            self.keep = (set(config["synthesize_labels"]) - {"PERSON"}
                         | {"EMPLOYEE_ID", "ACCOUNT_NUMBER", "LOCATION_ADDRESS", "USERNAME"})
            self.employee_id = [regex.compile(r) for r in config["custom_entities"]["employee_id"]["regexes"]]

    def predict(self, texts: List[str], five_labels_only: bool) -> List[List[dict]]:
        results: List[List[dict]] = [[] for _ in texts]
        batches = [list(range(i, min(i + BATCH_SIZE, len(texts))))
                   for i in range(0, len(texts), BATCH_SIZE)]

        def one(ids: List[int]) -> None:
            strings = [_bmp_safe(texts[i]) for i in ids]
            kwargs = {"label_allow_lists": self.allow}
            if self.block:
                kwargs["label_block_lists"] = self.block
            for attempt in range(1, 5):
                try:
                    resp = self.client.redact_bulk(strings, **kwargs)
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt == 4:
                        raise
                    print(f"  batch retry {attempt}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
                    time.sleep(5 * attempt)
            for i, dets in zip(ids, resp.de_identify_results):
                spans = []
                for det in dets or []:
                    label = LABEL_FOLDS.get(det.label, det.label)
                    score = float(getattr(det, "score", 1.0) or 1.0)
                    if score < self.min_score:
                        continue
                    if self.keep is not None and label not in self.keep:
                        continue
                    if self.keep is None and label not in ORIGINAL_LABELS:
                        continue
                    spans.append({"label": label, "start": int(det.start), "end": int(det.end),
                                  "text": det.text, "score": round(score, 3)})
                for pat in self.employee_id:
                    for m in pat.finditer(texts[i]):
                        if m.end() > m.start():
                            spans.append({"label": "EMPLOYEE_ID", "start": m.start(), "end": m.end(),
                                          "text": m.group(0), "score": 1.0})
                if five_labels_only:
                    spans = [s for s in spans if s["label"] in ORIGINAL_LABELS]
                spans.sort(key=lambda s: (s["start"], s["end"], s["label"]))
                # dedupe only in config mode (parity with each mode's published run)
                results[i] = _dedupe(spans) if self.keep is not None else spans

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
            list(exe.map(one, batches))
        return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", action="append", required=True, metavar="FILE_OR_GLOB",
                        help="JSONL with a row id and a text field per row (the dataset's "
                             "human_annotations files work directly)")
    parser.add_argument("--out", type=Path, required=True,
                        help="directory for the predictions files (one per input, same basename)")
    parser.add_argument("--config", type=Path,
                        help="textual_config.json: run with the graph pipeline's allow/block lists "
                             "(the card's document rows); default is the plain message mode")
    parser.add_argument("--min-score", type=float, default=0.5)
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

    config = json.loads(args.config.read_text()) if args.config else None
    runner = TextualRunner(config, args.min_score)
    args.out.mkdir(parents=True, exist_ok=True)
    for path in inputs:
        rows = list(load_jsonl(path))
        # message files score on the five original labels; document pages on all ten
        is_pages = bool(rows) and (rows[0].get("meta") or {}).get("page") is not None
        if is_pages and config is None:
            print(f"{path.stem}: document pages — skipped (pass --config to reproduce the document rows)")
            continue
        spans = runner.predict([r.get("text") or "" for r in rows], five_labels_only=not is_pages)
        dest = args.out / path.name
        with dest.open("w", encoding="utf-8") as fh:
            for row, row_spans in zip(rows, spans):
                fh.write(json.dumps({"row_id": _row_id(row), "spans": row_spans},
                                    ensure_ascii=False) + "\n")
        print(f"{path.stem}: {len(rows)} rows ({'pages' if is_pages else 'messages'}) "
              f"-> {dest} ({sum(len(s) for s in spans)} spans)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

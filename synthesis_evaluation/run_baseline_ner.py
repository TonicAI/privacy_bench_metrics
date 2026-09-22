"""Presidio / GLiNER2 baselines over the human-annotation files -> `run_ner_eval` prediction files.

Reproduces the dataset card's Presidio and GLiNER2 rows. Reads the published human annotations
(`human_annotations/<set>.jsonl` messages, `human_annotations/<set>_documents.jsonl` document pages),
runs one of the two open-source NER engines over the row texts and writes one prediction file per
input file (same file name) with rows `{"row_id": <id>, "spans": [{label, start, end, text, score}]}`;
page rows are keyed `<document row_id>#p<page>` like the gold. Score with:

    python -m synthesis_evaluation.run_ner_eval --gold "$DATA/human_annotations/*.jsonl" \\
        --predictions-dir presidio_predictions --run-name presidio_human_gold

Message files are scored on the five original labels, so the engines are asked only for person /
organization / email / username there; document pages carry all ten labels, so the engines are
additionally asked for phone numbers, addresses, ids / account numbers and URLs (the extra labels
were added for the document rows on 2026-09-14; the message configuration is the original 2026-07
message pipeline). Engine labels outside the benchmark vocabulary (Presidio's LOCATION, US_BANK_NUMBER,
CREDIT_CARD, IBAN_CODE, ID_NUMBER) are emitted as-is; `types.LABEL_ALIASES` folds them onto the
benchmark labels at scoring time (ID_NUMBER counts as a detection of either EMPLOYEE_ID or
ACCOUNT_NUMBER).

Presidio
    spaCy `en_core_web_trf` through presidio's NlpEngineProvider (ORG kept at spaCy's 0.85 instead of
    presidio's down-weighting), presidio's built-in EMAIL_ADDRESS (and, for pages, PHONE_NUMBER at its
    own 0.4 recogniser score, URL, US_BANK_NUMBER, CREDIT_CARD, IBAN_CODE), a USERNAME pattern
    recogniser for Slack mentions `<@U...>` and `@handles`, and for pages a generic prefix-dash-digits
    id pattern (`[A-Z]{2,5}-\\d{4,9}`, years excluded) emitted as ID_NUMBER. Person spans are split into a
    NAME_GIVEN first token and a NAME_FAMILY last token (middle tokens dropped, as the benchmark tags
    no middle names); overlapping spans are dropped positionally.
        pip install presidio-analyzer spacy && python -m spacy download en_core_web_trf

GLiNER2
    `fastino/gliner2-privacy-filter-PII-multi` (schema-conditioned; labels first_name / last_name /
    email / username / organization, plus phone_number / address / employee_id / account_number / url
    for pages), threshold 0.3, texts windowed at 350 words with 50 words of overlap. Same name split
    and overlap policy. gliner2 needs a recent `transformers` that conflicts with `spacy-transformers`,
    so use a separate virtualenv for it:
        pip install gliner2

Usage
    python -m synthesis_evaluation.run_baseline_ner --engine presidio \\
        --input "$DATA/human_annotations/*.jsonl" --out presidio_predictions
    python -m synthesis_evaluation.run_baseline_ner --engine gliner \\
        --input "$DATA/human_annotations/*.jsonl" --out gliner_predictions
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .score_ner import _row_id

MESSAGE_LABELS = {"NAME_GIVEN", "NAME_FAMILY", "EMAIL_ADDRESS", "USERNAME", "ORGANIZATION"}
_TOKEN_RE = re.compile(r"\S+")


# ---------------------------------------------------------------------------
# shared post-processing (identical to the benchmark's other NER pipelines)
# ---------------------------------------------------------------------------

def split_name_span(text: str, start: int, score: float, *, single_label: str = "NAME_GIVEN") -> List[dict]:
    """A person-name span -> NAME_GIVEN on the first token + NAME_FAMILY on the last (middle tokens
    dropped); a single token keeps `single_label`. Tokens are trimmed of punctuation and possessives."""
    toks: List[dict] = []
    for m in _TOKEN_RE.finditer(text):
        tok, s = m.group(), start + m.start()
        while tok and not tok[0].isalnum():
            tok, s = tok[1:], s + 1
        while tok and not tok[-1].isalnum():
            tok = tok[:-1]
        if tok.lower().endswith(("'s", "’s")):
            tok = tok[:-2]
        if tok:
            toks.append({"text": tok, "start": s, "end": s + len(tok)})
    if not toks:
        return []
    if len(toks) == 1:
        return [{"label": single_label, **toks[0], "score": float(score)}]
    return [{"label": "NAME_GIVEN", **toks[0], "score": float(score)},
            {"label": "NAME_FAMILY", **toks[-1], "score": float(score)}]


def drop_overlaps(spans: List[dict]) -> List[dict]:
    """Non-overlapping subset: earlier start wins; ties -> longer span, then higher score."""
    out: List[dict] = []
    last_end = -1
    for s in sorted(spans, key=lambda s: (s["start"], -(s["end"] - s["start"]), -(s.get("score") or 0.0))):
        if s["start"] < last_end:
            continue
        out.append(s)
        last_end = s["end"]
    return out


# ---------------------------------------------------------------------------
# Presidio
# ---------------------------------------------------------------------------

PRESIDIO_MESSAGE_ENTITIES = ["PERSON", "ORGANIZATION", "EMAIL_ADDRESS", "USERNAME"]
PRESIDIO_PAGE_ENTITIES = PRESIDIO_MESSAGE_ENTITIES + ["PHONE_NUMBER", "LOCATION", "URL", "US_BANK_NUMBER", "CREDIT_CARD", "IBAN_CODE", "ID_NUMBER"]
_SPACY_IGNORE = ["CARDINAL", "DATE", "EVENT", "FAC", "LANGUAGE", "LAW", "MONEY", "NORP", "ORDINAL", "PERCENT", "PRODUCT", "QUANTITY", "TIME", "WORK_OF_ART"]


class PresidioRunner:
    def __init__(self, spacy_model: str = "en_core_web_trf", min_score: float = 0.5):
        self.spacy_model, self.min_score = spacy_model, min_score
        self._analyzers: Dict[bool, object] = {}

    def _analyzer(self, pages: bool):
        if pages in self._analyzers:
            return self._analyzers[pages]
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        mapping = {"PER": "PERSON", "PERSON": "PERSON", "ORG": "ORGANIZATION", "ORGANIZATION": "ORGANIZATION"}
        ignore = list(_SPACY_IGNORE)
        if pages:
            mapping.update({"GPE": "LOCATION", "LOC": "LOCATION"})
        else:
            ignore += ["GPE", "LOC"]
        conf = {"nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": self.spacy_model}],
                "ner_model_configuration": {"model_to_presidio_entity_mapping": mapping, "low_score_entity_names": [], "labels_to_ignore": ignore}}
        engine = NlpEngineProvider(nlp_configuration=conf).create_engine()
        analyzer = AnalyzerEngine(nlp_engine=engine, supported_languages=["en"])
        analyzer.registry.add_recognizer(PatternRecognizer(
            supported_entity="USERNAME", name="slack_mention_recognizer",
            patterns=[Pattern(name="slack_mention", regex=r"<@[A-Za-z0-9._-]+>", score=0.9),
                      Pattern(name="at_handle", regex=r"(?<![\w@.])@[A-Za-z][A-Za-z0-9._-]*[A-Za-z0-9]", score=0.6)]))
        if pages:
            analyzer.registry.add_recognizer(PatternRecognizer(
                supported_entity="ID_NUMBER", name="generic_id_recognizer",
                patterns=[Pattern(name="prefix_dash_digits", regex=r"\b[A-Z]{2,5}-(?!(?:19|20)\d{2}\b)\d{4,9}\b", score=0.6)]))
        self._analyzers[pages] = analyzer
        return analyzer

    def predict(self, texts: List[str], *, pages: bool, log=print) -> List[List[dict]]:
        analyzer = self._analyzer(pages)
        entities = PRESIDIO_PAGE_ENTITIES if pages else PRESIDIO_MESSAGE_ENTITIES
        out: List[List[dict]] = []
        t0 = time.monotonic()
        for i, text in enumerate(texts):
            spans: List[dict] = []
            if text:
                for r in analyzer.analyze(text=text, language="en", entities=entities):
                    if r.score < (0.4 if r.entity_type == "PHONE_NUMBER" else self.min_score):
                        continue
                    surface = text[r.start:r.end]
                    if r.entity_type == "PERSON":
                        spans.extend(split_name_span(surface, r.start, r.score))
                    else:
                        spans.append({"label": r.entity_type, "start": r.start, "end": r.end, "text": surface, "score": float(r.score)})
            out.append(drop_overlaps(spans))
            if (i + 1) % 200 == 0:
                log(f"    presidio {i + 1}/{len(texts)} ({(i + 1) / (time.monotonic() - t0):.1f}/s)")
        return out


# ---------------------------------------------------------------------------
# GLiNER2
# ---------------------------------------------------------------------------

GLINER_MESSAGE_SCHEMA = {
    "first_name": "A person's given name, first name, or nickname (e.g. 'Megan', 'meg', 'Mike')",
    "last_name": "A person's family name / surname (e.g. 'Donovan', 'Tessler')",
    "email": "A full email address (e.g. 'megan.donovan@lilly.com')",
    "username": "A username, user mention, or handle, such as slack mentions like '<@U02MEGAN_DONOVAN>' or '@linda.cho'",
    "organization": "A company, organization, team, or brand name (e.g. 'Eli Lilly', 'CBRE', 'Novo Nordisk')",
}
GLINER_PAGE_SCHEMA = {**GLINER_MESSAGE_SCHEMA,
    "phone_number": "A telephone number in any format (e.g. '+1 (317) 555-0114', '317-555-0114')",
    "address": "A physical street or mailing address (e.g. '450 E 29th St, Indianapolis, IN 46205')",
    "employee_id": "An employee or personnel identifier code (e.g. 'AB-12345')",
    "account_number": "A bank, billing, or customer account number (e.g. 'ACCT-483920')",
    "url": "A web address or URL (e.g. 'https://example.com/page')",
}
GLINER_TO_BENCH = {"first_name": "NAME_GIVEN", "last_name": "NAME_FAMILY", "email": "EMAIL_ADDRESS", "username": "USERNAME",
                   "organization": "ORGANIZATION", "phone_number": "PHONE_NUMBER", "address": "LOCATION_ADDRESS",
                   "employee_id": "EMPLOYEE_ID", "account_number": "ACCOUNT_NUMBER", "url": "URL"}


def _windows(text: str, max_words: int, overlap: int) -> List[Tuple[int, str]]:
    toks = list(_TOKEN_RE.finditer(text))
    if len(toks) <= max_words:
        return [(0, text)]
    step = max_words - overlap
    out: List[Tuple[int, str]] = []
    for w in range(0, len(toks), step):
        lo = toks[w].start()
        hi = toks[min(w + max_words, len(toks)) - 1].end()
        out.append((lo, text[lo:hi]))
        if w + max_words >= len(toks):
            break
    return out


class GlinerRunner:
    def __init__(self, model_id: str = "fastino/gliner2-privacy-filter-PII-multi", threshold: float = 0.3, batch_size: int = 16,
                 window_words: int = 350, window_overlap_words: int = 50):
        self.model_id, self.threshold, self.batch_size = model_id, threshold, batch_size
        self.window_words, self.window_overlap_words = window_words, window_overlap_words
        self._model = None

    def predict(self, texts: List[str], *, pages: bool, log=print) -> List[List[dict]]:
        if self._model is None:
            from gliner2 import GLiNER2
            log(f"    loading {self.model_id} ...")
            self._model = GLiNER2.from_pretrained(self.model_id)
        schema = GLINER_PAGE_SCHEMA if pages else GLINER_MESSAGE_SCHEMA
        chunks: List[Tuple[int, int, str]] = []
        for i, text in enumerate(texts):
            for off, sub in _windows(text or "", self.window_words, self.window_overlap_words):
                if sub.strip():
                    chunks.append((i, off, sub))
        log(f"    gliner: {len(texts)} texts -> {len(chunks)} windows")
        results = self._model.batch_extract_entities([c[2] for c in chunks], schema, batch_size=self.batch_size,
                                                     threshold=self.threshold, include_confidence=True, include_spans=True)
        per_row: List[List[dict]] = [[] for _ in texts]
        for (i, off, _sub), res in zip(chunks, results):
            for glab, ents in ((res or {}).get("entities") or {}).items():
                bench = GLINER_TO_BENCH.get(glab)
                if bench is None:
                    continue
                for e in ents:
                    start, end = off + int(e["start"]), off + int(e["end"])
                    surface = (texts[i] or "")[start:end]
                    if surface != e["text"]:
                        continue
                    score = float(e.get("confidence", 1.0))
                    if bench in ("NAME_GIVEN", "NAME_FAMILY"):
                        per_row[i].extend(split_name_span(surface, start, score, single_label=bench))
                    else:
                        per_row[i].append({"label": bench, "start": start, "end": end, "text": surface, "score": score})
        out: List[List[dict]] = []
        for spans in per_row:
            best: Dict[Tuple[int, int], dict] = {}
            for s in spans:
                key = (s["start"], s["end"])
                if key not in best or s["score"] > best[key]["score"]:
                    best[key] = s
            out.append(drop_overlaps(list(best.values())))
        return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=("presidio", "gliner"), required=True)
    ap.add_argument("--input", action="append", required=True, metavar="FILE_OR_GLOB",
                    help="human-annotation jsonl files (messages and/or *_documents pages); repeatable")
    ap.add_argument("--out", type=Path, required=True, help="output directory: one prediction file per input file")
    ap.add_argument("--min-score", type=float, default=0.5, help="presidio: minimum recogniser score (phones keep presidio's 0.4)")
    ap.add_argument("--spacy-model", default="en_core_web_trf")
    ap.add_argument("--gliner-model", default="fastino/gliner2-privacy-filter-PII-multi")
    ap.add_argument("--threshold", type=float, default=0.3, help="gliner: extraction threshold")
    a = ap.parse_args()

    paths: List[Path] = []
    for pattern in a.input:
        hits = sorted(glob.glob(pattern))
        if not hits:
            print(f"warning: no files match {pattern}", file=sys.stderr)
        paths += [Path(h) for h in hits]
    if not paths:
        sys.exit("no input files")
    runner = PresidioRunner(a.spacy_model, a.min_score) if a.engine == "presidio" else GlinerRunner(a.gliner_model, a.threshold)
    a.out.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)   # noqa: E731
    meta = {"engine": a.engine, "min_score": a.min_score, "spacy_model": a.spacy_model, "gliner_model": a.gliner_model, "threshold": a.threshold, "files": {}}
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
        pages = bool(rows) and (rows[0].get("meta") or {}).get("page") is not None
        log(f"{path.stem}: {len(rows)} rows ({'pages, ten labels' if pages else 'messages, five labels'})")
        preds = runner.predict([r.get("text") or "" for r in rows], pages=pages, log=log)
        if not pages:
            preds = [[s for s in p if s["label"] in MESSAGE_LABELS] for p in preds]
        dest = a.out / path.name
        with open(dest, "w", encoding="utf-8") as fh:
            for row, spans in zip(rows, preds):
                fh.write(json.dumps({"row_id": _row_id(row), "spans": spans}, ensure_ascii=False) + "\n")
        n = sum(len(p) for p in preds)
        meta["files"][path.name] = {"rows": len(rows), "spans": n, "pages": pages}
        log(f"  -> {dest} ({n} spans)")
    (a.out / "run_metadata.json").write_text(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

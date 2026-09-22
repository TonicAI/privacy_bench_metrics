# privacy_bench_metrics

The evaluation code for **[PrivacyBench](https://huggingface.co/datasets/TonicAI/Privacy-Bench)** —
a benchmark for synthesizing personal workplace data (detecting PII in
emails/Slack messages and replacing it with coherent synthetic values).

This repo scores a synthesizer's output against the benchmark's ground
truth and reports three metrics with fixed denominators, so each stage's
failures land in exactly one column:

1. **NER recall** — of the gold PII spans, the fraction the synthesizer
   detected (detection only — replacement is scored separately).
   `detected / gold`. Detection requires a predicted span that overlaps
   the gold span *and carries its label* — a span found only under a
   different label is an NER miss. The denominator is always the total
   gold-span count.
2. **Synthesis accuracy** — of the detected gold spans, the fraction
   whose synthetic value is coherent, judged by an LLM against each
   character's PII. `coherent / detected`. An identity mapping
   (`new_text` equal to `text` — the value was detected but not
   actually changed) always counts as a miss, and spans the judge
   failed to evaluate stay in the denominator.
3. **Synthesis + NER accuracy** — the bottom-line, end-to-end score: of
   all gold PII spans, the fraction that was detected *and* coherently
   replaced. `coherent / gold`. Equals NER recall × synthesis accuracy,
   and a pipeline cannot inflate it by detecting less.

Each stage's failures land in exactly one metric: a span the NER step
missed only hurts recall, a detected span that was left unchanged or
replaced incoherently only hurts synthesis accuracy, and the combined
score is their product.

The dataset itself (inputs, ground-truth spans, and character rosters)
lives on Hugging Face:
**https://huggingface.co/datasets/TonicAI/Privacy-Bench**

## Install

```bash
git clone git@github.com:TonicAI/privacy_bench_metrics.git
cd privacy_bench_metrics
# NER recall is pure-stdlib; the LLM judge needs anthropic:
pip install anthropic
export ANTHROPIC_API_KEY=...   # for synthesis accuracy
```

Requires Python 3.10+.

## Input: a synthesizer output file

One JSON object per line, carrying only the detected entities and their
synthetic replacements, joined to the ground truth by `row_id`:

```json
{
  "row_id": "<matches meta.row_id in the ground-truth file>",
  "entities": [
    {"start": 46, "end": 58, "label": "USERNAME",
     "text": "<@U02CARLOS>", "new_text": "<@U02MIGUEL_SANTOS>"}
  ]
}
```

`start`/`end` are character offsets into the original message text;
`label` is one of the ten PrivacyBench labels (see below). An entity whose `new_text` equals its `text` counts as
detected but not synthesized — it scores as a synthesis miss.

## Run

Point the evaluator at your output file plus the dataset's ground-truth
files (downloaded from the Hugging Face dataset):

```bash
python -m synthesis_evaluation.run_eval \
  --predictions  <your_output>.jsonl \
  --ground-truth <set>_ground_truth_spans.jsonl \
  --characters   <set>_characters_ground_truth.json \
  --run-name     my_run
```

This writes `synthesis_evaluation/runs/my_run/` with:

- `results.json` — the three headline scores as structured data
  (overall, per entity type, and per character, under `metrics`), plus
  supporting detail (missed-PII examples and the judge's raw verdicts,
  under `detail`),
- `summary.md` — the headline scores + per-entity-type and
  per-character tables,
- `viewer.html` — an interactive view of missed PII, PII left
  unchanged, and the judge's incoherent verdicts.

To score only detection (no API key, no roster), add `--skip-llm-judge`
and drop `--characters`.

## Scoring a raw-export pipeline (native coordinates)

The published dataset is a file-tree benchmark: `tasks/<set>/` holds raw `.eml`
files, a Slack export and a Drive listing with PDF/DOCX/XLSX/CSV documents, and
`ground_truth/<set>/ground_truth.jsonl` locates every gold span in the containing
file's **native coordinates** (file offsets for `.eml`, JSON pointer + offsets
for Slack, per-glyph boxes for PDF, XPath fragments for DOCX, cells for XLSX,
row/column for CSV). A pipeline that works on the raw export reports what it
detected and replaced in the same shape, one JSON line per file unit:

```json
{"file": {"kind": "pdf", "path": "email/<id>.eml",
          "container": {"kind": "eml_attachment", "path": "email/<id>.eml", "part_index": 0}},
 "spans": [{"label": "NAME_GIVEN", "text": "Megan", "new_text": "Alicia",
            "location": {"kind": "pdf", "pages": [1], "chars": [["M", 1, 72.0, 96.4, 78.1, 105.4], ...]}}]}
```

`file` names the export file (and, for a document embedded in an email, the MIME
part); `location` uses exactly the ground truth's coordinate schema for that kind
(copy the shape from `ground_truth.jsonl`); `text` is the original string and
`new_text` the rendered replacement. Score it with:

```bash
python -m synthesis_evaluation.run_eval_native \
  --predictions  my_pipeline/aaron_pfizer/predictions.jsonl \
  --ground-truth $DATA/ground_truth/aaron_pfizer/ground_truth.jsonl \
  --characters   $DATA/ground_truth/aaron_pfizer/characters.json \
  --run-name     aaron_pfizer_my_pipeline [--judge-model claude-opus-5] [--skip-llm-judge]
```

Each gold span is matched to the predicted span of the same file unit that
overlaps it most in native coordinates (at least half of the union) and carries
its label; the matched replacement then goes through the same recall, LLM-judge
and metric code as the message-level evaluator, so the three headline numbers
keep their definitions. `summary.md` adds a per-file-kind table (eml, slack,
pdf, docx, xlsx, csv, messages, documents). Every gold span of every file is in
the denominator whether or not the pipeline emitted that file; the only
exclusion is the 166 PDF spans the renderer clipped (`location.status ==
"not_rendered"`). Predictions that overlap no gold span are not scored: the
gold covers the seed characters and their organizations only, so an unmatched
prediction is not necessarily wrong. Because the whole export is judged
together, a pipeline that replaces the same person differently in its emails
and in its documents is scored as incoherent; consistency across the export is
part of the task.

All ten labels are in scope (`NAME_GIVEN`, `NAME_FAMILY`, `EMAIL_ADDRESS`,
`USERNAME`, `ORGANIZATION`, `PHONE_NUMBER`, `LOCATION_ADDRESS`, `EMPLOYEE_ID`,
`ACCOUNT_NUMBER`, `URL`); the judge groups a character's names, emails,
handles, phones, addresses and ids and checks they form one coherent synthetic
identity, and groups each organization's names, addresses, URLs, phones and
account numbers likewise. The message-level evaluator above (`run_eval`,
offsets into the row text, joined by `row_id`) accepts the same ten labels.

## NER metrics against the human annotations

The dataset also ships exhaustive human NER annotations:
`human_annotations/<set>.jsonl` — the email and Slack messages of six of
the datasets (the five original labels) — and
`human_annotations/<set>_documents.jsonl` — the PDF/DOCX document pages
of the same six datasets, over all ten labels. Against that gold, precision and
F1 are meaningful (against the generated ground truth only recall is),
so a second, pure-stdlib evaluator scores raw NER predictions per
dataset, pooled (micro), and macro, under two matching rules: *overlap*
(label-matched character overlap) and *exact* (identical
`start`/`end`/`label`). This is the scoring behind the dataset card's
message and document NER tables.

Predictions are one JSON line per message or document page: a row id
(`row_id`, `meta.row_id`, or `meta.cell_id`; document pages append
`#p<meta.page>` since a document's pages share its row id — the runner
below does this for you) plus spans under `entities` or `spans`, each
with `start`/`end`/`label` (`new_text` is not needed). Labels outside
the benchmark's ten are ignored, and common engine vocabularies are
aliased onto them (`LOCATION` → `LOCATION_ADDRESS`, `US_BANK_NUMBER` →
`ACCOUNT_NUMBER`, `NUMERIC_PII` → the id labels, …) — see
`LABEL_ALIASES` in `synthesis_evaluation/types.py`.

```bash
DATA=/path/to/the/downloaded/dataset
python -m synthesis_evaluation.run_ner_eval \
  --gold "$DATA/human_annotations/*.jsonl" \
  --predictions-dir my_ner_output \
  --run-name my_engine_human_gold
```

`--predictions-dir` pairs each gold file with the file of the same name
in that directory; alternatively repeat `--gold FILE --predictions FILE`
for explicit pairs. This writes `synthesis_evaluation/runs/<run-name>/`
with `results.json` and `summary.md` (per-dataset, pooled, and macro
P/R/F1 tables plus pooled per-label recall).

To reproduce the dataset card's Tonic Textual scores, `pip install
tonic-textual regex`, set `TONIC_TEXTUAL_API_KEY`, and produce the predictions
with the bundled runner (it prints the Textual server version — the card
states which version its scores came from). The card's **Textual via SDK**
rows, messages and document pages alike, apply the graph pipeline's Textual
configuration expressed in SDK terms (`synthesis_evaluation/textual_config.json`):
allow lists for ACCOUNT_NUMBER, LOCATION_ADDRESS and ORGANIZATION and the
username regexes forced server side, an EMAIL_ADDRESS block list, and the
employee-id regexes applied client side. The ORGANIZATION allow list is the
graph config's, restricted to the entries whose comment names one of the six
human-annotated datasets (Coronado and Trimble Fleet Telematics API for
`nora_caterpillar`, Walmart and Fresh & Value-Added Poultry for
`renee_tyson_foods`); the config carries two more for datasets without human
annotations, which never match their text. Message files are filtered to the
five original labels automatically; page files keep all ten:

```bash
python -m synthesis_evaluation.run_textual_ner \
  --input "$DATA/human_annotations/*.jsonl" \
  --config synthesis_evaluation/textual_config.json \
  --out textual_predictions
python -m synthesis_evaluation.run_ner_eval \
  --gold "$DATA/human_annotations/*.jsonl" \
  --predictions-dir textual_predictions \
  --run-name textual_human_gold
```

Without `--config` the runner falls back to the plain mode of the earlier
message-only scores (built-in labels plus the USERNAME allow-list regex, so
Slack mentions like `<@U02CARLOS>` come back as single bracket-inclusive
spans); document-page files in the glob are skipped in that mode. Scores
reproduce to within about a tenth of a point (the Textual service is very
slightly nondeterministic on borderline detections). Note the config's
allow-list regexes were derived from the benchmark's generated ground
truth and, for organizations, from the human gold itself — the card
carries the same caveat.

### Reproducing the Presidio and GLiNER2 rows

The card's two open-source baselines run through
`synthesis_evaluation/run_baseline_ner.py`, which reads the same
human-annotation files and writes `run_ner_eval` prediction files (one
per input file, page rows keyed `<row_id>#p<page>`). Message files are
scored on the five original labels, so the engines are asked only for
person / organization / email / username there; document pages carry all
ten labels, so phone numbers, addresses, ids and URLs are requested as
well. Engine labels outside the benchmark vocabulary (Presidio's
`LOCATION`, `US_BANK_NUMBER`, `ID_NUMBER`, …) are written as-is and folded
onto the benchmark labels by `LABEL_ALIASES` at scoring time. Person spans
are split into a first-token NAME_GIVEN and a last-token NAME_FAMILY, as
the benchmark's other NER pipelines do.

**Presidio** — spaCy `en_core_web_trf` through presidio's NLP engine
(ORG kept at spaCy's score instead of presidio's down-weighting),
presidio's built-in recognisers, a USERNAME pattern for Slack mentions
and `@handles`, and on pages a generic `[A-Z]{2,5}-\d{4,9}` id pattern
emitted as `ID_NUMBER`; minimum score 0.5 (phones keep presidio's 0.4):

```bash
pip install presidio-analyzer spacy && python -m spacy download en_core_web_trf
python -m synthesis_evaluation.run_baseline_ner --engine presidio \
  --input "$DATA/human_annotations/*.jsonl" --out presidio_predictions
python -m synthesis_evaluation.run_ner_eval --gold "$DATA/human_annotations/*.jsonl" \
  --predictions-dir presidio_predictions --run-name presidio_human_gold
```

**GLiNER2** — `fastino/gliner2-privacy-filter-PII-multi`, schema-conditioned
with the benchmark's label descriptions, threshold 0.3, texts windowed at
350 words with 50 words of overlap. `gliner2` needs a newer `transformers`
than `spacy-transformers` accepts, so give it its own virtualenv:

```bash
pip install gliner2
python -m synthesis_evaluation.run_baseline_ner --engine gliner \
  --input "$DATA/human_annotations/*.jsonl" --out gliner_predictions
python -m synthesis_evaluation.run_ner_eval --gold "$DATA/human_annotations/*.jsonl" \
  --predictions-dir gliner_predictions --run-name gliner_human_gold
```

Both engines are deterministic on CPU; the pooled rows of the two
`summary.md` files are the card's Presidio and GLiNER2 rows. The model
checkpoint, threshold and spaCy model are flags (`--gliner-model`,
`--threshold`, `--spacy-model`, `--min-score`) and are recorded in
`run_metadata.json` next to the predictions.

## License

Maintained by [Tonic AI](https://www.tonic.ai/); license to be confirmed.
The PrivacyBench dataset is licensed separately — see its Hugging Face
dataset card.

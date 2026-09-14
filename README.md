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
of four of them, over all ten labels. Against that gold, precision and
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
tonic-textual`, set `TONIC_TEXTUAL_API_KEY`, and produce the predictions
with the bundled runner (it prints the Textual server version — the card
states which version its scores came from). The card's **message** rows
use the plain mode, which applies the USERNAME allow-list regex the
published scores used (so Slack mentions like `<@U02CARLOS>` come back
as single bracket-inclusive spans); document-page files in the glob are
skipped in this mode:

```bash
python -m synthesis_evaluation.run_textual_ner \
  --input "$DATA/human_annotations/*.jsonl" \
  --out textual_msgs
```

The card's **document** rows ("Textual (SDK, graph config)") add
`--config` — the graph pipeline's Textual configuration expressed in SDK
terms (`synthesis_evaluation/textual_config.json`: ACCOUNT_NUMBER and
LOCATION_ADDRESS allow lists, an EMAIL_ADDRESS block list, username
regexes server side, employee-id regexes client side). It needs
`pip install regex` (the config uses variable-width look-behinds):

```bash
python -m synthesis_evaluation.run_textual_ner \
  --input "$DATA/human_annotations/*_documents.jsonl" \
  --config synthesis_evaluation/textual_config.json \
  --out textual_docs
```

then score each output folder with the `run_ner_eval` command above
(`--gold` the same files). Message scores reproduce to within about a
tenth of a point (the Textual service is very slightly nondeterministic
on borderline detections); the document scores reproduce the card's
"Textual (SDK, graph config)" row. Note the config's allow-list regexes
were derived from the benchmark's generated ground truth — the card
carries the same caveat.

## License

Maintained by [Tonic AI](https://www.tonic.ai/); license to be confirmed.
The PrivacyBench dataset is licensed separately — see its Hugging Face
dataset card.

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

To run the judge on Amazon Bedrock instead of the Claude API, install
`anthropic[bedrock]` and add:

```bash
  --judge-provider bedrock --judge-model global.anthropic.claude-opus-5-5 [--judge-region us-east-1]
```

`--judge-model` is required there (a Bedrock model or inference-profile id or
ARN) and the region defaults to `AWS_REGION`, then `AWS_DEFAULT_REGION`.
`AWS_BEARER_TOKEN_BEDROCK`, when set, takes precedence; otherwise credentials
come from the standard AWS chain. `ANTHROPIC_API_KEY` is not read. The calls
stream, so the caller needs `bedrock:InvokeModelWithResponseStream` on the
inference profile and the foundation models it routes to.

A Bedrock judge call that fails with a throttling, overloaded, timeout, 5xx
or connection error, including one raised partway through the response
stream, is retried. The SDK's own retries are turned off on this client, so
the judge's loop is the only retry layer: each judge call (one per
character, org-group chunk or unowned batch) sends at most 5 HTTP requests.
Auth, validation and access-denied errors are not retried.

- **Backoff.** The four waits are 8, 16, 32 and 64 s, each give or take 25%,
  so a call that keeps being throttled waits 90 to 150 s in all, about the
  two minutes the SDK's own retries used to allow. When the failed response
  carries `retry-after-ms` or `retry-after` (seconds or an HTTP date), the
  wait is at least that long, capped at 60 s, which makes the most a call
  can wait 260 s. A failure elsewhere in the run ends any wait at once.
- **Timeouts.** An attempt has two separate limits, and either one ends it
  as a timeout, which is retried.
  - *Idle read: 300 s.* The client's timeout is 300 s per read, with 10 s to
    connect, instead of the SDK's 600 s, so 300 s with no bytes ends the
    attempt. With the default `display: "omitted"`, the thinking streams no
    text, so the longest silent gap can be the whole thinking phase. At a
    conservative 40 output tokens a second, 300 s is 12,000 tokens of
    thinking, the whole `MAX_TOKENS` budget. An idle limit alone cannot end
    an attempt that keeps receiving bytes, and a hung stream can keep
    sending keepalives indefinitely.
  - *Whole stream: 900 s.* No read of the response starts once 900 s have
    passed since the attempt began. The check runs before every read of
    the body, not per event, so bytes or keepalives that never complete an
    event cannot extend it. A reply whose `message_stop` event has arrived
    is kept however late it is, so a finished reply is never thrown away.
    A reply is at most `MAX_TOKENS` (12,000) output tokens, adaptive
    thinking included, whatever the owner's size; the largest chunk, 25
    org-group pairs, needs far fewer. At 15 output tokens a second, under
    half the conservative 40 above, a full-length reply streams in 800 s,
    and the other 100 s covers the time to the first token. Neither speed
    is measured: the docs give no output rate, and M5's first judged run is
    the first measurement.
- **Worst-case wall time per call.** One attempt takes at most about
  1,200 s: no read starts after 900 s, and the read in progress then ends
  within its 300 s idle limit. An attempt that receives nothing at all ends
  after about 310 s (10 s to connect, then the 300 s read timeout). So a
  call takes at most 5 × 1,200 + 260 s, about 105 minutes. A call whose
  stream hangs silently takes at most 5 × 310 + 260 s, about 30 minutes.
  A call that is only throttled waits 90 to 150 s in all, or up to 260 s
  with `retry-after`. Each worker runs one call at a time, so a run sends
  at most `--workers` (default 8) requests at once. The caller's own
  timeout should bound the whole run.

Unlike the default provider, the Bedrock judge does not skip a call. The run
exits non-zero, writing no `results.json` and leaving no run directory, on any
of these:
- missing credentials;
- a call that still fails;
- a reply that does not parse, judges none of its pairs, or carries a
  malformed verdict for one of them (a `coherent` that is not a boolean, or
  `values` that is not a list of `{value, coherent}` entries);
- a run in which every pair is unjudged.

Only a reply's verdicts for the call's own pairs are kept, so the tallies and
the headline metrics read exactly the entries that were checked. A verdict
for a pair the call did not show the judge, such as one from another chunk of
the same org group, or one under a label the call did not ask about, is
dropped. `usage.n_dropped_verdicts` counts these, and the run prints the
count. A pair that an otherwise valid reply leaves out still counts as
skipped, so it stays in the synthesis-accuracy denominator; the run prints
how many pairs went unjudged, and `results.json` keeps the counts.

The run is built in a hidden sibling of the run directory,
`.<run-name>.partial-*`, which is created before any judge call. A
`--runs-dir` that cannot hold it, or a run name that is already taken,
therefore fails before anything is billed. The sibling has the same mode as
a run directory on the default provider (what the umask gives). It is
renamed to the run name only once `results.json`, `summary.md` and
`viewer.html` are all written, and it is removed on any failure, on Ctrl-C
and on SIGTERM, which the Bedrock run turns into exit code 143 after
cleaning up. Only a process killed outright (SIGKILL, the OOM killer) leaves
the hidden sibling behind; it does no harm, since a rerun makes a new one
and checks only the run name. If anything has taken the run name meanwhile,
even an empty directory or a file, the run exits non-zero with `refusing to
overwrite existing run dir` rather than replace it: the name is claimed with
`mkdir`, which fails on any existing entry, and the rename then replaces
only that empty directory. `config` records the provider,
the model (with an ARN's account id redacted) and `judge_effort: "default"`,
since no effort level is sent.

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
configuration expressed in SDK terms (`synthesis_evaluation/textual_config.json`). 
Message files are filtered to the
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
reproduce to within about a tenth of a point.

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

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
`label` ∈ {`NAME_GIVEN`, `NAME_FAMILY`, `EMAIL_ADDRESS`, `USERNAME`,
`ORGANIZATION`}. An entity whose `new_text` equals its `text` counts as
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

## NER metrics against the human annotations

The dataset also ships `human_annotations/<set>.jsonl` — an exhaustive
human annotation of the email and Slack messages of six of the datasets.
Against that gold, precision and F1 are meaningful (against the
generated ground truth only recall is), so a second, pure-stdlib
evaluator scores raw NER predictions per dataset, pooled (micro), and
macro, under two matching rules: *overlap* (label-matched character
overlap) and *exact* (identical `start`/`end`/`label`). This is the
scoring behind the dataset card's "NER engines scored on the human
gold" table.

Predictions are one JSON line per message: a row id (`row_id`,
`meta.row_id`, or `meta.cell_id`) plus spans under `entities` or
`spans`, each with `start`/`end`/`label` (`new_text` is not needed;
labels outside the five above are ignored).

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
with the bundled runner — it applies the USERNAME allow-list regex the
published scores used (so Slack mentions like `<@U02CARLOS>` come back
as single bracket-inclusive spans) and prints the Textual server version
(the card states which version its scores came from):

```bash
python -m synthesis_evaluation.run_textual_ner \
  --input "$DATA/human_annotations/*.jsonl" \
  --out textual_predictions
```

then score `textual_predictions` with the `run_ner_eval` command above.
Scores reproduce to within about a tenth of a point — the Textual
service is very slightly nondeterministic on borderline detections.

## License

Maintained by [Tonic AI](https://www.tonic.ai/); license to be confirmed.
The PrivacyBench dataset is licensed separately — see its Hugging Face
dataset card.

# privacy_bench_metrics

The evaluation code for **[PrivacyBench](https://huggingface.co/datasets/TonicAI/Privacy-Bench)** —
a benchmark for synthesizing personal workplace data (detecting PII in
emails/Slack messages and replacing it with coherent synthetic values).

This repo scores a synthesizer's output against the benchmark's ground
truth and reports three metrics:

1. **NER recall** — of the gold PII spans, the fraction the synthesizer
   both detected *and* replaced. `TP / (TP + FN)`. A miss is a privacy
   leak, so this is the primary detection axis.
2. **Synthesis accuracy** — of the spans that *were* detected and
   replaced, the fraction whose synthetic value is coherent, judged by an
   LLM against each character's PII. `coherent / (coherent + incoherent)`.
3. **Synthesis + NER accuracy** — the bottom-line score, folding
   detection misses into the denominator so a pipeline cannot inflate its
   number by detecting less.
   `coherent / (coherent + incoherent + missed)`.

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
`ORGANIZATION`}; an entity counts as "detected and replaced" only when
`new_text` differs from `text`.

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

- `results.json` — the metrics as structured data,
- `summary.md` — the three headline scores + per-entity-type tables,
- `viewer.html` — an interactive view of missed PII and the judge's
  incoherent verdicts.

To score only detection (no API key, no roster), add `--skip-llm-judge`
and drop `--characters`.

## License

Maintained by [Tonic AI](https://www.tonic.ai/); license to be confirmed.
The PrivacyBench dataset is licensed separately — see its Hugging Face
dataset card.

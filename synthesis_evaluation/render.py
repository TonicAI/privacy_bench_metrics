"""Render the markdown summary and HTML failure viewer from results.json."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Iterable, List

from .types import LABELS

# Character-level entity types (ORGANIZATION is scored at the
# org-group level, not per character).
CHAR_LABELS = tuple(l for l in LABELS if l != "ORGANIZATION")


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def _fmt_pct(x):
    if x is None:
        return "n/a"
    return f"{x*100:.1f}%"


def compute_overall_scores(results: dict) -> dict:
    """The three headline scores as ``(numerator, denominator)`` pairs.

    Every original→synthetic span mapping is scored by its synthetic
    value's coherence verdict (the ``synthesis_precision`` /
    ``synthesis_recall`` keys are kept for backward compat, but the
    reports display them as "synthesis accuracy" / "synthesis + NER
    accuracy"):
      synthesis accuracy       = coherent / (coherent + incoherent)
      synthesis + NER accuracy = coherent / (coherent + incoherent + missed),
    where missed = NER false negatives (spans never synthesized at
    all). Both are ``None`` when the LLM judge was skipped.

    Shared by render_summary and the cross-run comparison tables.
    """
    m = results["metrics"]
    rec_ov = m["recall"]["overall"]
    rl = m["realism_llm"]
    out = {
        "ner_recall": (rec_ov["tp"], rec_ov["total"]),
        "synthesis_precision": None,
        "synthesis_recall": None,
    }
    if rl.get("skipped_reason"):
        return out
    plst = rl.get("per_label_span_totals") or {}
    ogst = rl.get("org_group_span_totals") or {}

    def _span_totals(lab: str) -> dict:
        # ORGANIZATION lives in the org-group totals rather than
        # per_label_span_totals.
        return ogst if lab == "ORGANIZATION" else (plst.get(lab) or {})

    coh = sum(_span_totals(lab).get("coherent_spans", 0) for lab in LABELS)
    incoh = sum(_span_totals(lab).get("incoherent_spans", 0) for lab in LABELS)
    missed = sum(
        int((m["recall"]["by_label"].get(lab) or {}).get("fn", 0) or 0)
        for lab in LABELS
    )
    out["synthesis_precision"] = (coh, coh + incoh)
    out["synthesis_recall"] = (coh, coh + incoh + missed)
    return out


def render_summary(results: dict) -> str:
    cfg = results["config"]
    m = results["metrics"]
    lines: List[str] = []

    lines.append(f"# Synthesis Evaluation: {cfg['run_name']}\n")
    lines.append(f"- Predictions: `{cfg.get('predictions')}`")
    lines.append(f"- Ground truth: `{cfg.get('ground_truth')}`")
    lines.append(f"- Characters: `{cfg.get('characters')}`")
    lines.append(f"- Rows evaluated: {cfg['n_rows']}\n")

    # Overall scores at the top
    lines.append("## Overall scores\n")
    rl = m["realism_llm"]

    def _pct_frac(num: int, denom: int) -> str:
        if denom == 0:
            return "—"
        return f"{_fmt_pct(num / denom)}  ({num}/{denom})"

    overall = compute_overall_scores(results)
    score_rows: List[List[str]] = [
        ["NER recall", _pct_frac(*overall["ner_recall"])],
    ]
    if rl.get("skipped_reason"):
        score_rows.append(["LLM judge", f"_skipped: {rl['skipped_reason']}_"])
    else:
        score_rows.append([
            "LLM judge synthesis accuracy",
            _pct_frac(*overall["synthesis_precision"]),
        ])
        score_rows.append([
            "LLM judge synthesis + NER accuracy",
            _pct_frac(*overall["synthesis_recall"]),
        ])
    lines.append("| Metric | Score |")
    lines.append("|---|---|")
    for k, v in score_rows:
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # NER recall (mask-rate)
    r = m["recall"]
    lines.append("## NER recall\n")
    lines.append(
        "Recall = TP / (TP + FN), where a TP is a ground-truth PII span "
        "that the synthesizer detected and replaced with a new value.\n"
    )
    lines.append("**By entity type:**\n")
    lines.append("| Label | Recall | TP | FN | Total |")
    lines.append("|---|---|---|---|---|")
    for lab in LABELS:
        x = r["by_label"][lab]
        lines.append(
            f"| {lab} | {_fmt_pct(x['recall'])} | {x['tp']} | "
            f"{x['fn']} | {x['total']} |"
        )
    lines.append("")

    # Per-character per-label recall table.
    #
    # Multi-character gold spans (e.g. "Donovan" shared by megan+brian)
    # count under EACH character, so the per-character TP+FN sums can
    # exceed the global totals.
    pc = r.get("per_character") or {}
    if pc:
        lines.append(
            "**Per-character recall.** "
            "Each cell shows `recall (TP/total)` for that (character, label). "
            "`—` means the character had zero ground-truth spans for that "
            "label. Multi-character gold spans count under each character, "
            "so per-character TP+FN sums can exceed the global totals.\n"
        )
        header = "| Character | " + " | ".join(LABELS) + " | Overall |"
        lines.append(header)
        lines.append("|---" * (len(LABELS) + 2) + "|")

        def _cell(block: dict) -> str:
            if block["total"] == 0:
                return "—"
            return f"{_fmt_pct(block['recall'])} ({block['tp']}/{block['total']})"

        for cid in sorted(pc):
            row_block = pc[cid]
            cells = [_cell(row_block["by_label"][lab]) for lab in LABELS]
            overall_cell = _cell(row_block["overall"])
            lines.append(f"| {cid} | " + " | ".join(cells) + f" | {overall_cell} |")
        lines.append("")

    # FN buckets — every (text, label) surface form that was missed,
    # with how many times. Sorted by count desc.
    fn_buckets = r.get("fn_by_surface") or []
    if fn_buckets:
        lines.append(
            "**Missed (text, label) buckets.** Every ground-truth surface "
            "form that was a false negative, with how many times. Sorted "
            "by count desc.\n"
        )
        lines.append("| Original surface | Label | Missed count |")
        lines.append("|---|---|---|")
        for b in fn_buckets:
            lines.append(f"| `{b['text']}` | {b['label']} | {b['count']} |")
        lines.append("")

    # Realism LLM
    rl = m["realism_llm"]
    lines.append("## Synthesis accuracy / synthesis + NER accuracy — LLM judge\n")
    if rl.get("skipped_reason"):
        lines.append(f"_Skipped: {rl['skipped_reason']}_\n")
    else:
        usage = rl.get("usage") or {}
        lines.append(f"- LLM calls: {usage.get('n_calls', 0)} "
                     f"(parse failures: {usage.get('n_parse_failures', 0)})")
        lines.append(f"- Tokens: input={usage.get('input_tokens', 0):,} "
                     f"output={usage.get('output_tokens', 0):,} "
                     f"cache_read={usage.get('cache_read_input_tokens', 0):,} "
                     f"cache_create={usage.get('cache_creation_input_tokens', 0):,}\n")

        plst_d = rl.get("per_label_span_totals") or {}
        ogst_d = rl.get("org_group_span_totals") or {}
        # Per-label FN counts (from recall) widen the recall denominator
        # to include spans that were never synthesized.
        fn_by_label = {
            lab: (m["recall"]["by_label"].get(lab) or {}).get("fn", 0)
            for lab in LABELS
        }
        if plst_d or ogst_d:
            lines.append(
                "Every original→synthetic span mapping is scored by the LLM "
                "judge's coherence verdict for its synthetic value. "
                "Synthesis accuracy = coherent / (coherent + incoherent); "
                "synthesis + NER accuracy adds the missed spans (NER false "
                "negatives) to the denominator. Skipped spans (no parsable "
                "verdict) are excluded from both.\n"
            )
            lines.append("**By entity type:**\n")
            lines.append("| Label | Synthesis accuracy | Synthesis + NER accuracy | Coherent | "
                         "Incoherent | Missed | Skipped |")
            lines.append("|---|---|---|---|---|---|---|")
            for lab in LABELS:
                s = ogst_d if lab == "ORGANIZATION" else (plst_d.get(lab) or {})
                coh_s = s.get("coherent_spans", 0)
                incoh_s = s.get("incoherent_spans", 0)
                miss_s = int(fn_by_label.get(lab, 0) or 0)
                lines.append(
                    f"| {lab} | {_pct_frac(coh_s, coh_s + incoh_s)} | "
                    f"{_pct_frac(coh_s, coh_s + incoh_s + miss_s)} | "
                    f"{coh_s} | {incoh_s} | {miss_s} | "
                    f"{s.get('skipped_spans', 0)} |"
                )
            lines.append("")

        org_inc = rl.get("org_incoherent_verdicts") or []
        if org_inc:
            lines.append("**Incoherent org surface buckets:**")
            for v in org_inc:
                lines.append(f"- **{v['org_group']}** `{v.get('surface', '?')}` "
                             f"(confidence: {v.get('confidence', 'n/a')})")
                for issue in (v.get("issues") or [])[:5]:
                    lines.append(f"  - {issue}")
            lines.append("")

        per_char = rl.get("per_character") or {}
        if per_char:
            lines.append(
                "**Per-character coherent surface buckets per entity type** "
                "(`coherent/judged` over the character's ground-truth surface "
                "forms; `+k?` = k surfaces without a parsable verdict). "
                "`—` means the character had no synthesis data for that label.\n"
            )
            lines.append("| Character | " + " | ".join(CHAR_LABELS) + " |")
            lines.append("|---" * (len(CHAR_LABELS) + 1) + "|")
            for cid in sorted(per_char):
                v = per_char[cid]
                mapping = v.get("mapping") or {}
                parsed = v.get("parsed") or {}
                verdicts_by_pair = {}
                for entry in (parsed.get("verdicts") or []):
                    if (isinstance(entry, dict)
                            and entry.get("label") in CHAR_LABELS
                            and isinstance(entry.get("surface"), str)):
                        verdicts_by_pair[(entry["label"], entry["surface"])] = \
                            bool(entry.get("coherent"))
                cells = []
                for lab in CHAR_LABELS:
                    surfaces = mapping.get(lab) or {}
                    if not surfaces:
                        cells.append("—")
                        continue
                    judged = [verdicts_by_pair[(lab, s)] for s in surfaces
                              if (lab, s) in verdicts_by_pair]
                    n_skip = len(surfaces) - len(judged)
                    cell = f"{sum(judged)}/{len(judged)}" if judged else "?"
                    if judged and n_skip:
                        cell += f" +{n_skip}?"
                    cells.append(cell)
                lines.append(f"| {cid} | " + " | ".join(cells) + " |")
            lines.append("")

        per_org = rl.get("per_org_group") or {}
        if per_org:
            lines.append("**Per-org-group coherent surface buckets** "
                         "(`coherent/judged`; `+k?` = k surfaces without a "
                         "parsable verdict):\n")
            lines.append("| Org group | Coherent surfaces |")
            lines.append("|---|---|")
            for grp in sorted(per_org):
                v = per_org[grp]
                mapping = v.get("mapping") or {}
                parsed = v.get("parsed") or {}
                verdicts_by_surface = {}
                for entry in (parsed.get("verdicts") or []
                              if isinstance(parsed, dict) else []):
                    if isinstance(entry, dict) and isinstance(entry.get("surface"), str):
                        verdicts_by_surface[entry["surface"]] = \
                            bool(entry.get("coherent"))
                judged = [verdicts_by_surface[s] for s in mapping
                          if s in verdicts_by_surface]
                n_skip = len(mapping) - len(judged)
                cell = f"{sum(judged)}/{len(judged)}" if judged else "?"
                if judged and n_skip:
                    cell += f" +{n_skip}?"
                lines.append(f"| {grp} | {cell} |")
            lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# HTML viewer
# ---------------------------------------------------------------------------

def _h(x) -> str:
    return html.escape("" if x is None else str(x), quote=False)


def _snippet_html(text: str, start: int, end: int) -> str:
    """Return a snippet with the [start:end] region wrapped in <mark>."""
    if not text:
        return ""
    lo = max(0, start - 60)
    hi = min(len(text), end + 60)
    pre  = _h(text[lo:start])
    mid  = _h(text[start:end])
    post = _h(text[end:hi])
    lead = "…" if lo > 0 else ""
    trail = "…" if hi < len(text) else ""
    return f"{lead}{pre}<mark>{mid}</mark>{post}{trail}"


def render_viewer(results: dict) -> str:
    m = results["metrics"]
    cfg = results["config"]

    def _row_table(headers: List[str], rows: Iterable[List[str]]) -> str:
        h = "".join(f"<th>{_h(c)}</th>" for c in headers)
        body = []
        for r in rows:
            body.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
        return f"<table><thead><tr>{h}</tr></thead><tbody>{''.join(body)}</tbody></table>"

    # Recall false-negative tab
    fn_rows = []
    for ex in m["recall"].get("fn_examples", []):
        g = ex["gold"]
        snippet_offset = max(0, g["start"] - 60)
        fn_rows.append([
            _h(ex["cell_id"]),
            _h(g["label"]),
            _h(g["text"]),
            _h(",".join(g.get("characters") or [])),
            _snippet_html(
                ex.get("snippet", ""),
                g["start"] - snippet_offset,
                g["end"] - snippet_offset,
            ),
        ])
    fn_tab = _row_table(
        ["cell_id", "label", "gold text", "characters", "context"],
        fn_rows,
    )

    # LLM incoherent (per character × label × surface). Each synthetic
    # value carries its own verdict (✓/✗) and gold-span count.
    llm_rows = []
    for entry in (m["realism_llm"].get("incoherent_verdicts") or []):
        issues = "<ul>" + "".join(f"<li>{_h(i)}</li>"
                                  for i in (entry.get("issues") or [])) + "</ul>"
        values = entry.get("values") or []
        if values:
            values_html = ", ".join(
                f"<code>{_h(v['value'])}</code> "
                f"{'✓' if v.get('coherent') else '✗'}"
                f"&nbsp;×{v.get('count', 0)}"
                for v in values if isinstance(v, dict)
            )
        else:
            mapping = entry.get("mapping") or {}
            values_html = "; ".join(
                _h(", ".join(synths)) for synths in mapping.values())
        llm_rows.append([
            _h(entry["character"]),
            _h(entry["label"]),
            f"<code>{_h(entry.get('surface', '?'))}</code>",
            _h(entry.get("confidence", "n/a")),
            issues,
            values_html,
        ])
    llm_tab = _row_table(
        ["character", "label", "GT surface", "confidence", "issues",
         "synthetic values (✓ coherent / ✗ incoherent, × span count)"],
        llm_rows,
    )

    tabs = [
        ("Recall FN (missed PII)",  fn_tab),
        ("LLM incoherent verdicts", llm_tab),
    ]

    nav = "".join(
        f'<button class="tab" data-target="t{i}">{_h(name)}</button>'
        for i, (name, _) in enumerate(tabs)
    )
    panels = "".join(
        f'<section id="t{i}" class="panel">{body}</section>'
        for i, (_, body) in enumerate(tabs)
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Synthesis eval: {_h(cfg['run_name'])}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif;
        margin: 16px; color: #222; }}
h1 {{ margin: 0 0 4px; }}
.meta {{ color: #666; font-size: 13px; margin-bottom: 16px; }}
nav {{ display: flex; gap: 6px; flex-wrap: wrap; border-bottom: 1px solid #ccc;
       margin-bottom: 12px; }}
nav .tab {{ background: #f3f3f3; border: 1px solid #ccc; border-bottom: none;
            padding: 6px 10px; cursor: pointer; font-size: 13px; }}
nav .tab.active {{ background: #fff; font-weight: 600; }}
.panel {{ display: none; }}
.panel.active {{ display: block; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top;
          text-align: left; }}
th {{ background: #fafafa; }}
mark {{ background: #ffec8b; padding: 0 2px; }}
ul {{ margin: 0; padding-left: 16px; }}
</style>
</head>
<body>
<h1>Synthesis eval: {_h(cfg['run_name'])}</h1>
<div class="meta">
  tier=<code>{_h(cfg.get('tier'))}</code>
  · rows=<code>{_h(cfg['n_rows'])}</code> · predictions=<code>{_h(cfg.get('predictions'))}</code>
</div>
<nav>{nav}</nav>
{panels}
<script>
const buttons = document.querySelectorAll('nav .tab');
const panels  = document.querySelectorAll('.panel');
function activate(id) {{
  buttons.forEach(b => b.classList.toggle('active', b.dataset.target === id));
  panels.forEach(p => p.classList.toggle('active', p.id === id));
}}
buttons.forEach(b => b.addEventListener('click', () => activate(b.dataset.target)));
activate('t0');
</script>
</body>
</html>
"""


def render(results: dict, summary_path: Path, viewer_path: Path) -> None:
    summary_path.write_text(render_summary(results))
    viewer_path.write_text(render_viewer(results))


def main() -> int:
    """Re-render summary.md + viewer.html from saved results.json files,
    without re-running any evaluation. Accepts one or more run dirs
    (each containing results.json)."""
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("run_dirs", nargs="+", type=Path,
                    help="Dirs containing results.json.")
    args = ap.parse_args()
    n = 0
    for d in args.run_dirs:
        rp = d / "results.json"
        if not rp.is_file():
            print(f"skip {d}: no results.json")
            continue
        render(json.loads(rp.read_text()), d / "summary.md", d / "viewer.html")
        n += 1
    print(f"re-rendered {n} run(s)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

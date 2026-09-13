"""Render the markdown summary and HTML failure viewer from results.json."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Iterable, List

from .types import LABELS, ORG_LABELS, PERSON_LABELS

# Character-level entity types (ORGANIZATION is scored at the
# org-group level, not per character).
CHAR_LABELS = PERSON_LABELS


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def _fmt_pct(x):
    if x is None:
        return "n/a"
    return f"{x*100:.1f}%"


def _fmt_prf(block: dict) -> str:
    return (f"P={_fmt_pct(block.get('precision'))}  "
            f"R={_fmt_pct(block.get('recall'))}  "
            f"F1={_fmt_pct(block.get('f1'))}")


def render_summary(results: dict) -> str:
    cfg = results["config"]
    metrics = results["metrics"]
    detail = results["detail"]
    ov = metrics["overall"]
    lines: List[str] = []

    lines.append(f"# Synthesis Evaluation: {cfg['run_name']}\n")
    lines.append(f"- Synthesis: `{(cfg.get('synthesis') or cfg.get('predictions'))}`")
    lines.append(f"- Characters: `{cfg['characters']}`")
    lines.append(f"- Adapter: `{(cfg.get('adapter') or cfg.get('format') or '-')}`  Tier: `{cfg['tier']}`")
    lines.append(f"- Rows evaluated: {cfg['n_rows']}\n")

    # Overall scores at the top
    lines.append("## Overall scores\n")

    def _pct_frac(num: int, denom: int) -> str:
        if denom == 0:
            return "—"
        return f"{_fmt_pct(num / denom)}  ({num}/{denom})"

    score_rows: List[List[str]] = [
        ["NER recall", _pct_frac(ov["detected"], ov["gold"])],
    ]
    if metrics.get("judge_skipped"):
        reason = (detail["judge"] or {}).get("skipped_reason", "skipped")
        score_rows.append(["LLM judge", f"_skipped: {reason}_"])
    else:
        score_rows.append([
            "Synthesis accuracy",
            _pct_frac(ov["coherent"], ov["detected"]),
        ])
        score_rows.append([
            "Synthesis + NER accuracy",
            _pct_frac(ov["coherent"], ov["gold"]),
        ])
    lines.append("| Metric | Score |")
    lines.append("|---|---|")
    for k, v in score_rows:
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append(
        "NER recall = detected gold spans / all gold spans, where "
        "detection requires a label-matched overlapping prediction. "
        "Synthesis accuracy = coherently synthesized / detected (an "
        "unchanged value or an incoherent replacement both count "
        "against it; spans the judge failed to evaluate stay in the "
        "denominator). Synthesis + NER accuracy = coherently "
        "synthesized / all gold spans — the product of the other two.\n")

    # NER recall (detection)
    r = detail["recall"]
    lines.append("## NER recall\n")
    lines.append(
        "Detection only: a TP is a ground-truth PII span the "
        "synthesizer detected under the right label (whether the value "
        "was actually changed is scored under synthesis accuracy "
        "below).\n"
    )
    lines.append("**By entity type:**\n")
    lines.append("| Label | Recall | Detected | Missed | Total |")
    lines.append("|---|---|---|---|---|")
    for lab in LABELS:
        x = metrics["by_label"][lab]
        missed = x["gold"] - x["detected"]
        lines.append(
            f"| {lab} | {_fmt_pct(x['ner_recall'])} | {x['detected']} | "
            f"{missed} | {x['gold']} |"
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

    # Synthesis accuracy — LLM judge + identity rule
    rl = detail["judge"]
    lines.append("## Synthesis accuracy / synthesis + NER accuracy\n")
    if metrics.get("judge_skipped"):
        lines.append(f"_Skipped: {rl.get('skipped_reason', 'skipped')}_\n")
    else:
        usage = rl.get("usage") or {}
        lines.append(f"- LLM judge calls: {usage.get('n_calls', 0)} "
                     f"(parse failures: {usage.get('n_parse_failures', 0)})")
        lines.append(f"- Tokens: input={usage.get('input_tokens', 0):,} "
                     f"output={usage.get('output_tokens', 0):,} "
                     f"cache_read={usage.get('cache_read_input_tokens', 0):,} "
                     f"cache_create={usage.get('cache_creation_input_tokens', 0):,}\n")

        lines.append(
            "Each detected gold span is scored by the LLM judge's "
            "coherence verdict for its synthetic value. Synthesis "
            "accuracy = coherent / detected: an identity mapping "
            "(value left unchanged) always counts as incoherent, and "
            "spans without a parsable verdict count as skipped — both "
            "stay in the denominator. Synthesis + NER accuracy folds "
            "detection misses back in: coherent / all gold spans.\n"
        )
        n_identity = sum(b["count"] for b in
                         (metrics.get("identity_by_surface") or []))
        lines.append("**By entity type:**\n")
        lines.append("| Label | Synthesis accuracy | Synthesis + NER accuracy | Coherent | "
                     "Incoherent | Skipped | Missed |")
        lines.append("|---|---|---|---|---|---|---|")
        for lab in LABELS:
            x = metrics["by_label"][lab]
            lines.append(
                f"| {lab} | {_pct_frac(x['coherent'], x['detected'])} | "
                f"{_pct_frac(x['coherent'], x['gold'])} | "
                f"{x['coherent']} | {x['incoherent']} | {x['skipped']} | "
                f"{x['gold'] - x['detected']} |"
            )
        lines.append("")
        if n_identity:
            lines.append(
                f"{n_identity} of the incoherent spans are identity "
                f"mappings — detected but left unchanged.\n")

        # Identity buckets — every (text, label) surface left unchanged.
        id_buckets = metrics.get("identity_by_surface") or []
        if id_buckets:
            lines.append(
                "**Unchanged (identity) buckets.** Every ground-truth "
                "surface form that was detected but returned with its "
                "original value, with how many times. Sorted by count "
                "desc.\n"
            )
            lines.append("| Original surface | Label | Unchanged count |")
            lines.append("|---|---|---|")
            for b in id_buckets:
                lines.append(f"| `{b['text']}` | {b['label']} | {b['count']} |")
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
            lines.append("| Org group | " + " | ".join(ORG_LABELS) + " |")
            lines.append("|---" * (len(ORG_LABELS) + 1) + "|")
            for grp in sorted(per_org):
                v = per_org[grp]
                mapping = v.get("mapping") or {}
                if mapping and not any(k in ORG_LABELS for k in mapping):
                    mapping = {"ORGANIZATION": mapping}          # pre-label runs: surfaces only
                parsed = v.get("parsed") or {}
                verdicts_by_pair = {}
                for entry in (parsed.get("verdicts") or []
                              if isinstance(parsed, dict) else []):
                    if isinstance(entry, dict) and isinstance(entry.get("surface"), str):
                        lab = entry.get("label") if entry.get("label") in ORG_LABELS else "ORGANIZATION"
                        verdicts_by_pair[(lab, entry["surface"])] = bool(entry.get("coherent"))
                cells = []
                for lab in ORG_LABELS:
                    surfaces = mapping.get(lab) or {}
                    if not surfaces:
                        cells.append("—")
                        continue
                    judged = [verdicts_by_pair[(lab, s)] for s in surfaces if (lab, s) in verdicts_by_pair]
                    n_skip = len(surfaces) - len(judged)
                    cell = f"{sum(judged)}/{len(judged)}" if judged else "?"
                    if judged and n_skip:
                        cell += f" +{n_skip}?"
                    cells.append(cell)
                cell = " | ".join(cells)
                lines.append(f"| {grp} | {cell} |")
            lines.append("")

    # Grouping
    g = detail["grouping"]
    lines.append("## Grouping\n")
    if g.get("skipped"):
        lines.append(f"_Skipped: {g.get('skipped_reason')}_\n")
    else:
        pw = g["pairwise"]
        b3 = g["b_cubed"]
        lines.append(f"- Pairwise: {_fmt_prf(pw)}")
        lines.append(f"- B³: P={_fmt_pct(b3['precision'])}  "
                     f"R={_fmt_pct(b3['recall'])}  F1={_fmt_pct(b3['f1'])}  "
                     f"(n={b3['n']})\n")

    # Per-character synthesis mapping — one table per character,
    # listing every (text, label) bucket and the synthetic values it
    # mapped to with occurrence counts. Passthrough entries show how
    # many times the synthesizer left the original surface in the
    # synthetic output unchanged. The data comes from the consistency
    # scorer, whose score itself is no longer reported.
    c = detail["consistency"]
    pc = c.get("per_character") or {}
    if pc:
        lines.append("## Per-character synthesis mapping\n")
        lines.append(
            "For each character, every (label, original surface) bucket "
            "with the synthetic value(s) it mapped to and how many times "
            "each was emitted. Entries tagged `(passthrough)` are gold "
            "spans the synthesizer left unchanged in the synthetic text "
            "(either undetected or detected-but-not-replaced).\n"
        )
        for cid in sorted(pc):
            block = pc[cid]
            # Build a flat list of (label, orig, [{value, count, ...}]).
            rows = []
            for lab in LABELS:
                blab = block.get(lab) or {}
                counts_map = blab.get("original_to_synthetic_counts") or {}
                for orig, items in counts_map.items():
                    rows.append((lab, orig, items))
            if not rows:
                continue
            lines.append(f"### {cid}\n")
            lines.append("| Label | Original surface | Synthetic → count |")
            lines.append("|---|---|---|")
            for lab, orig, items in rows:
                cells = ", ".join(
                    f"`{it['value']}` → {it['count']}"
                    + (" (passthrough)" if it.get("passthrough") else "")
                    for it in items
                )
                lines.append(f"| {lab} | `{orig}` | {cells} |")
            lines.append("")

    # Organization synthesis mapping — one table per org group,
    # character-independent.
    pog = c.get("per_org_group") or {}
    if pog:
        lines.append("## Organization synthesis mapping\n")
        lines.append(
            "For each organization group (an employer pulled from the "
            "characters table), every original surface form referencing "
            "that organization with the synthetic value(s) it mapped to "
            "and counts. `(passthrough)` = left unchanged in the "
            "synthetic text.\n"
        )
        for grp in sorted(pog):
            block = pog[grp]
            counts_map = block.get("original_to_synthetic_counts") or {}
            if not counts_map:
                continue
            lines.append(f"### {grp}\n")
            lines.append("| Original surface | Synthetic → count |")
            lines.append("|---|---|")
            for orig, items in counts_map.items():
                cells = ", ".join(
                    f"`{it['value']}` → {it['count']}"
                    + (" (passthrough)" if it.get("passthrough") else "")
                    for it in items
                )
                lines.append(f"| `{orig}` | {cells} |")
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
    metrics = results["metrics"]
    detail = results["detail"]
    cfg = results["config"]

    def _row_table(headers: List[str], rows: Iterable[List[str]]) -> str:
        h = "".join(f"<th>{_h(c)}</th>" for c in headers)
        body = []
        for r in rows:
            body.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
        return f"<table><thead><tr>{h}</tr></thead><tbody>{''.join(body)}</tbody></table>"

    def _example_rows(examples: Iterable[dict], id_key: str) -> List[List[str]]:
        rows = []
        for ex in examples:
            g = ex["gold"]
            snippet_offset = max(0, g["start"] - 60)
            rows.append([
                _h(ex.get(id_key)),
                _h(g["label"]),
                _h(g["text"]),
                _h(",".join(g.get("characters") or [])),
                _snippet_html(
                    ex.get("snippet", ""),
                    g["start"] - snippet_offset,
                    g["end"] - snippet_offset,
                ),
            ])
        return rows

    # Recall false-negative tab (missed PII)
    fn_tab = _row_table(
        ["row_id", "label", "gold text", "characters", "context"],
        _example_rows(detail["recall"].get("fn_examples", []), "cell_id"),
    )

    # Identity-mapping tab (detected but left unchanged)
    identity_tab = _row_table(
        ["row_id", "label", "gold text", "characters", "context"],
        _example_rows(metrics.get("identity_examples", []), "row_id"),
    )

    # Consistency offenders
    cons_rows = []
    for off in (detail["consistency"].get("inconsistent_buckets") or []):
        counts = off.get("synthetic_value_counts") or []
        rendered = ", ".join(f"{c['value']} ({c['count']})" for c in counts[:10])
        if len(counts) > 10:
            rendered += f", … +{len(counts) - 10} more"
        cons_rows.append([
            _h(off["character"]),
            _h(off["label"]),
            _h(off["original"]),
            _h(off["n_distinct_synthetic"]),
            _h(rendered),
        ])
    cons_tab = _row_table(
        ["character", "label", "original surface", "#synth", "synthetic values (count)"],
        cons_rows,
    )

    # Rule violations: email
    email_rows = []
    for v in detail["realism_rule"]["email_violations"]:
        email_rows.append([
            _h(v["character"]),
            _h(v["synthetic_email"]),
            _h(", ".join(v["synthetic_name_pool"])),
        ])
    email_tab = _row_table(["character", "synthetic email", "synthetic name pool"], email_rows)

    # Rule violations: username
    user_rows = []
    for v in detail["realism_rule"]["username_violations"]:
        user_rows.append([
            _h(v["character"]),
            _h(v["synthetic_username"]),
            _h(v["stripped"]),
            _h(", ".join(v["synthetic_name_pool"])),
        ])
    user_tab = _row_table(
        ["character", "synthetic username", "stripped", "synthetic name pool"],
        user_rows,
    )

    # LLM incoherent (per character × label × surface). Each synthetic
    # value carries its own verdict (✓/✗) and gold-span count.
    llm_rows = []
    for entry in (detail["judge"].get("incoherent_verdicts") or []):
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
        ("Missed PII (NER FN)",        fn_tab),
        ("Unchanged PII (identity)",   identity_tab),
        ("Consistency offenders",      cons_tab),
        ("Email rule violations",      email_tab),
        ("Username rule violations",   user_tab),
        ("LLM incoherent verdicts",    llm_tab),
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
  adapter=<code>{_h((cfg.get('adapter') or cfg.get('format') or '-'))}</code> · tier=<code>{_h(cfg['tier'])}</code>
  · rows=<code>{_h(cfg['n_rows'])}</code> · synthesis=<code>{_h((cfg.get('synthesis') or cfg.get('predictions')))}</code>
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

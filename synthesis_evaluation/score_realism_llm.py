"""Per-entity-type LLM coherence judge.

For each ground-truth character we build the full (ground-truth surface
form → deduplicated synthetic values) mapping, broken down by entity
label. The mapping preserves casing — distinct case-variant surface
forms get their own rows.

We then send Opus 4.7 ONE message per character containing the
mapping, and ask for one coherence verdict PER (entity type, ground-
truth surface form) pair that appears in the mapping, with a nested
judgment per synthetic VALUE inside each pair. So for

    'Megan' → ['Damon']
    'Meg'   → ['Dame', 'Eddie']

the judge returns two NAME_GIVEN verdicts: ('Megan', coherent, with
'Damon' coherent) and ('Meg', incoherent, with 'Dame' coherent and
'Eddie' incoherent).

The two granularities power two metrics:
  - unique precision: one count per (character, label, surface) bucket
    from the surface-level "coherent" field;
  - precision: each synthetic value's verdict weighted by the number
    of gold spans that mapped that surface form to that value.

Output schema (per character):
  {
    "verdicts": [
      {
        "label": "NAME_GIVEN" | "NAME_FAMILY" | "EMAIL_ADDRESS" | "USERNAME",
        "surface": str,           # ground-truth surface form, verbatim
        "coherent": bool,         # the whole surface bucket
        "values": [
          {"value": str, "coherent": bool},
          ...
        ],
        "issues":   [str],
        "confidence": "sure" | "unsure"
      },
      ...
    ]
  }

Concurrent via ThreadPoolExecutor (default 8 workers). Prompt cache on
the system prompt. JSON parse with regex fallback.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import threading
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from .types import EvalRow, GroundTruthSpan, LABELS, SynthEntity

MODEL = "claude-opus-4-7"
MAX_TOKENS = 2048
PROGRESS_EVERY = 8

# Character-level judging covers everything except ORGANIZATION —
# organization coherence is judged per org GROUP, independent of
# characters (an org belongs to many characters at once).
CHAR_LABELS = tuple(l for l in LABELS if l != "ORGANIZATION")

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

SYSTEM_PROMPT = """\
You are auditing a PII-synthesis system. The system replaces real
names, emails, and slack usernames with synthetic values, ideally
keeping the synthetic identity for a single person internally
coherent.

For ONE character at a time you will be shown the ground-truth surface
forms that appeared in the corpus, grouped by entity type, along with
the deduplicated synthetic values the synthesizer mapped each surface
form to. The mapping is shown separately for each entity type. The
entity types are:

  - NAME_GIVEN     (given names like "Megan", "Meg", "MEGAN")
  - NAME_FAMILY    (family names like "Donovan", "DONOVAN")
  - EMAIL_ADDRESS  (full email addresses)
  - USERNAME       (slack mentions, in the form <@UNAMEORI1>)

How to judge
------------

First infer the character's intended synthetic identity from the FULL
mapping (across all entity types shown). The identity is whatever the
synthesizer most consistently mapped the character to — e.g. if
'Megan' → ['Damon'], the synthetic given name is "Damon".

Then judge ONE verdict per (entity type, ground-truth surface form)
pair shown in the input, and inside each verdict judge EVERY synthetic
value separately against that identity. A surface form's verdict is
coherent if and only if ALL of its synthetic values are coherent.

Worked example: given 'Megan' → ['Damon'] and 'Meg' → ['Dame', 'Eddie'],
the identity is "Damon". 'Megan' is coherent ('Damon' coherent).
'Meg' is INCOHERENT: 'Dame' is coherent (it refers to the same
synthetic identity — a nickname-form of Damon) but 'Eddie' is
incoherent (an unrelated given name). 'Meg' → ['Damon'] would have
been coherent too: a value does not need to mirror the surface's
nickname form, it just needs to refer to the same identity.

Per-entity-type rules for judging each synthetic value:

NAME_GIVEN
  - A value is coherent when it refers to the synthetic identity's
    given name in ANY form: the full name, a nickname-form of it, or
    any casing variant. The value's form does NOT need to mirror the
    surface's form — a nickname surface may map to the full synthetic
    name ('Meg' → 'Damon' and 'Meg' → 'Dame' are BOTH coherent), and
    the value's casing does not need to match the surface's casing
    ('MEGAN' → 'Damon' and 'MEGAN' → 'DAMON' are both coherent).
  - A value that is an unrelated given name is INCOHERENT
    ('Meg' → 'Eddie' when the identity is Damon).

NAME_FAMILY
  - A value is coherent when it refers to the identity's one logical
    surname (any casing, form need not mirror the surface). An
    unrelated surname is INCOHERENT.

EMAIL_ADDRESS
  - Each ground-truth email should map to a single synthetic email,
    compared CASE-INSENSITIVELY: synthetic values that differ only in
    casing count as the same email and are all coherent (e.g.
    'megan.donovan@lilly.com' → ['damon.stouds@norvex.com',
    'Damon.Stouds@norvex.com'] is coherent). Genuinely different
    synthetic emails for one ground-truth email means at most one of
    them can be coherent.
  - A coherent value's local-part (everything before '@') should
    plausibly derive from the synthetic given/family names — contain
    at least a ≈4-character contiguous substring of one of them.
  - When an ORGANIZATION CONTEXT block is provided (the character's
    real employer and the synthetic organization it was mapped to),
    a coherent value's DOMAIN must be consistent with the SYNTHETIC
    organization. E.g. real org "Eli Lilly and Company" → synthetic
    org "Norvex Pharmaceuticals" means a real "...@lilly.com" address
    should map to a domain derived from Norvex (e.g. "...@norvex.com"),
    NOT keep the original domain and NOT use an unrelated domain.

USERNAME
  - Each ground-truth slack handle should map to a single synthetic
    handle (1-to-1): if a surface lists multiple synthetic values, at
    most one of them can be coherent.
  - A coherent value's handle body (the bit between <@ and >) should
    plausibly derive from the synthetic given/family names.

Output format
-------------

Output a SINGLE JSON object with this exact shape:

  {
    "verdicts": [
      {
        "label": "NAME_GIVEN" | "NAME_FAMILY" | "EMAIL_ADDRESS" | "USERNAME",
        "surface": "<ground-truth surface form, exactly as shown>",
        "coherent": true | false,
        "values": [
          {"value": "<synthetic value, exactly as shown>", "coherent": true | false},
          ...
        ],
        "issues":   [ "short description of each issue, if any" ],
        "confidence": "sure" | "unsure"
      },
      ...
    ]
  }

Include one verdict per (entity type, surface form) pair that was
shown to you, with one entry in "values" for each synthetic value
listed for that surface form. Repeat "surface" and "value" strings
EXACTLY as shown, including casing. Do not invent verdicts for pairs
that were not shown.

Use "sure" when the case is clear-cut. Use "unsure" when the data is
too thin to be confident (e.g. only one synthetic value with nothing
to compare it against).

Output JSON only. No prose, no markdown fencing.
"""


ORG_SYSTEM_PROMPT = """\
You are auditing a PII-synthesis system's handling of ORGANIZATION
references. All surface forms shown to you reference ONE real-world
organization (e.g. "Eli Lilly and Company", "Lilly", "LLY",
"Lilly Pharmaceuticals" are all the same employer). The synthesizer
should have mapped them all onto ONE coherent synthetic organization.

First infer the intended synthetic organization from the FULL mapping
(whatever the synthesizer most consistently mapped the org to). Then
judge ONE verdict per ground-truth surface form shown, and inside each
verdict judge EVERY synthetic value separately against that synthetic
org. A surface form's verdict is coherent if and only if ALL of its
synthetic values are coherent.

Rules for judging each synthetic value:

  - A value is coherent when it clearly refers to the SAME synthetic
    organization, in ANY form — full name, short form, abbreviation,
    ticker, possessive, or casing variant. The value's form does NOT
    need to mirror the surface form's transformation: with
    "Novo Nordisk" → "Solano Biotech", the surface "novo" mapping to
    "Solano Biotech" is coherent, and "novo" → "solano" is coherent
    too — they all refer to the same synthetic org.
  - MULTIPLE synthetic values for one surface can ALL be coherent
    when each refers to the same synthetic org: "Novo Nordisk" →
    ["Solano Biotech", "Solano Biotech US"] is coherent for both
    values.
  - A value that refers to a DIFFERENT synthetic organization is
    INCOHERENT: given "Novo Nordisk" → ["Solano Biotech",
    "Pacifica Biotech"] and "Novo" → ["Solano"], the values
    "Solano Biotech" and "Solano" are coherent but "Pacifica Biotech"
    is incoherent.

Output ONE JSON object:

  {
    "verdicts": [
      {
        "surface": "<ground-truth surface form, exactly as shown>",
        "coherent": true | false,
        "values": [
          {"value": "<synthetic value, exactly as shown>", "coherent": true | false},
          ...
        ],
        "issues":   [ "short description of each issue, if any" ],
        "confidence": "sure" | "unsure"
      },
      ...
    ]
  }

Include one verdict per surface form shown, with one entry in "values"
for each synthetic value listed for that surface form. Repeat
"surface" and "value" strings EXACTLY as shown, including casing.

Output JSON only. No prose, no markdown fencing.
"""


def _parse_response(text: str) -> Optional[dict]:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n", "", t)
        t = re.sub(r"\n```\s*$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = _JSON_BLOCK_RE.search(t)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _overlaps(a_s: int, a_e: int, b_s: int, b_e: int) -> bool:
    return a_s < b_e and b_s < a_e


def _match_pred(g: GroundTruthSpan, preds: List[SynthEntity]) -> Optional[SynthEntity]:
    best = None
    for p in preds:
        if p.label != g.label:
            continue
        if _overlaps(g.start, g.end, p.start, p.end):
            ov = min(g.end, p.end) - max(g.start, p.start)
            if best is None or ov > best[0]:
                best = (ov, p)
    return best[1] if best else None


def _build_mappings(rows: List[EvalRow]) -> Tuple[
    Dict[str, Dict[str, Dict[str, List[str]]]],
    Dict[str, Dict[str, Dict[str, Dict[str, int]]]],
]:
    """Return ``(mappings, span_counts)``.

    ``mappings`` is ``{character_id: {label: {orig_surface: [synthetic_values]}}}``
    with surface forms case-preserved and synthetic values deduped and sorted.

    ``span_counts`` is ``{character_id: {label: {orig_surface: {synth_value: n}}}}``
    — the number of detected-and-synthesized spans for each
    (character, label, surface, synthetic value) across the corpus.
    Used to weight per-value coherence verdicts into span-level
    precision and recall metrics.
    """
    out: Dict[str, Dict[str, Dict[str, set]]] = defaultdict(
        lambda: {lab: defaultdict(set) for lab in CHAR_LABELS}
    )
    counts: Dict[str, Dict[str, Dict[str, Counter]]] = defaultdict(
        lambda: {lab: defaultdict(Counter) for lab in CHAR_LABELS}
    )
    for row in rows:
        gts = [g for g in row.ground_truth_spans if g.label in CHAR_LABELS]
        preds = [p for p in (row.synthesis.entities or []) if p.label in CHAR_LABELS]
        for g in gts:
            p = _match_pred(g, preds)
            if p is None:
                continue
            for cid in g.characters:
                out[cid][g.label][g.text].add(p.new_text)
                counts[cid][g.label][g.text][p.new_text] += 1

    mappings = {
        cid: {
            lab: {orig: sorted(s) for orig, s in sorted(d.items())}
            for lab, d in by_lab.items()
            if d
        }
        for cid, by_lab in out.items()
    }
    span_counts = {
        cid: {
            lab: {orig: dict(c) for orig, c in d.items()}
            for lab, d in counts[cid].items()
            if d
        }
        for cid in mappings
    }
    return mappings, span_counts


def _build_org_mappings(rows: List[EvalRow]) -> Tuple[
    Dict[str, Dict[str, List[str]]],
    Dict[str, Dict[str, Dict[str, int]]],
]:
    """Return ``(org_mappings, org_span_counts)``.

    ``org_mappings`` is ``{org_group: {orig_surface: [synthetic_values]}}``
    over ORGANIZATION gold spans that carry an ``org_group``.
    ``org_span_counts`` is ``{org_group: {orig_surface: {synth_value: n}}}``
    counting detected-and-synthesized spans per (group, surface, value).
    """
    out: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    counts: Dict[str, Dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for row in rows:
        gts = [g for g in row.ground_truth_spans
               if g.label == "ORGANIZATION" and g.org_group]
        preds = [p for p in (row.synthesis.entities or [])
                 if p.label == "ORGANIZATION"]
        for g in gts:
            p = _match_pred(g, preds)
            if p is None or p.new_text == p.text:
                continue
            out[g.org_group][g.text].add(p.new_text)
            counts[g.org_group][g.text][p.new_text] += 1
    return (
        {grp: {orig: sorted(s) for orig, s in sorted(d.items())}
         for grp, d in out.items()},
        {grp: {orig: dict(c) for orig, c in d.items()}
         for grp, d in counts.items()},
    )


def _build_user_message(
    cid: str,
    mapping: Dict[str, Dict[str, List[str]]],
    org_context: Optional[List[Tuple[str, List[str]]]] = None,
) -> str:
    parts: List[str] = [f"character_id: {cid}\n"]
    if org_context:
        parts.append("ORGANIZATION CONTEXT (real employer → synthetic org "
                     "the synthesizer mapped it to):")
        for real_org, synth_orgs in org_context:
            synth_str = ", ".join(repr(s) for s in synth_orgs) if synth_orgs \
                else "(no synthetic mapping observed)"
            parts.append(f"  {real_org!r} → {synth_str}")
        parts.append("")
    for lab in CHAR_LABELS:
        sub = mapping.get(lab) or {}
        if not sub:
            continue
        parts.append(f"{lab} synthesis mapping (ground truth surface → synthetic values):")
        for orig, synths in sub.items():
            parts.append(f"  {orig!r} → {synths}")
        parts.append("")  # blank line between sections
    parts.append(
        f"Output a JSON object with a 'verdicts' array. Include one "
        f"verdict per (entity type, ground-truth surface form) pair "
        f"shown above (entity types: "
        f"{', '.join(lab for lab in CHAR_LABELS if mapping.get(lab))}"
        f"), with one 'values' entry per synthetic value listed for "
        f"that surface form. No other entity types or surface forms."
    )
    return "\n".join(parts)


def _build_org_user_message(grp: str, mapping: Dict[str, List[str]]) -> str:
    parts = [f"organization: {grp!r}\n",
             "Synthesis mapping (ground truth surface → synthetic values):"]
    for orig, synths in mapping.items():
        parts.append(f"  {orig!r} → {synths}")
    parts.append(
        "\nRespond with the JSON object from the system prompt: one "
        "verdict per surface form shown above, with one 'values' entry "
        "per synthetic value listed for that surface form."
    )
    return "\n".join(parts)


def _call_llm(client, system_prompt: str, user_msg: str) -> Tuple[str, dict]:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_msg}],
        thinking={"type": "adaptive"},
    )
    text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    raw = "".join(text_parts)
    usage = {
        "input_tokens":                getattr(resp.usage, "input_tokens", 0) or 0,
        "output_tokens":               getattr(resp.usage, "output_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens":     getattr(resp.usage, "cache_read_input_tokens", 0)     or 0,
    }
    return raw, usage


def _char_orgs(characters: Optional[dict], cid: str) -> List[str]:
    """Pull a character's organizations from the roster (dataclass or
    raw dict tolerated)."""
    if not characters or cid not in characters:
        return []
    c = characters[cid]
    orgs = getattr(c, "organizations", None)
    if orgs is None and isinstance(c, dict):
        orgs = c.get("organizations")
    return list(orgs or [])


def score(rows: List[EvalRow], *, characters: Optional[dict] = None,
          workers: int = 8) -> dict:
    """Run the LLM judge over every character with synth data (one
    call per character, one verdict per (label, surface) pair), plus
    one call per organization group (one verdict per surface).

    ``characters`` (optional) is the roster — used to inject each
    character's employer organization + its synthetic mapping into the
    character prompt so the judge can check email-domain ↔ synthetic-org
    consistency.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {
            "skipped_reason": "ANTHROPIC_API_KEY not set",
            "per_character": {},
            "per_label_totals": {},
        }

    mappings, span_counts = _build_mappings(rows)
    org_mappings, org_span_counts = _build_org_mappings(rows)
    chars = sorted(mappings)
    org_groups = sorted(org_mappings)
    if not chars and not org_groups:
        return {
            "skipped_reason": "no characters with synthetic values",
            "per_character": {},
            "per_label_totals": {},
            "per_label_span_totals": {},
        }

    import anthropic
    client = anthropic.Anthropic(max_retries=6)

    metrics_lock = threading.Lock()
    totals = {"input_tokens": 0, "output_tokens": 0,
              "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
              "n_calls": 0, "n_parse_failures": 0}

    def _record(usage, parsed):
        with metrics_lock:
            for k in ("input_tokens", "output_tokens",
                      "cache_creation_input_tokens", "cache_read_input_tokens"):
                totals[k] += usage.get(k, 0)
            totals["n_calls"] += 1
            if parsed is None:
                totals["n_parse_failures"] += 1

    def process_one(cid: str) -> Tuple[str, Optional[dict], Optional[str], str]:
        # Organization context: the character's employer(s) and the
        # synthetic org(s) each was mapped to across the corpus.
        org_ctx: List[Tuple[str, List[str]]] = []
        for real_org in _char_orgs(characters, cid):
            synths = sorted({s for d in (org_mappings.get(real_org) or {}).values()
                             for s in d})
            org_ctx.append((real_org, synths))
        user_msg = _build_user_message(cid, mappings[cid],
                                       org_context=org_ctx or None)
        try:
            raw, usage = _call_llm(client, SYSTEM_PROMPT, user_msg)
        except Exception as exc:
            return cid, None, f"{type(exc).__name__}: {exc}", ""
        parsed = _parse_response(raw)
        _record(usage, parsed)
        return cid, parsed, None, raw

    def process_org(grp: str) -> Tuple[str, Optional[dict], Optional[str], str]:
        user_msg = _build_org_user_message(grp, org_mappings[grp])
        try:
            raw, usage = _call_llm(client, ORG_SYSTEM_PROMPT, user_msg)
        except Exception as exc:
            return grp, None, f"{type(exc).__name__}: {exc}", ""
        parsed = _parse_response(raw)
        _record(usage, parsed)
        return grp, parsed, None, raw

    t0 = time.monotonic()
    print(f"  LLM judge: {len(chars)} characters + {len(org_groups)} org groups, "
          f"{workers} workers")
    per_character: Dict[str, dict] = {}
    per_org_group: Dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as exe:
        char_futs = {exe.submit(process_one, cid): ("char", cid) for cid in chars}
        org_futs = {exe.submit(process_org, grp): ("org", grp) for grp in org_groups}
        all_futs = {**char_futs, **org_futs}
        for n, fut in enumerate(concurrent.futures.as_completed(all_futs), 1):
            kind, _ = all_futs[fut]
            key, parsed, err, raw = fut.result()
            if kind == "char":
                per_character[key] = {
                    "mapping": mappings[key],
                    "raw":     raw,
                    "parsed":  parsed,
                    "error":   err,
                }
            else:
                per_org_group[key] = {
                    "mapping":     org_mappings[key],
                    "raw":         raw,
                    "parsed":      parsed,
                    "error":       err,
                    "span_counts": org_span_counts.get(key, {}),
                }
            if n % PROGRESS_EVERY == 0 or n == len(all_futs):
                dt = time.monotonic() - t0
                print(f"    {n}/{len(all_futs)} done in {dt:.0f}s")

    # Tally per-surface-bucket coherence (one count per (character,
    # label, surface) bucket — "unique precision") AND per-label span
    # totals (each synthetic value's verdict weighted by how many gold
    # spans mapped that surface to that value — "precision"/"recall").
    per_label_totals = {lab: {"coherent": 0, "incoherent": 0, "skipped": 0}
                        for lab in CHAR_LABELS}
    per_label_span_totals = {lab: {"coherent_spans": 0,
                                   "incoherent_spans": 0,
                                   "skipped_spans": 0}
                             for lab in CHAR_LABELS}
    incoherent_verdicts: List[dict] = []

    for cid in chars:
        v = per_character[cid]
        parsed = v.get("parsed") or {}
        verdicts = parsed.get("verdicts") if isinstance(parsed, dict) else None
        cid_counts = span_counts.get(cid, {})
        judged: Dict[Tuple[str, str], dict] = {}
        if isinstance(verdicts, list):
            for entry in verdicts:
                if not isinstance(entry, dict):
                    continue
                lab, surf = entry.get("label"), entry.get("surface")
                if lab in CHAR_LABELS and isinstance(surf, str):
                    judged[(lab, surf)] = entry
        # Walk the mapping (not the verdicts) so surface buckets the
        # LLM failed to judge are counted as 'skipped', and invented
        # verdicts for pairs not in the mapping are ignored. Labels
        # absent from the mapping aren't counted at all ("no data").
        for lab, sub in mappings[cid].items():
            for surf, synth_values in sub.items():
                value_counts = (cid_counts.get(lab) or {}).get(surf) or {}
                entry = judged.get((lab, surf))
                if entry is None:
                    per_label_totals[lab]["skipped"] += 1
                    per_label_span_totals[lab]["skipped_spans"] += sum(
                        value_counts.values())
                    continue
                ok = bool(entry.get("coherent"))
                per_label_totals[lab]["coherent" if ok else "incoherent"] += 1
                # Per-value verdicts; a value the judge didn't return
                # inherits the surface-level verdict.
                value_verdicts: Dict[str, bool] = {}
                for vv in (entry.get("values") or []):
                    if isinstance(vv, dict) and isinstance(vv.get("value"), str):
                        value_verdicts[vv["value"]] = bool(vv.get("coherent"))
                values_out = [{"value":    val,
                               "coherent": value_verdicts.get(val, ok),
                               "count":    int(value_counts.get(val, 0) or 0)}
                              for val in synth_values]
                for val_entry in values_out:
                    key = ("coherent_spans" if val_entry["coherent"]
                           else "incoherent_spans")
                    per_label_span_totals[lab][key] += val_entry["count"]
                if not ok:
                    incoherent_verdicts.append({
                        "character":  cid,
                        "label":      lab,
                        "surface":    surf,
                        "coherent":   ok,
                        "confidence": entry.get("confidence"),
                        "issues":     entry.get("issues") or [],
                        "values":     values_out,
                        "mapping":    {surf: synth_values},
                    })

    incoherent_verdicts.sort(
        key=lambda d: (d["character"], d["label"], d["surface"]))

    # Attach per-(character, label) span counts so downstream code
    # (render.py) can drill in if needed.
    for cid in chars:
        per_character[cid]["span_counts"] = span_counts.get(cid, {})

    # ---- Org-group tallies (character-independent) ----
    # Buckets are (org_group, surface); spans weighted per value, same
    # as the character labels.
    org_group_totals = {"coherent": 0, "incoherent": 0, "skipped": 0}
    org_group_span_totals = {"coherent_spans": 0, "incoherent_spans": 0,
                             "skipped_spans": 0}
    org_incoherent_verdicts: List[dict] = []
    for grp in org_groups:
        v = per_org_group[grp]
        parsed = v.get("parsed")
        grp_counts = org_span_counts.get(grp) or {}
        verdicts = parsed.get("verdicts") if isinstance(parsed, dict) else None
        judged: Dict[str, dict] = {}
        if isinstance(verdicts, list):
            for entry in verdicts:
                if isinstance(entry, dict) and isinstance(entry.get("surface"), str):
                    judged[entry["surface"]] = entry
        for surf, synth_values in org_mappings[grp].items():
            value_counts = grp_counts.get(surf) or {}
            entry = judged.get(surf)
            if entry is None:
                org_group_totals["skipped"] += 1
                org_group_span_totals["skipped_spans"] += sum(
                    value_counts.values())
                continue
            ok = bool(entry.get("coherent"))
            org_group_totals["coherent" if ok else "incoherent"] += 1
            value_verdicts: Dict[str, bool] = {}
            for vv in (entry.get("values") or []):
                if isinstance(vv, dict) and isinstance(vv.get("value"), str):
                    value_verdicts[vv["value"]] = bool(vv.get("coherent"))
            values_out = [{"value":    val,
                           "coherent": value_verdicts.get(val, ok),
                           "count":    int(value_counts.get(val, 0) or 0)}
                          for val in synth_values]
            for val_entry in values_out:
                key = ("coherent_spans" if val_entry["coherent"]
                       else "incoherent_spans")
                org_group_span_totals[key] += val_entry["count"]
            if not ok:
                org_incoherent_verdicts.append({
                    "org_group":  grp,
                    "surface":    surf,
                    "confidence": entry.get("confidence"),
                    "issues":     entry.get("issues") or [],
                    "values":     values_out,
                    "mapping":    {surf: synth_values},
                })

    return {
        "model": MODEL,
        "usage": {
            "n_calls":          totals["n_calls"],
            "n_parse_failures": totals["n_parse_failures"],
            "input_tokens":     totals["input_tokens"],
            "output_tokens":    totals["output_tokens"],
            "cache_read_input_tokens":     totals["cache_read_input_tokens"],
            "cache_creation_input_tokens": totals["cache_creation_input_tokens"],
        },
        "per_label_totals":      per_label_totals,
        "per_label_span_totals": per_label_span_totals,
        "incoherent_verdicts":   incoherent_verdicts,
        "per_character":         per_character,
        "org_group_totals":      org_group_totals,
        "org_group_span_totals": org_group_span_totals,
        "org_incoherent_verdicts": org_incoherent_verdicts,
        "per_org_group":         per_org_group,
    }

"""Per-owner LLM coherence judge.

Every ground-truth span belongs to an owner: a character (``characters``) or an
organization group (``org_group``; ORGANIZATION spans always, and any other span
that carries an org_group and no characters). For each owner we build the full
(ground-truth surface form → deduplicated synthetic values) mapping, broken down
by entity label, over everything that owner owns — the original person labels
(names, emails, usernames) and the newer ones (phone numbers, postal addresses,
employee ids, account numbers, URLs) alike. The mapping preserves casing —
distinct case-variant surface forms get their own rows.

We then send the judge ONE message per owner containing the mapping, and ask
for one coherence verdict PER (entity label, ground-truth surface form) pair
that appears in the mapping, with a nested judgment per synthetic VALUE inside
each pair. So for

    'Megan' → ['Damon']
    'Meg'   → ['Dame', 'Eddie']

the judge returns two NAME_GIVEN verdicts: ('Megan', coherent, with 'Damon'
coherent) and ('Meg', incoherent, with 'Dame' coherent and 'Eddie' incoherent).

The two granularities power two metrics:
  - unique precision: one count per (owner, label, surface) bucket from the
    surface-level "coherent" field;
  - precision: each synthetic value's verdict weighted by the number of gold
    spans that mapped that surface form to that value.

Output schema (per owner):
  {
    "verdicts": [
      {
        "label": <one of the entity labels>,
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

Spans with no owner at all (PII of people and organizations outside the
roster) are judged in label-sectioned batches with a generic "plausible
replacement of the same type" rubric, so they can score instead of sitting in
the synthesis-accuracy denominator unjudged.

Concurrent via ThreadPoolExecutor (default 8 workers). Prompt cache on the
system prompt. JSON parse with regex fallback.
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

from .score_recall import _match_pred
from .types import EvalRow, LABELS, ORG_LABELS, PERSON_LABELS

MODEL = "claude-opus-4-7"
MAX_TOKENS = 12_000          # adaptive thinking shares this budget with the verdict list
PROGRESS_EVERY = 8

# Character-level judging covers every label a character can own; ORGANIZATION
# is judged per org GROUP, together with the addresses, phones, URLs and
# accounts that belong to the organization rather than to a person.
CHAR_LABELS = PERSON_LABELS

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

_SHARED_RULES = """\
PHONE_NUMBER
  - Each ground-truth phone number should map to ONE synthetic phone number
    (compare ignoring spacing and punctuation): the same real number seen in
    several places must always become the same synthetic number, and two
    different real numbers must not collapse onto one synthetic number.
  - A coherent value is a plausible phone number of the same general shape
    (an extension stays an extension, an international prefix stays
    international) and is not the original number, not a digit-shuffled
    near-copy of it, and not a real-looking number that merely differs in
    one or two digits from the original.

LOCATION_ADDRESS
  - Each ground-truth address should map to ONE synthetic postal address;
    partial forms of the same real address (a street line alone, or the
    street line plus its city/state/ZIP block) should map to matching
    partial forms of the same synthetic address.
  - A coherent value is a plausible postal address (street number and
    street, and city/state/ZIP when the original had them), is not the
    original or a lightly edited copy of it (same street name with a new
    number does NOT count), and keeps the owner's synthetic identity
    consistent: an organization's office address stays an office-style
    address; a person's home address stays residential.

EMPLOYEE_ID and ACCOUNT_NUMBER
  - Each ground-truth id should map to ONE synthetic id (1-to-1 and
    consistent everywhere it appears), and two different real ids must not
    map onto the same synthetic id.
  - A coherent value preserves the shape of the original (same prefix
    pattern, length and separators — LLY-40718 → LLY-58231, not 40718 or
    a phone number) and is not the original.
  - Masked forms (****-****-7734) count as coherent when the visible digits
    changed and the masking pattern was kept.

URL
  - A LinkedIn profile URL (linkedin.com/in/<slug>) is coherent when the
    slug plausibly derives from the synthetic given/family names of the
    person (it must not keep the original slug).
  - A company web address or bare domain is coherent when the domain
    plausibly derives from the SYNTHETIC organization (as with email
    domains: real "marriott.com" under synthetic org "Harborline Hotels"
    should become something like "harborline.com"); keeping the original
    domain is incoherent, an unrelated domain is incoherent.
  - Paths may be kept or changed; the judgment is about the identifying
    host / slug part.
"""

SYSTEM_PROMPT = """\
You are auditing a PII-synthesis system. The system replaces real names,
emails, slack usernames, phone numbers, postal addresses, employee ids,
account numbers and URLs with synthetic values, ideally keeping the
synthetic identity for a single person internally coherent.

For ONE character at a time you will be shown the ground-truth surface
forms that appeared in the corpus, grouped by entity type, along with
the deduplicated synthetic values the synthesizer mapped each surface
form to. The mapping is shown separately for each entity type. The
entity types are:

  - NAME_GIVEN        (given names like "Megan", "Meg", "MEGAN")
  - NAME_FAMILY       (family names like "Donovan", "DONOVAN")
  - EMAIL_ADDRESS     (full email addresses)
  - USERNAME          (slack mentions, in the form <@UNAMEORI1>, and login handles)
  - PHONE_NUMBER      (the person's direct, mobile or desk numbers)
  - LOCATION_ADDRESS  (the person's home or mailing addresses)
  - EMPLOYEE_ID       (employee / badge ids that identify the person)
  - ACCOUNT_NUMBER    (bank, brokerage, member or policy numbers held by the person)
  - URL               (the person's LinkedIn or other personal profile URLs)

Only the entity types present in the input need verdicts.

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

""" + _SHARED_RULES + """
Output format
-------------

Output a SINGLE JSON object with this exact shape:

  {
    "verdicts": [
      {
        "label": "<entity type exactly as shown in the input section header>",
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
that were not shown. Keep "issues" to one short sentence each.

Use "sure" when the case is clear-cut. Use "unsure" when the data is
too thin to be confident (e.g. only one synthetic value with nothing
to compare it against).

Output JSON only. No prose, no markdown fencing.
"""


ORG_SYSTEM_PROMPT = """\
You are auditing a PII-synthesis system's handling of ONE real-world
organization. You will be shown every ground-truth surface form owned by
that organization, grouped by entity type, with the deduplicated synthetic
values the synthesizer mapped each to. The entity types are:

  - ORGANIZATION      (the organization's names: "Eli Lilly and Company",
                       "Lilly", "LLY", "Lilly Pharmaceuticals" are all the
                       same employer)
  - LOCATION_ADDRESS  (its office, branch, store or campus addresses)
  - PHONE_NUMBER      (its main, front-desk, hotline or fax numbers)
  - URL               (its web addresses, bare domains, SharePoint/Box tenants)
  - ACCOUNT_NUMBER    (account, customer or policy numbers the organization holds)

Only the entity types present in the input need verdicts.

First infer the intended synthetic organization from the FULL mapping
(whatever the synthesizer most consistently mapped the org names to).
Then judge ONE verdict per (entity type, ground-truth surface form)
shown, and inside each verdict judge EVERY synthetic value separately
against that synthetic organization. A surface form's verdict is
coherent if and only if ALL of its synthetic values are coherent.

ORGANIZATION
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

""" + _SHARED_RULES + """
Output ONE JSON object:

  {
    "verdicts": [
      {
        "label": "<entity type exactly as shown in the input section header>",
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

Include one verdict per (entity type, surface form) shown, with one
entry in "values" for each synthetic value listed for that surface
form. Repeat "surface" and "value" strings EXACTLY as shown, including
casing. Keep "issues" to one short sentence each.

Output JSON only. No prose, no markdown fencing.
"""


UNOWNED_SYSTEM_PROMPT = """\
You are auditing a PII-synthesis system. The surface forms shown to you are
PII of people and organizations OUTSIDE the corpus roster (a customer's
clinic address, a vendor's phone number, an external candidate's employee id,
a third-party company name), so there is no known identity to check them
against. Judge each (entity type, ground-truth surface form) pair on its own
terms: is every synthetic value a plausible replacement of the same type that
protects the original?

Rules for judging each synthetic value:
  - It must not be the original value, a casing variant of it, or a lightly
    edited copy (same street with a new number, same id with one digit
    changed, same name with a different spelling).
  - It must be a plausible instance of the same entity type: a name stays a
    name of the same kind (given/family), an organization stays an
    organization name, an email stays an email, a phone stays a phone of the
    same shape (valid-looking country/area structure, an extension stays an
    extension), an address stays a postal address with the same level of
    detail, an id keeps the original's prefix pattern, length and
    separators, a URL keeps a plausible host or profile slug.
  - One real value should map to ONE synthetic value (compare emails and
    phones ignoring casing / punctuation); several genuinely different
    synthetic values for one real value means at most one of them is
    coherent.

""" + _SHARED_RULES + """
Output ONE JSON object:

  {
    "verdicts": [
      {
        "label": "<entity type exactly as shown in the input section header>",
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

Include one verdict per (entity type, surface form) shown, with one entry in
"values" for each synthetic value listed for that surface form. Repeat
"surface" and "value" strings EXACTLY as shown. Keep "issues" to one short
sentence each. Output JSON only. No prose, no markdown fencing.
"""

UNOWNED_BATCH = 50


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


# The label-matched span matcher lives in score_recall (the single
# matching rule shared by detection, this judge join, and metrics.py).


def _build_mappings(rows: List[EvalRow]) -> Tuple[
    Dict[str, Dict[str, Dict[str, List[str]]]],
    Dict[str, Dict[str, Dict[str, Dict[str, int]]]],
]:
    """Return ``(mappings, span_counts)`` for character-owned spans.

    ``mappings`` is ``{character_id: {label: {orig_surface: [synthetic_values]}}}``
    with surface forms case-preserved and synthetic values deduped and sorted.

    ``span_counts`` is ``{character_id: {label: {orig_surface: {synth_value: n}}}}``
    — the number of detected-and-synthesized spans for each
    (character, label, surface, synthetic value) across the corpus.
    """
    out: Dict[str, Dict[str, Dict[str, set]]] = defaultdict(
        lambda: {lab: defaultdict(set) for lab in PERSON_LABELS}
    )
    counts: Dict[str, Dict[str, Dict[str, Counter]]] = defaultdict(
        lambda: {lab: defaultdict(Counter) for lab in PERSON_LABELS}
    )
    for row in rows:
        gts = [g for g in row.ground_truth_spans if g.owner == "person" and g.label in PERSON_LABELS]
        preds = list(row.synthesis.entities or [])
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
    Dict[str, Dict[str, Dict[str, List[str]]]],
    Dict[str, Dict[str, Dict[str, Dict[str, int]]]],
]:
    """Return ``(org_mappings, org_span_counts)`` for org-owned spans.

    ``org_mappings`` is ``{org_group: {label: {orig_surface: [synthetic_values]}}}``
    over gold spans owned by an org group (ORGANIZATION spans with an
    ``org_group``, plus addresses / phones / URLs / accounts that carry an
    org_group and no characters). ``org_span_counts`` mirrors it with
    ``{synth_value: n}`` per surface.
    """
    out: Dict[str, Dict[str, Dict[str, set]]] = defaultdict(
        lambda: {lab: defaultdict(set) for lab in ORG_LABELS}
    )
    counts: Dict[str, Dict[str, Dict[str, Counter]]] = defaultdict(
        lambda: {lab: defaultdict(Counter) for lab in ORG_LABELS}
    )
    for row in rows:
        gts = [g for g in row.ground_truth_spans if g.owner == "org" and g.label in ORG_LABELS]
        preds = list(row.synthesis.entities or [])
        for g in gts:
            p = _match_pred(g, preds)
            if p is None or p.new_text == p.text:
                continue
            out[g.org_group][g.label][g.text].add(p.new_text)
            counts[g.org_group][g.label][g.text][p.new_text] += 1
    org_mappings = {
        grp: {lab: {orig: sorted(s) for orig, s in sorted(d.items())} for lab, d in by_lab.items() if d}
        for grp, by_lab in out.items()
    }
    org_mappings = {grp: m for grp, m in org_mappings.items() if m}
    org_span_counts = {
        grp: {lab: {orig: dict(c) for orig, c in d.items()} for lab, d in counts[grp].items() if d}
        for grp in org_mappings
    }
    return org_mappings, org_span_counts


def _build_unowned_mappings(rows: List[EvalRow]) -> Tuple[
    Dict[str, Dict[str, List[str]]],
    Dict[str, Dict[str, Dict[str, int]]],
]:
    """``({label: {orig_surface: [synthetic_values]}}, {label: {orig_surface: {value: n}}})`` over gold
    spans with no owner (no characters, no org_group)."""
    out: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    counts: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for row in rows:
        gts = [g for g in row.ground_truth_spans if g.owner is None and g.label in LABELS]
        preds = list(row.synthesis.entities or [])
        for g in gts:
            p = _match_pred(g, preds)
            if p is None:
                continue
            out[g.label][g.text].add(p.new_text)
            counts[g.label][g.text][p.new_text] += 1
    return ({lab: {orig: sorted(v) for orig, v in sorted(d.items())} for lab, d in out.items()},
            {lab: {orig: dict(c) for orig, c in d.items()} for lab, d in counts.items()})


def _unowned_batches(mapping: Dict[str, Dict[str, List[str]]]) -> List[Dict[str, Dict[str, List[str]]]]:
    """Split the unowned mapping into label-sectioned batches of at most UNOWNED_BATCH surfaces."""
    batches: List[Dict[str, Dict[str, List[str]]]] = []
    cur: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    n = 0
    for lab in LABELS:
        for surf, vals in (mapping.get(lab) or {}).items():
            cur[lab][surf] = vals
            n += 1
            if n >= UNOWNED_BATCH:
                batches.append(dict(cur))
                cur, n = defaultdict(dict), 0
    if n:
        batches.append(dict(cur))
    return batches


def _build_unowned_user_message(mapping: Dict[str, Dict[str, List[str]]]) -> str:
    parts = ["Surface forms owned by people and organizations outside the roster.\n"]
    for lab in LABELS:
        sub = mapping.get(lab) or {}
        if not sub:
            continue
        parts.append(f"{lab} synthesis mapping (ground truth surface → synthetic values):")
        for orig, synths in sub.items():
            parts.append(f"  {orig!r} → {synths}")
        parts.append("")
    parts.append("Respond with the JSON object from the system prompt: one verdict per (entity type, surface form) "
                 "shown above, with one 'values' entry per synthetic value listed for that surface form.")
    return "\n".join(parts)


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
    for lab in PERSON_LABELS:
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
        f"{', '.join(lab for lab in PERSON_LABELS if mapping.get(lab))}"
        f"), with one 'values' entry per synthetic value listed for "
        f"that surface form. No other entity types or surface forms."
    )
    return "\n".join(parts)


def _build_org_user_message(grp: str, mapping: Dict[str, Dict[str, List[str]]]) -> str:
    parts = [f"organization: {grp!r}\n"]
    for lab in ORG_LABELS:
        sub = mapping.get(lab) or {}
        if not sub:
            continue
        parts.append(f"{lab} synthesis mapping (ground truth surface → synthetic values):")
        for orig, synths in sub.items():
            parts.append(f"  {orig!r} → {synths}")
        parts.append("")
    parts.append(
        f"Respond with the JSON object from the system prompt: one verdict per "
        f"(entity type, surface form) shown above (entity types: "
        f"{', '.join(lab for lab in ORG_LABELS if mapping.get(lab))}), with one "
        f"'values' entry per synthetic value listed for that surface form."
    )
    return "\n".join(parts)


def _call_llm(client, system_prompt: str, user_msg: str) -> Tuple[str, dict]:
    with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_msg}],
        thinking={"type": "adaptive"},
    ) as stream:
        resp = stream.get_final_message()
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


def _verdict_label(entry: dict, default: str) -> str:
    """Verdict entries carry a label; older org verdicts did not (they were all ORGANIZATION)."""
    lab = entry.get("label")
    return lab if isinstance(lab, str) and lab in LABELS else default


def _tally(mapping: Dict[str, Dict[str, List[str]]], counts: Dict[str, Dict[str, Dict[str, int]]],
           judged: Dict[Tuple[str, str], dict], totals: Dict[str, dict], span_totals: Dict[str, dict],
           owner_key: str, owner_field: str, incoherent_out: List[dict]) -> None:
    """Walk the mapping (not the verdicts) so surface buckets the LLM failed to judge count as
    'skipped', and invented verdicts for pairs not in the mapping are ignored."""
    for lab, sub in mapping.items():
        for surf, synth_values in sub.items():
            value_counts = (counts.get(lab) or {}).get(surf) or {}
            entry = judged.get((lab, surf))
            if entry is None:
                totals[lab]["skipped"] += 1
                span_totals[lab]["skipped_spans"] += sum(value_counts.values())
                continue
            ok = bool(entry.get("coherent"))
            totals[lab]["coherent" if ok else "incoherent"] += 1
            value_verdicts: Dict[str, bool] = {}
            for vv in (entry.get("values") or []):
                if isinstance(vv, dict) and isinstance(vv.get("value"), str):
                    value_verdicts[vv["value"]] = bool(vv.get("coherent"))
            values_out = [{"value": val, "coherent": value_verdicts.get(val, ok),
                           "count": int(value_counts.get(val, 0) or 0)} for val in synth_values]
            for val_entry in values_out:
                key = "coherent_spans" if val_entry["coherent"] else "incoherent_spans"
                span_totals[lab][key] += val_entry["count"]
            if not ok:
                incoherent_out.append({
                    owner_field: owner_key,
                    "label":      lab,
                    "surface":    surf,
                    "coherent":   ok,
                    "confidence": entry.get("confidence"),
                    "issues":     entry.get("issues") or [],
                    "values":     values_out,
                    "mapping":    {surf: synth_values},
                })


def score(rows: List[EvalRow], *, characters: Optional[dict] = None,
          workers: int = 8) -> dict:
    """Run the LLM judge over every owner with synth data: one call per
    character (one verdict per (label, surface) pair over everything the
    character owns) and one call per organization group (one verdict per
    (label, surface) pair over everything the organization owns).

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
    unowned_mapping, unowned_counts = _build_unowned_mappings(rows)
    unowned_batches = _unowned_batches(unowned_mapping)
    chars = sorted(mappings)
    org_groups = sorted(org_mappings)
    if not chars and not org_groups and not unowned_batches:
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
            synths = sorted({s for d in ((org_mappings.get(real_org) or {}).get("ORGANIZATION") or {}).values()
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
        # A dominant org (a corpus protagonist's employer) can carry more surface buckets than
        # one MAX_TOKENS response can hold — the verdict JSON truncates mid-list and the whole
        # group used to count as skipped. Chunk the group's buckets across calls and merge.
        mapping = org_mappings[grp]
        flat = [(lab, surf) for lab in mapping for surf in mapping[lab]]
        chunk_size = 25
        if len(flat) <= chunk_size:
            chunks = [mapping]
        else:
            chunks = []
            for start in range(0, len(flat), chunk_size):
                piece: Dict[str, Dict[str, List[str]]] = {}
                for lab, surf in flat[start:start + chunk_size]:
                    piece.setdefault(lab, {})[surf] = mapping[lab][surf]
                chunks.append(piece)
        verdicts: List[dict] = []
        raws: List[str] = []
        first_error: Optional[str] = None
        for piece in chunks:
            user_msg = _build_org_user_message(grp, piece)
            try:
                raw, usage = _call_llm(client, ORG_SYSTEM_PROMPT, user_msg)
            except Exception as exc:
                first_error = first_error or f"{type(exc).__name__}: {exc}"
                continue
            parsed = _parse_response(raw)
            _record(usage, parsed)
            raws.append(raw)
            if parsed and isinstance(parsed.get("verdicts"), list):
                verdicts.extend(parsed["verdicts"])
        if not verdicts:
            return grp, None, first_error or "no chunk parsed", "\n".join(raws)
        return grp, {"verdicts": verdicts}, None, "\n".join(raws)

    def process_unowned(i: int) -> Tuple[str, Optional[dict], Optional[str], str]:
        user_msg = _build_unowned_user_message(unowned_batches[i])
        try:
            raw, usage = _call_llm(client, UNOWNED_SYSTEM_PROMPT, user_msg)
        except Exception as exc:
            return f"batch_{i}", None, f"{type(exc).__name__}: {exc}", ""
        parsed = _parse_response(raw)
        _record(usage, parsed)
        return f"batch_{i}", parsed, None, raw

    t0 = time.monotonic()
    print(f"  LLM judge: {len(chars)} characters + {len(org_groups)} org groups + "
          f"{len(unowned_batches)} unowned batch(es), {workers} workers")
    per_character: Dict[str, dict] = {}
    per_org_group: Dict[str, dict] = {}
    per_unowned: Dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as exe:
        char_futs = {exe.submit(process_one, cid): ("char", cid) for cid in chars}
        org_futs = {exe.submit(process_org, grp): ("org", grp) for grp in org_groups}
        un_futs = {exe.submit(process_unowned, i): ("unowned", i) for i in range(len(unowned_batches))}
        all_futs = {**char_futs, **org_futs, **un_futs}
        for n, fut in enumerate(concurrent.futures.as_completed(all_futs), 1):
            kind, _ = all_futs[fut]
            key, parsed, err, raw = fut.result()
            if kind == "unowned":
                per_unowned[key] = {"mapping": unowned_batches[int(key.split("_")[1])], "raw": raw, "parsed": parsed, "error": err}
            elif kind == "char":
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

    # ---- Character tallies: one count per (character, label, surface) bucket
    # ("unique precision") and per-label span totals weighted by how many gold
    # spans mapped that surface to that value ("precision"/"recall").
    per_label_totals = {lab: {"coherent": 0, "incoherent": 0, "skipped": 0} for lab in PERSON_LABELS}
    per_label_span_totals = {lab: {"coherent_spans": 0, "incoherent_spans": 0, "skipped_spans": 0}
                             for lab in PERSON_LABELS}
    incoherent_verdicts: List[dict] = []
    for cid in chars:
        parsed = per_character[cid].get("parsed") or {}
        verdicts = parsed.get("verdicts") if isinstance(parsed, dict) else None
        judged: Dict[Tuple[str, str], dict] = {}
        if isinstance(verdicts, list):
            for entry in verdicts:
                if isinstance(entry, dict) and isinstance(entry.get("surface"), str):
                    lab = _verdict_label(entry, "")
                    if lab in PERSON_LABELS:
                        judged[(lab, entry["surface"])] = entry
        _tally(mappings[cid], span_counts.get(cid, {}), judged, per_label_totals, per_label_span_totals,
               cid, "character", incoherent_verdicts)
        per_character[cid]["span_counts"] = span_counts.get(cid, {})
    incoherent_verdicts.sort(key=lambda d: (d["character"], d["label"], d["surface"]))

    # ---- Org-group tallies (character-independent), per label plus the
    # aggregate the summary has always shown.
    org_by_label_totals = {lab: {"coherent": 0, "incoherent": 0, "skipped": 0} for lab in ORG_LABELS}
    org_by_label_span_totals = {lab: {"coherent_spans": 0, "incoherent_spans": 0, "skipped_spans": 0}
                                for lab in ORG_LABELS}
    org_incoherent_verdicts: List[dict] = []
    for grp in org_groups:
        parsed = per_org_group[grp].get("parsed")
        verdicts = parsed.get("verdicts") if isinstance(parsed, dict) else None
        judged: Dict[Tuple[str, str], dict] = {}
        if isinstance(verdicts, list):
            for entry in verdicts:
                if isinstance(entry, dict) and isinstance(entry.get("surface"), str):
                    lab = _verdict_label(entry, "ORGANIZATION")
                    if lab in ORG_LABELS:
                        judged[(lab, entry["surface"])] = entry
        _tally(org_mappings[grp], org_span_counts.get(grp, {}), judged, org_by_label_totals,
               org_by_label_span_totals, grp, "org_group", org_incoherent_verdicts)
    # ---- Unowned tallies (spans of people / orgs outside the roster), per label.
    unowned_by_label_totals = {lab: {"coherent": 0, "incoherent": 0, "skipped": 0} for lab in LABELS}
    unowned_by_label_span_totals = {lab: {"coherent_spans": 0, "incoherent_spans": 0, "skipped_spans": 0} for lab in LABELS}
    unowned_incoherent_verdicts: List[dict] = []
    for key, blk in per_unowned.items():
        parsed = blk.get("parsed")
        verdicts = parsed.get("verdicts") if isinstance(parsed, dict) else None
        judged: Dict[Tuple[str, str], dict] = {}
        if isinstance(verdicts, list):
            for entry in verdicts:
                if isinstance(entry, dict) and isinstance(entry.get("surface"), str):
                    lab = _verdict_label(entry, "")
                    if lab in LABELS:
                        judged[(lab, entry["surface"])] = entry
        batch_counts = {lab: {surf: (unowned_counts.get(lab) or {}).get(surf, {}) for surf in sub}
                        for lab, sub in blk["mapping"].items()}
        _tally(blk["mapping"], batch_counts, judged, unowned_by_label_totals, unowned_by_label_span_totals,
               key, "unowned_batch", unowned_incoherent_verdicts)
    org_group_totals = {k: sum(v[k] for v in org_by_label_totals.values()) for k in ("coherent", "incoherent", "skipped")}
    org_group_span_totals = {k: sum(v[k] for v in org_by_label_span_totals.values())
                             for k in ("coherent_spans", "incoherent_spans", "skipped_spans")}

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
        "org_by_label_totals":   org_by_label_totals,
        "org_by_label_span_totals": org_by_label_span_totals,
        "org_incoherent_verdicts": org_incoherent_verdicts,
        "per_org_group":         per_org_group,
        "unowned_by_label_totals": unowned_by_label_totals,
        "unowned_by_label_span_totals": unowned_by_label_span_totals,
        "unowned_incoherent_verdicts": unowned_incoherent_verdicts,
        "per_unowned":           per_unowned,
    }

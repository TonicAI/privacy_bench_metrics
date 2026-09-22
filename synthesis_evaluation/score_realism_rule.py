"""Rule-based realism checks.

For each ground-truth character we collect the deduplicated synthetic
NAME_GIVEN / NAME_FAMILY / EMAIL_ADDRESS / USERNAME values attributed
to that character (via overlap match against the ground-truth spans).

EMAIL_ADDRESS rule
------------------
The local-part (everything before the first '@') of each synthetic
email must contain a contiguous substring of length ≥ MIN_OVERLAP_LEN
(default 4) drawn from at least one of the character's synthetic first
or last names (case-insensitive). Pass / fail per email.

USERNAME rule
-------------
Same ≥ MIN_OVERLAP_LEN substring rule. We strip the slack-mention
wrapper `<@…>` and any trailing digit suffix (e.g. `<@UMEGANDON1>` →
`MEGANDON`) before checking.

Nicknames
---------
Explicitly deferred. We surface a soft flag when a synthetic NAME_GIVEN
value is neither a substring of nor a substring-target of any other
NAME_GIVEN value for the same character — that catches cases where the
synthesizer mapped "Meg" and "Megan" to unrelated synthetic strings.

Outputs include per-character pass/fail counts and a list of violation
rows for the failure viewer.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .types import EvalRow, GroundTruthSpan, LABELS, SynthEntity, in_scope, labels_match

MIN_OVERLAP_LEN = 4
TOP_K_VIOLATIONS = 50

_USERNAME_STRIP = re.compile(r"^<@(.+?)>$")


def _overlaps(a_s: int, a_e: int, b_s: int, b_e: int) -> bool:
    return a_s < b_e and b_s < a_e


def _match_pred(g: GroundTruthSpan, preds: List[SynthEntity]) -> Optional[SynthEntity]:
    best = None
    for p in preds:
        if not labels_match(g.label, p.label):
            continue
        if _overlaps(g.start, g.end, p.start, p.end):
            ov = min(g.end, p.end) - max(g.start, p.start)
            if best is None or ov > best[0]:
                best = (ov, p)
    return best[1] if best else None


def _contains_chunk(haystack: str, needles: Iterable[str], min_len: int) -> Optional[str]:
    """Return any needle that has a ≥min_len contiguous substring in haystack.

    Searches each needle's contiguous substrings of length min_len. As
    long as any substring appears in haystack, returns that needle.
    """
    h = haystack.lower()
    for n in needles:
        if not n:
            continue
        n_lower = n.lower()
        if len(n_lower) < min_len:
            # An overall name shorter than the min — accept the whole
            # name appearing if it does.
            if n_lower and n_lower in h:
                return n
            continue
        for i in range(0, len(n_lower) - min_len + 1):
            if n_lower[i:i + min_len] in h:
                return n
    return None


def _strip_username(u: str) -> str:
    """Strip the `<@…>` wrapper and a trailing digit suffix from a
    slack mention's synthetic form so the rule can compare it against
    name tokens.
    """
    m = _USERNAME_STRIP.match(u.strip())
    body = m.group(1) if m else u
    return re.sub(r"\d+$", "", body)


def score(rows: List[EvalRow]) -> dict:
    """Run the realism rule checks across all rows."""
    # For each character, collect synthetic first names + last names +
    # emails + usernames seen across the corpus.
    synth_first: Dict[str, Set[str]] = defaultdict(set)
    synth_last:  Dict[str, Set[str]] = defaultdict(set)
    synth_email: Dict[str, Set[str]] = defaultdict(set)
    synth_user:  Dict[str, Set[str]] = defaultdict(set)

    # Track sample row indices per (cid, label, synth_value) for offenders.
    synth_samples: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)

    for row_idx, row in enumerate(rows):
        gts = [g for g in row.ground_truth_spans if g.label in LABELS]
        preds = [p for p in (row.synthesis.entities or []) if in_scope(p.label)]
        for g in gts:
            p = _match_pred(g, preds)
            if p is None:
                continue
            for cid in g.characters:
                if p.label == "NAME_GIVEN":
                    synth_first[cid].add(p.new_text)
                elif p.label == "NAME_FAMILY":
                    synth_last[cid].add(p.new_text)
                elif p.label == "EMAIL_ADDRESS":
                    synth_email[cid].add(p.new_text)
                elif p.label == "USERNAME":
                    synth_user[cid].add(p.new_text)
                synth_samples[(cid, p.label, p.new_text)].append(row_idx)

    # Run the checks per character.
    per_character: Dict[str, dict] = {}
    email_violations: List[dict] = []
    user_violations: List[dict] = []

    chars = sorted(set(synth_first) | set(synth_last)
                   | set(synth_email) | set(synth_user))

    n_email_total = n_email_pass = 0
    n_user_total = n_user_pass = 0

    for cid in chars:
        names_pool = synth_first.get(cid, set()) | synth_last.get(cid, set())
        emails = synth_email.get(cid, set())
        users  = synth_user.get(cid, set())

        # Email rule
        email_pass = email_fail = 0
        for e in emails:
            local = e.split("@", 1)[0]
            hit = _contains_chunk(local, names_pool, MIN_OVERLAP_LEN) if names_pool else None
            if hit is not None:
                email_pass += 1
            else:
                email_fail += 1
                email_violations.append({
                    "character": cid,
                    "synthetic_email": e,
                    "synthetic_name_pool": sorted(names_pool),
                    "sample_row_idxs": synth_samples[(cid, "EMAIL_ADDRESS", e)][:5],
                })
        # Username rule
        user_pass = user_fail = 0
        for u in users:
            body = _strip_username(u)
            hit = _contains_chunk(body, names_pool, MIN_OVERLAP_LEN) if names_pool else None
            if hit is not None:
                user_pass += 1
            else:
                user_fail += 1
                user_violations.append({
                    "character": cid,
                    "synthetic_username": u,
                    "stripped": body,
                    "synthetic_name_pool": sorted(names_pool),
                    "sample_row_idxs": synth_samples[(cid, "USERNAME", u)][:5],
                })

        per_character[cid] = {
            "synth_first_names": sorted(synth_first.get(cid, set())),
            "synth_last_names":  sorted(synth_last.get(cid, set())),
            "synth_emails":      sorted(emails),
            "synth_usernames":   sorted(users),
            "email_rule": {
                "total": email_pass + email_fail,
                "pass": email_pass,
                "fail": email_fail,
            },
            "username_rule": {
                "total": user_pass + user_fail,
                "pass": user_pass,
                "fail": user_fail,
            },
        }
        n_email_total += email_pass + email_fail
        n_email_pass  += email_pass
        n_user_total  += user_pass + user_fail
        n_user_pass   += user_pass

    return {
        "config": {
            "min_overlap_len": MIN_OVERLAP_LEN,
        },
        "overall": {
            "email_pass_rate":    n_email_pass / n_email_total if n_email_total else None,
            "username_pass_rate": n_user_pass  / n_user_total  if n_user_total  else None,
            "email_total":    n_email_total,
            "email_pass":     n_email_pass,
            "username_total": n_user_total,
            "username_pass":  n_user_pass,
        },
        "per_character": per_character,
        "email_violations": email_violations[:TOP_K_VIOLATIONS],
        "username_violations": user_violations[:TOP_K_VIOLATIONS],
        "totals": {
            "email_violations":    len(email_violations),
            "username_violations": len(user_violations),
        },
    }

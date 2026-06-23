"""I/O helpers for loading ground truth and character metadata."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List

from .types import Character


def load_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_characters(path: Path) -> Dict[str, Character]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {cid: Character.from_dict(cid, d) for cid, d in raw.items()}


def load_ground_truth(path: Path) -> List[dict]:
    """Returns a list of raw ground-truth row dicts in input order."""
    return list(load_jsonl(path))

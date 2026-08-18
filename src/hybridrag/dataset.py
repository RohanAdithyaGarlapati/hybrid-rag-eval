"""Dataset loading helpers.

Locates and reads the JSONL files produced by ``data/build_dataset.py``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def default_data_dir() -> Path:
    """Best effort location of the ``data`` directory relative to the repo root."""
    here = Path(__file__).resolve()
    # src/hybridrag/dataset.py -> repo root is three parents up.
    candidate = here.parents[2] / "data"
    return candidate


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_dataset(data_dir: str | Path | None = None) -> dict:
    d = Path(data_dir) if data_dir else default_data_dir()
    corpus = load_jsonl(d / "corpus.jsonl")
    questions = load_jsonl(d / "questions.jsonl")
    unanswerable = load_jsonl(d / "unanswerable.jsonl")
    return {
        "data_dir": d,
        "corpus": corpus,
        "questions": questions,
        "unanswerable": unanswerable,
    }


def dataset_hash(data_dir: str | Path | None = None) -> str:
    d = Path(data_dir) if data_dir else default_data_dir()
    return hashlib.sha256((d / "corpus.jsonl").read_bytes()).hexdigest()

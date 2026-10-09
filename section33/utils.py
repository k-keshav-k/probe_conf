"""Small helpers shared by the Section 3.3 implementation."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch is optional for pure-data modules
        pass


# ---------------------------------------------------------------------------
# Hashing and JSON I/O
# ---------------------------------------------------------------------------
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_hash(items: Iterable[str]) -> str:
    ordered = sorted(str(x) for x in items)
    return hashlib.sha256("\n".join(ordered).encode()).hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def append_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, default=_json_default) + "\n" for row in rows)
    if not payload:
        return 0
    with open(path, "a") as handle:
        handle.write(payload)
        handle.flush()
    return payload.count("\n")


def read_jsonl(path: Path, skip_invalid: bool = False) -> list[dict]:
    rows = []
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if not skip_invalid:
                    raise
    return rows


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[section33 {stamp}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Iteration helpers
# ---------------------------------------------------------------------------
def batched(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def ensure_dirs(paths: dict[str, Path]) -> None:
    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: Path, payload) -> None:
    """Write JSON through a temporary file so a crash cannot truncate it."""

    tmp = Path(str(path) + ".tmp")
    write_json(tmp, payload)
    os.replace(tmp, path)


def warn(message: str) -> None:
    print(f"[section33 WARNING] {message}", file=sys.stderr, flush=True)

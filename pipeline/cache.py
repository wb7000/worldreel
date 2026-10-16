"""Content hashing for cache keys and output fingerprints."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

STATUS_FILE = "_stage.json"


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def tree_listing(root: Path) -> dict[str, str]:
    """SHA-256 of every file under `root` (relative POSIX path → hash), status file excluded."""
    return {
        p.relative_to(root).as_posix(): file_sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != STATUS_FILE
    }


def listing_hash(listing: Mapping[str, str]) -> str:
    return hashlib.sha256(json.dumps(dict(sorted(listing.items()))).encode()).hexdigest()


def cache_key(
    *,
    stage: str,
    version: str,
    impl: str,
    config: Mapping[str, Any],
    seed: int,
    source_sha256: str,
    input_hashes: Mapping[str, str],
) -> str:
    """Same stage code, config, seed and inputs → same key → the cached output is reused."""
    payload = {
        "stage": stage,
        "version": version,
        "impl": impl,
        "config": config,
        "seed": seed,
        "source": source_sha256,
        "inputs": dict(sorted(input_hashes.items())),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()

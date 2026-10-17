"""The stage contract (docs/ARCHITECTURE.md): stages read files, write files, nothing else."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol

Status = Literal["pending", "running", "done", "failed"]
Impl = Literal["mock", "real", "images", "oracle"]

SOURCE = "source"
"""Pseudo-dependency name for the item's raw source file."""


@dataclass(frozen=True)
class Item:
    """One ingested file, as recorded in the data manifest."""

    id: str
    sha256: str
    path: Path
    source: str = ""
    license: str = ""
    upload_rights: str = "unknown"

    @property
    def meta(self) -> dict[str, str]:
        return {
            "id": self.id,
            "sha256": self.sha256,
            "source": self.source,
            "license": self.license,
            "upload_rights": self.upload_rights,
        }


@dataclass
class StageContext:
    """Everything a stage may use. `inputs` maps each dependency to its finished output folder;
    `out` is an empty folder the stage writes into (moved into place only on success)."""

    item: Item
    inputs: Mapping[str, Path]
    out: Path
    config: Mapping[str, Any]
    seed: int
    log: logging.Logger
    upstream: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    """Status records of finished dependencies (versions, cache keys), for provenance."""


class Stage(Protocol):
    name: ClassVar[str]
    version: ClassVar[str]
    impl: ClassVar[Impl]
    depends_on: ClassVar[tuple[str, ...]]

    def run(self, ctx: StageContext) -> None: ...


class StageError(Exception):
    """A stage failed in an expected way (bad input, missing tool). Message is shown to users."""

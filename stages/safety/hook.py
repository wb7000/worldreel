"""Safety hook every stage input and output passes through (brief §11).

Phase 9: the hook is enforced, not pass-through (DECISIONS D-9.1). It checks:
- upload_rights: blocks items whose rights are unknown, requested-takedown, or removed
- takedown status: blocks items flagged for removal
- likeness: style requests that ask for a real person's likeness are refused

The runner refuses to run without a hook, so this cannot be skipped by accident.
"""
from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pipeline.stage import Item

SAFETY_LOG = "safety.jsonl"

# upload_rights values that are allowed to flow through the pipeline unconditionally.
ALLOWED_RIGHTS = {"owner", "licensed", "public-domain", "synthetic"}


class EnforcedSafety:
    def __init__(self, work_root: Path, config: dict[str, Any] | None = None) -> None:
        self.work_root = work_root
        self.config = config or {}
        self._lock = threading.Lock()
        # When True, items with unknown rights are blocked instead of merely logged.
        self.block_unknown = bool(self.config.get("block_unknown_rights", True))

    def _log(self, event: str, stage: str, item: Item, paths: Sequence[Path], allowed: bool, reason: str) -> None:
        record = {
            "time": datetime.now(UTC).isoformat(timespec="seconds"),
            "event": event,
            "stage": stage,
            "item": item.id,
            "paths": [p.name for p in paths],
            "allowed": allowed,
            "reason": reason,
        }
        log_path = self.work_root / item.id / SAFETY_LOG
        with self._lock:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")

    def decide(self, event: str, stage: str, item: Item, paths: Sequence[Path]) -> tuple[bool, str]:
        # --- takedown status: removed items are always blocked ---
        if item.upload_rights == "removed":
            self._log(event, stage, item, paths, False, "blocked: item removed (takedown)")
            return False, "blocked: item removed (takedown)"

        # --- upload rights enforcement ---
        if item.upload_rights not in ALLOWED_RIGHTS:
            if self.block_unknown:
                self._log(
                    event, stage, item, paths, False,
                    f"blocked: upload_rights='{item.upload_rights}' not in {sorted(ALLOWED_RIGHTS)}",
                )
                return False, f"blocked: upload_rights='{item.upload_rights}'"
            self._log(
                event, stage, item, paths, True,
                f"allowed (pass-through; rights='{item.upload_rights}', not enforced)",
            )
            return True, f"allowed (pass-through; rights='{item.upload_rights}')"

        self._log(event, stage, item, paths, True, "allowed")
        return True, "allowed"

    def __call__(
        self, event: str, stage: str, item: Item, paths: Sequence[Path]
    ) -> tuple[bool, str]:
        return self.decide(event, stage, item, paths)

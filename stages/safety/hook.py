"""Safety hook every stage input and output passes through (brief §11).

Phase 2: a logging pass-through that always allows, and records every decision. Later phases
replace `decide` with real checks (upload rights, likeness filter, takedown status). The runner
refuses to run without a hook, so this cannot be skipped by accident.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pipeline.stage import Item

SAFETY_LOG = "safety.jsonl"


class PassThroughSafety:
    def __init__(self, work_root: Path) -> None:
        self.work_root = work_root
        self._lock = threading.Lock()

    def decide(self, event: str, stage: str, item: Item, paths: Sequence[Path]) -> tuple[bool, str]:
        del event, stage, paths
        if item.upload_rights == "unknown":
            return True, "allowed (pass-through; upload rights unknown, not enforced until Phase 9)"
        return True, "allowed (pass-through)"

    def __call__(
        self, event: str, stage: str, item: Item, paths: Sequence[Path]
    ) -> tuple[bool, str]:
        allowed, reason = self.decide(event, stage, item, paths)
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
        return allowed, reason

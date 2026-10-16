"""Job runner: executes stages for one item in dependency order, in parallel where independent,
cached by content, resumable after a crash, and with failures isolated to dependent stages.

Layout per item (`work_root/<item id>/`):
    <stage>/            finished output of a stage, including `_stage.json`
    .<stage>.tmp-*      in-progress output (discarded on failure or crash)
    safety.jsonl        every safety-hook decision
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import traceback
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pipeline.cache import STATUS_FILE, cache_key, config_hash, listing_hash, tree_listing
from pipeline.stage import SOURCE, Item, Stage, StageContext, StageError, Status

SafetyHook = Callable[[str, str, Item, Sequence[Path]], tuple[bool, str]]
"""(event, stage name, item, paths) → (allowed, reason). Events: "input", "output"."""

log = logging.getLogger("worldreel.pipeline")


@dataclass
class StageResult:
    name: str
    status: Status
    cached: bool = False
    error: str = ""
    seconds: float = 0.0


@dataclass
class RunResult:
    item_id: str
    stages: dict[str, StageResult] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(r.status == "done" for r in self.stages.values())

    def status_of(self, name: str) -> Status:
        return self.stages[name].status


def topological_order(stages: Sequence[Stage]) -> list[Stage]:
    by_name = {s.name: s for s in stages}
    if len(by_name) != len(stages):
        raise ValueError("duplicate stage names")
    order: list[Stage] = []
    state: dict[str, int] = {}  # 1 = visiting, 2 = done

    def visit(name: str) -> None:
        if state.get(name) == 2:
            return
        if state.get(name) == 1:
            raise ValueError(f"dependency cycle through stage {name!r}")
        state[name] = 1
        for dep in by_name[name].depends_on:
            if dep == SOURCE:
                continue
            if dep not in by_name:
                raise ValueError(f"stage {name!r} depends on unknown stage {dep!r}")
            visit(dep)
        state[name] = 2
        order.append(by_name[name])

    for s in stages:
        visit(s.name)
    return order


def read_status(stage_dir: Path) -> dict[str, Any] | None:
    path = stage_dir / STATUS_FILE
    if not path.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data


class Runner:
    def __init__(
        self,
        work_root: Path,
        stages: Sequence[Stage],
        *,
        config: Mapping[str, Mapping[str, Any]] | None = None,
        seed: int = 0,
        max_workers: int = 4,
        safety: SafetyHook | None = None,
        verify_cache: bool = True,
    ) -> None:
        if safety is None:
            raise ValueError("a safety hook is required (use stages.safety.hook); see CLAUDE.md")
        self.work_root = work_root
        self.stages = topological_order(stages)
        self.config = config or {}
        self.seed = seed
        self.max_workers = max_workers
        self.safety = safety
        self.verify_cache = verify_cache

    # ------------------------------------------------------------------ public

    def run(self, item: Item) -> RunResult:
        item_root = self.work_root / item.id
        item_root.mkdir(parents=True, exist_ok=True)
        for leftover in item_root.glob(".*.tmp-*"):
            shutil.rmtree(leftover, ignore_errors=True)  # from a crashed earlier run

        result = RunResult(item.id)
        remaining = {s.name: s for s in self.stages}
        output_hashes: dict[str, str] = {}
        records: dict[str, dict[str, Any]] = {}
        running: dict[Future[StageResult], Stage] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            while remaining or running:
                for name, stage in list(remaining.items()):
                    deps = [d for d in stage.depends_on if d != SOURCE]
                    failed = [
                        d
                        for d in deps
                        if result.stages.get(d) and result.stages[d].status == "failed"
                    ]
                    if failed:
                        result.stages[name] = StageResult(
                            name, "failed", error=f"blocked: upstream {failed[0]} failed"
                        )
                        del remaining[name]
                        continue
                    if all(d in output_hashes for d in deps):
                        del remaining[name]
                        inputs = {d: output_hashes[d] for d in deps}
                        upstream = {d: records[d] for d in deps}
                        running[pool.submit(self._run_stage, item, stage, inputs, upstream)] = stage
                if not running:
                    if remaining:  # cannot happen with a valid DAG; guard against a stuck loop
                        for name in remaining:
                            result.stages[name] = StageResult(name, "failed", error="unschedulable")
                        remaining.clear()
                    break
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for fut in done:
                    stage = running.pop(fut)
                    res = fut.result()
                    result.stages[stage.name] = res
                    if res.status == "done":
                        record = read_status(item_root / stage.name) or {}
                        records[stage.name] = record
                        output_hashes[stage.name] = str(record.get("output_hash", ""))
        return result

    # ------------------------------------------------------------------ internals

    def _stage_config(self, stage: Stage) -> dict[str, Any]:
        return dict(self.config.get(stage.name, {}))

    def _run_stage(
        self,
        item: Item,
        stage: Stage,
        input_hashes: Mapping[str, str],
        upstream: Mapping[str, Mapping[str, Any]],
    ) -> StageResult:
        item_root = self.work_root / item.id
        final = item_root / stage.name
        cfg = self._stage_config(stage)
        key = cache_key(
            stage=stage.name,
            version=stage.version,
            impl=stage.impl,
            config=cfg,
            seed=self.seed,
            source_sha256=item.sha256,
            input_hashes=input_hashes,
        )
        existing = read_status(final)
        if existing and existing.get("status") == "done" and existing.get("cache_key") == key:
            if not self.verify_cache or listing_hash(tree_listing(final)) == existing.get(
                "output_hash"
            ):
                log.info("%s/%s: cached", item.id, stage.name)
                return StageResult(stage.name, "done", cached=True)
            log.warning("%s/%s: cached output was modified; re-running", item.id, stage.name)

        inputs = {d: item_root / d for d in stage.depends_on if d != SOURCE}
        input_paths: list[Path] = [item.path] if SOURCE in stage.depends_on else []
        input_paths += list(inputs.values())
        allowed, reason = self.safety("input", stage.name, item, input_paths)
        if not allowed:
            return self._fail(final, stage, key, f"blocked by safety hook: {reason}", 0.0)

        tmp = item_root / f".{stage.name}.tmp-{uuid.uuid4().hex[:8]}"
        tmp.mkdir()
        started = time.perf_counter()
        _write_status(
            final.parent / f".{stage.name}.running.json", {"status": "running", "cache_key": key}
        )
        try:
            stage.run(
                StageContext(
                    item=item,
                    inputs=inputs,
                    out=tmp,
                    config=cfg,
                    seed=self.seed,
                    log=logging.getLogger(f"worldreel.stage.{stage.name}"),
                    upstream=upstream,
                )
            )
        except StageError as exc:
            shutil.rmtree(tmp, ignore_errors=True)
            return self._fail(final, stage, key, str(exc), time.perf_counter() - started)
        except Exception as exc:
            shutil.rmtree(tmp, ignore_errors=True)
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            log.debug("stage %s crashed:\n%s", stage.name, traceback.format_exc())
            return self._fail(
                final, stage, key, f"crashed: {detail}", time.perf_counter() - started
            )
        finally:
            (final.parent / f".{stage.name}.running.json").unlink(missing_ok=True)

        allowed, reason = self.safety("output", stage.name, item, [tmp])
        if not allowed:
            shutil.rmtree(tmp, ignore_errors=True)
            return self._fail(final, stage, key, f"output blocked by safety hook: {reason}", 0.0)

        seconds = time.perf_counter() - started
        listing = tree_listing(tmp)
        _write_status(
            tmp / STATUS_FILE,
            {
                "stage": stage.name,
                "version": stage.version,
                "impl": stage.impl,
                "status": "done",
                "cache_key": key,
                "config": cfg,
                "config_sha256": config_hash(cfg),
                "seed": self.seed,
                "inputs": dict(input_hashes),
                "outputs": listing,
                "output_hash": listing_hash(listing),
                "seconds": round(seconds, 3),
                "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
        )
        if final.exists():
            shutil.rmtree(final)
        tmp.replace(final)
        return StageResult(stage.name, "done", seconds=seconds)

    def _fail(self, final: Path, stage: Stage, key: str, error: str, seconds: float) -> StageResult:
        log.error("stage %s failed: %s", stage.name, error)
        if final.exists():
            shutil.rmtree(final)
        final.mkdir(parents=True)
        _write_status(
            final / STATUS_FILE,
            {
                "stage": stage.name,
                "version": stage.version,
                "status": "failed",
                "cache_key": key,
                "error": error,
            },
        )
        return StageResult(stage.name, "failed", error=error, seconds=seconds)


def _write_status(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")

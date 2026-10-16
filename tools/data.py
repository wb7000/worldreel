"""Data workflow CLI (docs/DATA.md).

    python -m tools.data ingest                      # inbox/ → raw/ (+ manifest rows)
    python -m tools.data list
    python -m tools.data info <id> --source ... --license ... [--rights owner]
    python -m tools.data process [--id <id>] [--config pipeline/default.toml] [--mock]
    python -m tools.data freeze <name> [--description ...]
    python -m tools.data verify <name>
    python -m tools.data exclude <id> --reason ...
    python -m tools.data remove <id> --reason ... --yes

Data root: --root, else $WORLDREEL_DATA, else ./data in the repository.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from pipeline.registry import VIDEO_PIPELINE, PipelineConfig, build_stages
from pipeline.runner import Runner, RunResult
from stages.ingest.store import DataStore, DataStoreError
from stages.safety.hook import PassThroughSafety
from tools import datasets
from tools.toolpaths import REPO_ROOT


def default_root() -> Path:
    env = os.environ.get("WORLDREEL_DATA")
    return Path(env) if env else REPO_ROOT / "data"


def process(
    store: DataStore, config: PipelineConfig, ids: list[str] | None = None
) -> dict[str, RunResult]:
    stages = build_stages(VIDEO_PIPELINE, config.impl)
    runner = Runner(
        store.processed,
        stages,
        config=config.stage,
        seed=config.seed,
        max_workers=config.max_workers,
        safety=PassThroughSafety(store.processed),
    )
    rows = [r for r in store.active() if r.media_type == "video" and (ids is None or r.id in ids)]
    if ids:
        missing = set(ids) - {r.id for r in rows}
        if missing:
            raise DataStoreError(f"not active videos: {sorted(missing)}")
    return {row.id: runner.run(row.item(store.root)) for row in rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.data", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--root", type=Path, default=None, help="data folder (default ./data)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    sub.add_parser("ingest")
    sub.add_parser("list")
    sub.add_parser("versions")
    p = sub.add_parser("info")
    p.add_argument("id")
    p.add_argument("--source", required=True)
    p.add_argument("--license", required=True)
    p.add_argument("--rights", default="unknown")
    p.add_argument("--notes", default="")
    p = sub.add_parser("process")
    p.add_argument("--id", action="append", dest="ids")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--mock", action="store_true", help="use mock implementations for every stage")
    p = sub.add_parser("freeze")
    p.add_argument("name")
    p.add_argument("--description", default="")
    p = sub.add_parser("verify")
    p.add_argument("name")
    p = sub.add_parser("exclude")
    p.add_argument("id")
    p.add_argument("--reason", required=True)
    p = sub.add_parser("remove")
    p.add_argument("id")
    p.add_argument("--reason", required=True)
    p.add_argument("--yes", action="store_true", help="confirm permanent deletion")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    store = DataStore(args.root or default_root())
    try:
        match args.cmd:
            case "init":
                store.init()
                print(f"initialised {store.root}")
            case "ingest":
                report = store.ingest()
                print(report.summary())
                for fid in report.needs_info:
                    print(f"  needs info: {fid}: {store.get(fid).notes}")
            case "list":
                for r in store.rows():
                    print(f"{r.id}  {r.status:<10} {r.media_type:<5} {r.license:<16} ", end="")
                    print(r.original_name)
            case "versions":
                for name in datasets.list_versions(store):
                    print(name)
            case "info":
                store.set_info(args.id, args.source, args.license, args.rights, args.notes)
                print(f"{args.id} is now active")
            case "process":
                config = (
                    PipelineConfig.all_mock() if args.mock else PipelineConfig.load(args.config)
                )
                results = process(store, config, args.ids)
                failed = False
                for fid, res in results.items():
                    print(f"{fid}: {'ok' if res.ok else 'FAILED'}")
                    for st in res.stages.values():
                        mark = "cached" if st.cached else st.status
                        print(f"  {st.name:<12} {mark}{'  ' + st.error if st.error else ''}")
                    failed |= not res.ok
                if not results:
                    print("nothing to process (no active videos)")
                return 1 if failed else 0
            case "freeze":
                version = datasets.freeze(store, args.name, args.description)
                print(f"froze {args.name}: {version['count']} files")
            case "verify":
                check = datasets.verify(store, args.name)
                for w in check.warnings:
                    print(f"  warning: {w}")
                for e in check.errors:
                    print(f"  error: {e}")
                print(f"{args.name}: {'reproducible' if check.ok else 'NOT reproducible'}")
                return 0 if check.ok else 1
            case "exclude":
                store.exclude(args.id, args.reason)
                print(f"{args.id} excluded")
            case "remove":
                if not args.yes:
                    print(
                        "remove deletes the file permanently; re-run with --yes to confirm",
                        file=sys.stderr,
                    )
                    return 2
                used = datasets.versions_using(store, args.id)
                retrain = store.remove(args.id, args.reason, used)
                print(
                    f"{args.id} removed"
                    + (f"; RETRAIN NEEDED (used in {', '.join(used)})" if retrain else "")
                )
    except DataStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""`python -m formats validate <file> [...]` — validate WorldReel files; exit 1 on any error."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from formats.common import Report
from formats.wedit import validate_wedit
from formats.wreel import validate_wreel
from formats.wscript import validate_wscript

VALIDATORS: dict[str, Callable[[Path], Report]] = {
    ".wreel": validate_wreel,
    ".wedit": validate_wedit,
    ".wscript": validate_wscript,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m formats")
    sub = parser.add_subparsers(dest="command", required=True)
    val = sub.add_parser("validate", help="validate .wreel / .wedit / .wscript files")
    val.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)

    failed = False
    for path in args.paths:
        validator = VALIDATORS.get(path.suffix.lower())
        if validator is None:
            print(f"{path}: unknown extension (expected {', '.join(VALIDATORS)})")
            failed = True
            continue
        report = validator(path)
        status = "OK" if report.ok else "INVALID"
        print(f"{path}: {status} ({len(report.errors)} errors, {len(report.warnings)} warnings)")
        for issue in report.issues:
            print(f"  {issue}")
        failed |= not report.ok
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

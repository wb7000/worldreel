"""Read a libvmaf XML log and print the pooled VMAF mean."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/analysis/S016_vmaf.json")
    text = path.read_text(encoding="utf-8")
    scores = [float(x) for x in re.findall(r'\bvmaf="([0-9.]+)"', text)]
    if not scores:
        print(f"no vmaf scores in {path}")
        return 1
    print(
        f"{path}: frames={len(scores)} mean={sum(scores) / len(scores):.4f} "
        f"min={min(scores):.4f} max={max(scores):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

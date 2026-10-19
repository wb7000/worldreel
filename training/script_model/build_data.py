"""Build the script-model data set (D-7.5) from converted plays + the sample script.

    python -m training.script_model.build_data --plays data/plays/wscript --out data/plays/sft

Split by play (no scene of a test play is ever trained on). Writes chat-format JSONL
(`{"messages": [system, user, assistant]}`, Soup's `chatml` format) for train / val, and test
files that also carry the script's dictionary for scoring. `manifest.json` records the split,
counts, stride and seed (the frozen dataset version).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from formats.wscript import Script, read_wscript
from stages.generate.script_model import Example, examples
from tools.make_sample_wscript import OUT as SAMPLE

SPLIT = {
    "test": ["pg8121", "pg3618"],  # Ghosts (Ibsen/Archer), Arms and the Man (Shaw)
    "val": ["pg1756"],  # Uncle Vanya (Chekhov)
}
STRIDE = {"train": 8, "val": 24, "test": 24}
SEED = 7


def _row(ex: Example, script: Script, with_dict: bool) -> dict[str, object]:
    row: dict[str, object] = {"messages": ex.messages()}
    if with_dict:
        row["meta"] = {
            "title": ex.script_title,
            "scene": ex.scene_id,
            "cut": ex.cut,
            "dictionary": {
                k: sorted(v) for k, v in script.header["dictionary"].items() if isinstance(v, dict)
            },
        }
    return row


def build(plays: Path, out: Path) -> dict[str, object]:
    rng = random.Random(SEED)
    out.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": [], "ood": []}
    sources: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for d in sorted(plays.glob("pg*.wscript")):
        split = next((k for k, v in SPLIT.items() if d.stem in v), "train")
        script = read_wscript(d)
        assert script.report.ok, d
        sources[split].append(f"{d.stem}: {script.header['title']}")
        for ex in examples(script, STRIDE[split], rng):
            rows[split].append(_row(ex, script, split == "test"))
    sample = read_wscript(SAMPLE)
    rows["ood"] = [_row(ex, sample, True) for ex in examples(sample, 6, rng, min_prefix=4)]
    rng.shuffle(rows["train"])
    digest = hashlib.sha256()
    for name, rs in rows.items():
        data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs).encode("utf-8")
        (out / f"{name}.jsonl").write_bytes(data)
        digest.update(data)
    manifest = {
        "version": digest.hexdigest()[:16],
        "seed": SEED,
        "stride": STRIDE,
        "sources": sources,
        "ood": "The Lantern Keeper (eval/scripts/lantern_keeper.wscript)",
        "counts": {k: len(v) for k, v in rows.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m training.script_model.build_data")
    p.add_argument("--plays", type=Path, default=Path("data/plays/wscript"))
    p.add_argument("--out", type=Path, default=Path("data/plays/sft"))
    a = p.parse_args(argv)
    print(json.dumps(build(a.plays, a.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

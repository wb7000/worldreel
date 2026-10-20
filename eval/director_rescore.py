"""Re-score a director run from its saved model answers (no GPU): rebuild the delivery layers
with the current parser / meaning check and report the same numbers as the Kaggle job.

    python -m eval.director_rescore runs/kaggle/director/out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.director_eval import _norm, score, test_script
from formats.wscript import Script, read_wscript
from stages.generate.director import apply_rewrites, dialogue_lines, parse_answer, to_layer
from tools.make_sample_wscript import OUT as SAMPLE


def rebuild(run: Path) -> tuple[Script, Script, dict[str, Any]]:
    test, gold = test_script()
    sample = read_wscript(SAMPLE)
    scripts = {"test": test, "lantern": sample}
    a1 = json.loads((run / "stage1_answers.json").read_text())["answers"]
    s2 = run / "stage2_answers.json"
    a2 = json.loads(s2.read_text())["answers"] if s2.exists() else {}
    for key, a in a1.items():
        g, sid = key.split("/")
        try:
            parsed = parse_answer(a["answer"])
        except ValueError:
            parsed = []
        scripts[g].delivery[sid] = to_layer(scripts[g], sid, parsed, "rescore")[0]
    for key, a in a2.items():
        g, sid = key.split("/")[:2]  # v4 keys carry a chunk number
        try:
            parsed = parse_answer(a["answer"])
        except ValueError:
            parsed = []
        apply_rewrites(scripts[g], sid, scripts[g].delivery[sid], parsed)
    lantern = [
        (sample.delivery[sid]["lines"][str(i)], t)
        for sid in sample.scenes
        for i, _, t in dialogue_lines(sample, sid)
    ]
    report = {
        "test": {k: v for k, v in score(test, gold).items() if k != "rows"},
        "lantern": {
            "lines": len(lantern),
            "spoken_changed_ok": round(
                sum(d["meaning_ok"] and _norm(d["spoken"]) != _norm(t) for d, t in lantern)
                / len(lantern),
                3,
            ),
            "no_takes": sum(not d.get("takes") for d, _ in lantern),
        },
    }
    return test, sample, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.director_rescore")
    ap.add_argument("run", type=Path)
    a = ap.parse_args(argv)
    print(json.dumps(rebuild(a.run)[2], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

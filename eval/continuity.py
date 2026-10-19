"""A-7 measurement: does the continuity tracker catch seeded continuity errors?

    python -m eval.continuity [--trials 200] [--seed 0]

Starts from the clean sample script (0 issues) and applies one random corruption per trial:
- `drop_change`: delete one `=` line (a hand-off or move that is no longer written);
- `wrong_location`: a `location=` change points to a different place;
- `phantom_prop`: a character suddenly holds a random prop;
- `edit_start`: one value in a later scene's start state is changed (a wardrobe / prop slip).
Reported per kind: share detected by the full checker, by in-scene checks only and by the
scene-boundary check only (ablation). The no-checker baseline detects nothing.
"""

from __future__ import annotations

import argparse
import copy
import random
from typing import Any

from formats.wscript import Change, Script
from stages.generate.continuity import check
from tools.make_sample_wscript import build

KINDS = ("drop_change", "wrong_location", "phantom_prop", "edit_start")


def corrupt(script: Script, kind: str, rng: random.Random) -> Script | None:
    s = Script(header=copy.deepcopy(script.header), scenes=copy.deepcopy(script.scenes))
    d = s.header["dictionary"]
    sids = list(s.scenes)
    if kind in ("drop_change", "wrong_location"):
        spots = [
            (sid, i, st)
            for sid in sids
            for i, st in enumerate(s.scenes[sid])
            if isinstance(st, Change)
            and (kind == "drop_change" or any(k == "location" for k, _, _ in st.ops))
        ]
        sid, i, st = rng.choice(spots)
        if kind == "drop_change":
            del s.scenes[sid][i]
        else:
            ops = tuple(
                (k, op, rng.choice([x for x in d["locations"] if x != v]) if k == "location" else v)
                for k, op, v in st.ops
            )
            s.scenes[sid][i] = Change(st.target, ops)
    elif kind == "phantom_prop":
        sid = rng.choice(sids)
        stmts = s.scenes[sid]
        i = rng.randrange(1, len(stmts) + 1)
        stmts.insert(
            i,
            Change(
                rng.choice(list(d["characters"])),
                (("holding", "+=", rng.choice(list(d["props"]))),),
            ),
        )
    else:  # edit_start
        entry = rng.choice(s.header["scenes"][1:])
        cont: dict[str, Any] = entry["continuity"]
        cid = rng.choice(list(cont["characters"]))
        key = rng.choice(["location", "holding", "wearing"])
        ch = cont["characters"][cid]
        if key == "location":
            ch["location"] = rng.choice([x for x in d["locations"] if x != ch.get("location")])
        else:
            items = set(ch.get(key, []))
            pid = rng.choice(list(d["props"]))
            ch[key] = sorted(items ^ {pid})  # add it if missing, remove it if there
    return s


def detected(script: Script) -> tuple[bool, bool, bool]:
    report, _ = check(script)
    codes = [i.code for i in report.errors]
    boundary = any(c == "continuity_break" for c in codes)
    inside = any(c != "continuity_break" for c in codes)
    return boundary or inside, inside, boundary


def run(trials: int, seed: int) -> dict[str, dict[str, float]]:
    clean = build()
    assert check(clean)[0].ok
    rng = random.Random(seed)
    out: dict[str, dict[str, float]] = {}
    for kind in KINDS:
        hits = [0, 0, 0]
        for _ in range(trials):
            bad = corrupt(clean, kind, rng)
            assert bad is not None
            for k, flag in enumerate(detected(bad)):
                hits[k] += flag
        out[kind] = {
            "full": hits[0] / trials,
            "in_scene_only": hits[1] / trials,
            "boundary_only": hits[2] / trials,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.continuity")
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    for kind, rates in run(a.trials, a.seed).items():
        print(kind, {k: round(v, 3) for k, v in rates.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

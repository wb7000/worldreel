"""A-13 measurement helpers: the conversation test set as a small `.wscript`, and scoring of the
director's delivery layers against the hand labels (D-8.8).

The six exchanges of eval/conversations.json become six scenes (their situation as an action
line, then the *written* lines); the director never sees the hand labels or spoken versions.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from formats.wscript import Action, Dialogue, SceneHeader, Script, Statement, read_wscript
from tools.make_sample_wscript import OUT

TEST = Path("eval/conversations.json")


def test_script() -> tuple[Script, dict[str, list[dict[str, Any]]]]:
    """(script, gold turns per scene id). Characters, places and props from the sample script."""
    base = read_wscript(OUT)
    exchanges = json.loads(TEST.read_text(encoding="utf-8"))["exchanges"]
    header = copy.deepcopy(base.header)
    header["title"] = "Director test set"
    index: list[dict[str, Any]] = []
    scenes: dict[str, list[Statement]] = {}
    gold: dict[str, list[dict[str, Any]]] = {}
    start = base.header["scenes"][0]["continuity"]
    for n, ex in enumerate(exchanges, start=1):
        sid = f"S{n:03d}"
        stmts: list[Statement] = [SceneHeader(sid, (("loc", "L04"),)), Action(ex["situation"])]
        stmts += [Dialogue(t["cid"], t["written"]) for t in ex["turns"]]
        scenes[sid] = stmts
        gold[sid] = ex["turns"]
        everyone_here = {
            "characters": {c: {"location": "L04"} for c in start["characters"]},
            **({"offscreen": [f"{c}.location" for c in start["characters"]]} if n > 1 else {}),
        }
        index.append({"id": sid, "title": ex["id"], "summary": ex["situation"], "location": "L04",
                      "tension": 5, "continuity": everyone_here})  # fmt: skip
    header["scenes"] = index
    return Script(header=header, scenes=scenes), gold


def score(script: Script, gold: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Emotion agreement with the hand labels, meaning-check pass rate, coverage."""
    rows = []
    for sid, turns in gold.items():
        layer = script.delivery.get(sid, {}).get("lines", {})
        dialogue = [i for i, s in enumerate(script.scenes[sid]) if isinstance(s, Dialogue)]
        for i, t in zip(dialogue, turns, strict=True):
            d = layer.get(str(i), {})
            rows.append({"scene": sid, "gold_emotion": t["emotion"], "emotion": d.get("emotion"),
                         "gold_manner": t["manner"], "manner": d.get("manner"),
                         "meaning_ok": d.get("meaning_ok", False), "spoken": d.get("spoken"),
                         "proposed": d.get("proposed"), "gold_spoken": t["spoken"]})  # fmt: skip
    n = len(rows)
    return {
        "lines": n,
        "covered": sum(r["emotion"] is not None for r in rows),
        "emotion_agreement": round(sum(r["emotion"] == r["gold_emotion"] for r in rows) / n, 3),
        "meaning_ok": round(sum(bool(r["meaning_ok"]) for r in rows) / n, 3),
        "rows": rows,
    }

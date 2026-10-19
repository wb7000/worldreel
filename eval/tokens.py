"""D-7.3: how many tokens does a model read with `.wscript` scene loading?

    python -m eval.tokens [--script PATH]

Compared, per tokenizer:
- `whole_wscript`: header + every scene file (feeding the whole structured script);
- `whole_plain`: the whole script as a plain screenplay (names spelled out, `screenplay.render`);
- `context`: what `load_context(...).to_prompt()` gives for one scene (mean / max over scenes);
- `scene_only`: one scene's `.wsl` text alone, no dictionary / state / summaries (mean).
Plus `ids_vs_names`: one scene's `.wsl` vs the same scene with every `{ID}` expanded to its name,
summed over scenes — whether IDs save tokens at all.

Scaling: the same numbers for a 60-scene script (the sample repeated 6 times with new scene IDs,
about an hour by the page convention), to see what grows with length and what does not.

Tokenizers: tiktoken `o200k_base` (MIT) and the tokenizer of Qwen2.5-0.5B-Instruct (Apache-2.0,
`tokenizers`; file in git-ignored `data/models/qwen2.5-0.5b-instruct/tokenizer.json`).
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Callable
from pathlib import Path
from statistics import mean
from typing import Any

from formats.common import dump_json
from formats.wscript import (
    Action,
    Dialogue,
    SceneHeader,
    Script,
    Statement,
    load_context,
    read_wscript,
    serialize_scene,
)
from stages.generate.screenplay import render
from tools.make_sample_wscript import OUT

QWEN_TOKENIZER = Path("data/models/qwen2.5-0.5b-instruct/tokenizer.json")


def tokenizers() -> dict[str, Callable[[str], int]]:
    import tiktoken
    from tokenizers import Tokenizer

    out: dict[str, Callable[[str], int]] = {}
    enc = tiktoken.get_encoding("o200k_base")
    out["o200k_base"] = lambda s: len(enc.encode(s))
    if QWEN_TOKENIZER.exists():
        tok = Tokenizer.from_file(str(QWEN_TOKENIZER))
        out["qwen2.5"] = lambda s: len(tok.encode(s).ids)
    return out


def _expanded(script: Script, sid: str) -> str:
    from formats.wscript import expand_text

    d = script.header["dictionary"]
    names = {cid: str(c["name"]) for cid, c in d["characters"].items()}
    stmts: list[Statement] = []
    for s in script.scenes[sid]:
        match s:
            case Dialogue(speaker, text, manner):
                stmts.append(Dialogue(names[speaker], expand_text(text, d), manner))
            case Action(text):
                stmts.append(Action(expand_text(text, d)))
            case _:
                stmts.append(s)
    # Dialogue with a name instead of an ID is not valid .wsl; serialise by hand.
    lines = []
    for s in stmts:
        if isinstance(s, Dialogue):
            lines.append(
                f"{s.speaker} ({s.manner}): {s.text}" if s.manner else f"{s.speaker}: {s.text}"
            )
        else:
            lines.append(serialize_scene([s]).rstrip("\n"))
    return "\n".join(lines) + "\n"


def repeat(script: Script, times: int) -> Script:
    """The script repeated `times` times with new scene IDs (a longer script of the same kind)."""
    header = copy.deepcopy(script.header)
    scenes: dict[str, Any] = {}
    index = []
    n = 0
    for _ in range(times):
        for entry in script.header["scenes"]:
            n += 1
            sid = f"S{n:03d}"
            stmts = [
                SceneHeader(sid, s.params) if isinstance(s, SceneHeader) else s
                for s in script.scenes[entry["id"]]
            ]
            scenes[sid] = stmts
            index.append({**copy.deepcopy(entry), "id": sid, "path": f"scenes/{sid}.wsl"})
    header["scenes"] = index
    return Script(header=header, scenes=scenes)


def measure(script: Script, count: Callable[[str], int]) -> dict[str, float]:
    sids = [e["id"] for e in script.header["scenes"]]
    whole = dump_json(script.header).decode() + "".join(
        serialize_scene(script.scenes[s]) for s in sids
    )
    ctx = [count(load_context(script, s).to_prompt()) for s in sids]
    scene = [count(serialize_scene(script.scenes[s])) for s in sids]
    return {
        "scenes": len(sids),
        "whole_wscript": count(whole),
        "whole_plain": count(render(script)),
        "context_mean": round(mean(ctx), 1),
        "context_max": max(ctx),
        "scene_only_mean": round(mean(scene), 1),
        "ids_total": sum(scene),
        "names_total": sum(count(_expanded(script, s)) for s in sids),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.tokens")
    p.add_argument("--script", type=Path, default=OUT)
    a = p.parse_args(argv)
    script = read_wscript(a.script)
    for name, count in tokenizers().items():
        for label, s in (("10 scenes", script), ("60 scenes", repeat(script, 6))):
            print(name, label, measure(s, count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

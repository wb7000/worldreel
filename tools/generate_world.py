"""Generate one `.wreel` world from a `.wscript` scene (Phase 8, stage `generate`).

Usage:
    python -m tools.generate_world --config generate.toml [--out world.wreel]

The config (TOML) selects script, scene, timing, portraits and duration:
    [stage.generate]
    script = "eval/scripts/lantern_keeper.wscript"
    scene_id = "S001"
    duration = 10.0
    timing = "runs/voice_lantern_final/timing.json"
    portraits = "runs/kaggle/portraits_run3_merged"
    scene_audio = "runs/voice_lantern_final/scenes/S001.wav"
    seed = 7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from stages.generate.real import build_world

tomllib: Any  # the tomllib module, or None on older Python
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


def _load_config(path: Path) -> dict[str, Any]:
    if tomllib is None:
        raise RuntimeError("tomllib is unavailable (needs Python 3.11+)")
    with path.open("rb") as f:
        return cast(dict[str, Any], tomllib.load(f))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.generate_world")
    ap.add_argument("--config", type=Path, required=True, help="TOML config for the generate run")
    ap.add_argument(
        "--out", type=Path, default=Path("runs/generated.wreel"), help="output `.wreel` path"
    )
    ap.add_argument("--scene-id", default=None, help="override scene id")
    ap.add_argument("--duration", type=float, default=None, help="override duration (seconds)")
    ap.add_argument("--seed", type=int, default=None, help="override seed")
    a = ap.parse_args(argv)

    cfg = _load_config(a.config)
    gen_cfg = dict(cfg.get("stage", {}).get("generate", {}))
    if a.scene_id is not None:
        gen_cfg["scene_id"] = a.scene_id
    if a.duration is not None:
        gen_cfg["duration"] = a.duration
    if a.seed is not None:
        gen_cfg["seed"] = a.seed

    script_dir = Path(gen_cfg["script"])
    scene_id = gen_cfg["scene_id"]
    duration = float(gen_cfg.get("duration", 10.0))
    portraits_dir = Path(gen_cfg.get("portraits", ""))
    choice = {k: str(v) for k, v in (gen_cfg.get("choice") or {}).items()}
    seed = int(gen_cfg.get("seed", 7))
    timing_path = Path(gen_cfg["timing"]) if gen_cfg.get("timing") else None
    timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path else None
    scene_audio = Path(gen_cfg["scene_audio"]) if gen_cfg.get("scene_audio") else None

    parts = build_world(
        script_dir,
        scene_id,
        duration,
        timing,
        portraits_dir,
        choice,
        seed,
        surface=str(gen_cfg.get("surface", "stone")),
        weather=str(gen_cfg.get("weather", "rain")),
        scene_audio=scene_audio,
    )

    from formats.common import load_or_create_signing_key
    from formats.wreel import write_wreel

    key = load_or_create_signing_key()
    write_wreel(a.out, parts.manifest, parts.entries, key)
    print(
        f"wrote {a.out} ({a.out.stat().st_size} bytes, "
        f"{len(parts.entries)} entries, {len(parts.manifest['actors'])} actors)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

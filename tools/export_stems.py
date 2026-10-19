"""Export the audio stems of a `.wreel` as separate WAV files plus their mix (brief Phase 5:
"stems export separately").

    python -m tools.export_stems world.wreel --out stems/

Writes one `<layer>.wav` per audio track (dialogue, foley, ambience, effects, music, ...) and
`mix.wav` (fixed layer gains, `stages.audio.stems.MIX_GAINS`), plus `stems.json` listing each
file with its layer, licence and duration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from formats.wreel import read_wreel
from stages.audio.stems import DEFAULT_TARGET_LUFS, mix_report, mixdown
from stages.audio.synth import SAMPLE_RATE, wav_samples, write_wav


def export(world: Path, out: Path, target_lufs: float | None = None) -> list[dict[str, object]]:
    w = read_wreel(world)
    out.mkdir(parents=True, exist_ok=True)
    stems = {}
    listing: list[dict[str, object]] = []
    for track in w.manifest["audio"]["tracks"]:
        samples = wav_samples(w.entries[track["path"]])
        name = f"{track['layer']}.wav" if track["layer"] not in stems else f"{track['id']}.wav"
        write_wav(out / name, samples)
        stems[track["layer"]] = samples
        listing.append(
            {
                "file": name,
                "layer": track["layer"],
                "license": track["license"],
                "seconds": round(len(samples) / SAMPLE_RATE, 3),
            }
        )
    if stems:
        mix_cfg = w.manifest["audio"].get("mix", {})
        target = (
            target_lufs
            if target_lufs is not None
            else mix_cfg.get("loudness_target_lufs", DEFAULT_TARGET_LUFS)
        )
        mix = mixdown(stems, float(target), bool(mix_cfg.get("ducking", True)))
        write_wav(out / "mix.wav", mix)
        listing.append(
            {
                "file": "mix.wav",
                "layer": "mix",
                "license": "derived",
                "seconds": None,
                **mix_report(stems, mix, float(target)),
            }
        )
    (out / "stems.json").write_text(json.dumps(listing, indent=2), encoding="utf-8")
    return listing


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m tools.export_stems")
    p.add_argument("world", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--target", type=float, help="mix loudness in LUFS (default: the scene's)")
    a = p.parse_args(argv)
    for row in export(a.world, a.out, a.target):
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

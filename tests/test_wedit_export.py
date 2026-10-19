from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import opentimelineio as otio
import pytest

from formats.common import load_or_create_signing_key
from formats.wedit import read_wedit
from formats.wreel import read_wreel, write_wreel
from stages.audio.stems import render_stems, stem_tracks
from stages.audio.synth import wav_samples
from stages.export.wedit_export import export_wedit, sha256_file
from tools.make_sample_wreel import build_sample
from tools.toolpaths import find_tool

FFMPEG = find_tool("ffmpeg")
MELT = find_tool("melt")


def _world(tmp: Path) -> Path:
    manifest, entries = build_sample()
    manifest["duration"] = 1.0
    tracks, stem_entries = stem_tracks(render_stems(1.0, [], [{"t": 0.0, "tension": 5}], {}))
    manifest["audio"]["tracks"] = tracks
    entries.update(stem_entries)
    path = tmp / "w.wreel"
    write_wreel(path, manifest, entries, load_or_create_signing_key(tmp / "k.pem"))
    return path


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if FFMPEG is None:
        pytest.skip("ffmpeg not available")
    tmp = tmp_path_factory.mktemp("wedit")
    src = _world(tmp)
    world = read_wreel(src)
    cam = world.manifest["cameras"][0]["id"]
    out = tmp / "w.wedit"
    export_wedit(world, sha256_file(src), out, cam, fps=12.0, width=96, height=54)
    return out


def test_bundle_is_valid_signed_and_complete(bundle: Path) -> None:
    w = read_wedit(bundle)
    assert w.report.ok and w.signature_status == "valid"
    m = w.manifest
    assert m["format_version"] == "1.1" and m["level"] == "flat"
    assert m["provenance"]["ai_label"] and "PROVENANCE.txt" in w.entries
    assert {s["layer"] for s in m["stems"]} == {"ambience", "music"}
    assert {t["format"] for t in m["timeline"]} == {"otio", "mlt"}
    depth = next(lyr for lyr in m["render_layers"] if lyr["kind"] == "depth")
    frames = [p for p in w.entries if p.startswith(depth["path"])]
    assert len(frames) == round(m["video"]["duration"] * m["video"]["fps"]) == 12


def test_otio_timeline_reads_back_with_all_tracks(bundle: Path) -> None:
    tl = otio.adapters.read_from_file(str(bundle / "timeline" / "main.otio"))
    kinds = [(t.kind, t.enabled) for t in tl.tracks]
    assert kinds[0] == (otio.schema.TrackKind.Video, True)  # main picture
    assert sum(k == otio.schema.TrackKind.Audio for k, _ in kinds) == 2  # one per stem
    assert all(not e for k, e in kinds[1:] if k == otio.schema.TrackKind.Video)  # layers off
    for clip in tl.find_clips():
        ref = clip.media_reference
        url = ref.target_url if hasattr(ref, "target_url") else ref.target_url_base
        assert (bundle / "timeline" / url).resolve().exists(), url


def test_mlt_paths_resolve_from_the_timeline_folder(bundle: Path) -> None:
    root = ET.parse(bundle / "timeline" / "main.mlt").getroot()
    for prod in root.iter("producer"):
        props = {p.get("name"): p.text or "" for p in prod.findall("property")}
        if props.get("mlt_service") != "color":  # the black background is not a file
            path = props["resource"].split("?")[0]
            if "%06d" in path:
                path = path.replace("%06d", "000000")
            assert (bundle / "timeline" / path).resolve().exists(), path
    hides = {t.get("producer"): t.get("hide") for t in root.iter("track")}
    assert hides["pl_main"] == "audio"  # stems carry the sound
    # Shotcut's project markers: without them it opens the file as one clip, not a timeline.
    tractor = root.find("tractor")
    assert tractor is not None
    assert any(p.get("name") == "shotcut" and p.text == "1" for p in tractor.findall("property"))
    assert next(iter(hides)) == "background"
    kinds = {
        pl.get("id"): {p.get("name") for p in pl.findall("property")}
        for pl in root.iter("playlist")
    }
    assert "shotcut:video" in kinds["pl_main"] and "shotcut:audio" in kinds["pl_stem_music"]


@pytest.mark.skipif(MELT is None, reason="melt (Shotcut / MLT) not installed")
def test_melt_renders_main_picture_and_exactly_the_stems(bundle: Path, tmp_path: Path) -> None:
    out = tmp_path / "a.wav"
    subprocess.run(
        [
            str(MELT),
            "-quiet",
            "main.mlt",
            "-consumer",
            f"avformat:{out}",
            "frequency=44100",
            "channels=1",
            "acodec=pcm_s16le",
            "vn=1",
        ],
        cwd=bundle / "timeline",
        check=True,
        timeout=120,
        capture_output=True,
    )
    m = wav_samples(out.read_bytes()).astype(np.float64)
    stems = [
        wav_samples((bundle / s).read_bytes()).astype(np.float64)
        for s in ("stems/ambience.wav", "stems/music.wav")
    ]
    n = min(len(m), *(len(s) for s in stems))
    gains, *_ = np.linalg.lstsq(np.stack([s[:n] for s in stems], 1), m[:n], rcond=None)
    assert np.allclose(gains, 1.0, atol=0.02)  # each stem once; the main clip's mix is muted

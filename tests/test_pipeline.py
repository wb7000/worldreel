from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import pytest

from formats.wreel import validate_wreel
from pipeline.io import FRAMES_JSON, PROXY_NPY, WORLD_WREEL, load_npy, read_json
from pipeline.registry import IMPLEMENTATIONS, VIDEO_PIPELINE, PipelineConfig, build_stages
from pipeline.runner import Runner, topological_order
from pipeline.stage import SOURCE, Impl, Item, Stage, StageContext, StageError
from stages.frames.real import FramesStage
from stages.safety.hook import SAFETY_LOG, PassThroughSafety
from tests.media import HAS_FFMPEG, make_clip


@pytest.fixture(autouse=True)
def _signing_key(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "WORLDREEL_SIGNING_KEY", str(tmp_path_factory.getbasetemp() / "test_key.pem")
    )


def make_item(
    tmp_path: Path, name: str = "clip.mp4", content: bytes = b"not really a video"
) -> Item:
    src = tmp_path / name
    src.write_bytes(content)
    return Item(
        id="ftest",
        sha256="ab" * 32,
        path=src,
        source="test",
        license="CC0-1.0",
        upload_rights="synthetic",
    )


def mock_runner(work: Path, stages: Sequence[Stage] | None = None, **kw: object) -> Runner:
    return Runner(
        work,
        stages
        if stages is not None
        else build_stages(VIDEO_PIPELINE, PipelineConfig.all_mock().impl),
        safety=PassThroughSafety(work),
        **kw,  # type: ignore[arg-type]
    )


class FailingStage:
    name: ClassVar[str] = "camera"
    version: ClassVar[str] = "0.0.1"
    impl: ClassVar[Impl] = "mock"
    depends_on: ClassVar[tuple[str, ...]] = ("frames",)

    def run(self, ctx: StageContext) -> None:
        (ctx.out / "partial.txt").write_text("half-written")
        raise StageError("depth model exploded")


# --------------------------------------------------------------------------- graph


def test_every_stage_has_a_mock() -> None:
    from stages import STAGES

    for name in STAGES:
        if name in ("ingest", "safety"):
            continue  # not pipeline stages: ingest is the data store, safety is the hook
        assert "mock" in IMPLEMENTATIONS[name], name


def test_topological_order_and_errors() -> None:
    stages = build_stages(VIDEO_PIPELINE, PipelineConfig.all_mock().impl)
    order = [s.name for s in topological_order(list(reversed(stages)))]
    for stage in stages:
        for dep in stage.depends_on:
            if dep != SOURCE:
                assert order.index(dep) < order.index(stage.name)
    with pytest.raises(ValueError, match="unknown stage"):
        topological_order([s for s in stages if s.name != "frames"])


def test_runner_refuses_to_run_without_safety_hook(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="safety hook"):
        Runner(tmp_path, [], safety=None)


# --------------------------------------------------------------------------- end to end on mocks


def test_full_mock_pipeline_produces_valid_wreel(tmp_path: Path) -> None:
    item = make_item(tmp_path)
    work = tmp_path / "work"
    result = mock_runner(work).run(item)
    assert result.ok, {n: r.error for n, r in result.stages.items()}
    assert set(result.stages) == set(VIDEO_PIPELINE)
    world = work / item.id / "package" / WORLD_WREEL
    report = validate_wreel(world)
    assert report.ok, report.issues
    status = read_json(work / item.id / "package" / "_stage.json")
    assert status["status"] == "done"
    assert set(status["inputs"]) == {"frames", "camera", "reconstruct", "actors", "fill", "audio"}
    events = [json.loads(line) for line in (work / item.id / SAFETY_LOG).read_text().splitlines()]
    assert {(e["stage"], e["event"]) for e in events} == {
        (s, ev) for s in VIDEO_PIPELINE for ev in ("input", "output")
    }


def test_second_run_is_fully_cached(tmp_path: Path) -> None:
    item = make_item(tmp_path)
    work = tmp_path / "work"
    mock_runner(work).run(item)
    again = mock_runner(work).run(item)
    assert again.ok
    assert all(r.cached for r in again.stages.values())


def test_config_change_reruns_stage_and_dependents(tmp_path: Path) -> None:
    item = make_item(tmp_path)
    work = tmp_path / "work"
    mock_runner(work).run(item)
    changed = mock_runner(work, config={"package": {"title": "Renamed"}}).run(item)
    rerun = {n for n, r in changed.stages.items() if not r.cached}
    assert rerun == {"package", "export"}


def test_identical_output_does_not_cascade(tmp_path: Path) -> None:
    """Cache keys use upstream *content* hashes: a re-run that produces the same bytes leaves
    dependents cached."""
    item = make_item(tmp_path)
    work = tmp_path / "work"
    mock_runner(work).run(item)
    changed = mock_runner(work, config={"fill": {"unused_option": 1}}).run(item)
    rerun = {n for n, r in changed.stages.items() if not r.cached}
    assert rerun == {"fill"}


def test_tampered_output_is_recomputed(tmp_path: Path) -> None:
    item = make_item(tmp_path)
    work = tmp_path / "work"
    mock_runner(work).run(item)
    (work / item.id / "camera" / "camera.json").write_text("{}")
    again = mock_runner(work).run(item)
    assert not again.stages["camera"].cached
    assert again.ok
    assert read_json(work / item.id / "camera" / "camera.json")["poses"]


def test_failure_is_isolated_and_leaves_no_partial_output(tmp_path: Path) -> None:
    item = make_item(tmp_path)
    work = tmp_path / "work"
    stages = [
        s
        for s in build_stages(VIDEO_PIPELINE, PipelineConfig.all_mock().impl)
        if s.name != "camera"
    ]
    result = mock_runner(work, [*stages, FailingStage()]).run(item)
    assert result.stages["camera"].status == "failed"
    assert "exploded" in result.stages["camera"].error
    for blocked in ("reconstruct", "fill", "package", "export"):
        assert result.stages[blocked].status == "failed"
        assert "blocked" in result.stages[blocked].error
    assert result.stages["frames"].status == "done"
    assert result.stages["audio"].status == "done"  # independent of camera
    assert result.stages["actors"].status == "done"  # people are found before camera tracking
    assert not (work / item.id / "camera" / "partial.txt").exists()
    assert read_json(work / item.id / "camera" / "_stage.json")["status"] == "failed"


def test_resume_after_crash(tmp_path: Path) -> None:
    item = make_item(tmp_path)
    work = tmp_path / "work"
    mock_runner(work).run(item)
    # Simulate a crash mid-way: a stage's temp folder was left behind and its output lost.
    crashed = work / item.id / ".fill.tmp-deadbeef"
    crashed.mkdir()
    (crashed / "junk").write_text("x")
    import shutil

    shutil.rmtree(work / item.id / "fill")
    resumed = mock_runner(work).run(item)
    assert resumed.ok
    assert not crashed.exists()
    assert not resumed.stages["fill"].cached
    assert resumed.stages["frames"].cached


def test_safety_hook_can_block(tmp_path: Path) -> None:
    item = make_item(tmp_path)
    work = tmp_path / "work"

    class Deny(PassThroughSafety):
        def decide(
            self, event: str, stage: str, item: Item, paths: Sequence[Path]
        ) -> tuple[bool, str]:
            return (stage != "reconstruct", "test policy")

    stages = build_stages(VIDEO_PIPELINE, PipelineConfig.all_mock().impl)
    result = Runner(work, stages, safety=Deny(work)).run(item)
    assert result.stages["reconstruct"].status == "failed"
    assert "safety" in result.stages["reconstruct"].error
    assert result.stages["package"].status == "failed"


def test_default_config_loads_and_selects_real_frames() -> None:
    cfg = PipelineConfig.load()
    assert cfg.impl["frames"] == "real"
    assert cfg.stage["frames"]["fps"] == 6.0
    assert cfg.impl["camera"] == "real" and cfg.impl["reconstruct"] == "real"
    with pytest.raises(ValueError, match="no 'real' implementation"):
        build_stages(("actors",), {"actors": "real"})


# --------------------------------------------------------------------------- real frames stage


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_real_frames_stage(tmp_path: Path) -> None:
    clip = make_clip(tmp_path / "clip.mp4", seconds=2.0)
    from pipeline.cache import file_sha256

    item = Item(id="freal", sha256=file_sha256(clip), path=clip)
    work = tmp_path / "work"
    stages = build_stages(VIDEO_PIPELINE, {**PipelineConfig.all_mock().impl, "frames": "real"})
    result = mock_runner(work, stages, config={"frames": {"fps": 5.0}}).run(item)
    assert result.ok, {n: r.error for n, r in result.stages.items()}
    out = work / item.id / "frames"
    meta = read_json(out / FRAMES_JSON)
    assert meta["has_audio"] is True
    assert meta["count"] == 10  # 2 s at 5 fps
    assert len(list((out / "frames").glob("*.png"))) == 10
    proxy = load_npy(out / PROXY_NPY)
    assert proxy.shape == (meta["proxy_count"], meta["proxy_height"], meta["proxy_width"], 3)
    assert (out / "audio.wav").stat().st_size > 1000
    assert validate_wreel(work / item.id / "package" / WORLD_WREEL).ok


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_real_frames_detects_scene_cut(tmp_path: Path) -> None:
    import subprocess

    from tools.toolpaths import find_tool

    a = make_clip(tmp_path / "a.mp4", seconds=1.0, pattern="testsrc", audio=False)
    b = make_clip(tmp_path / "b.mp4", seconds=1.0, pattern="smptebars", audio=False)
    joined = tmp_path / "cut.mp4"
    subprocess.run(
        [
            str(find_tool("ffmpeg")),
            "-v",
            "error",
            "-i",
            str(a),
            "-i",
            str(b),
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1[v]",
            "-map",
            "[v]",
            "-pix_fmt",
            "yuv420p",
            str(joined),
        ],
        check=True,
    )
    ctx_out = tmp_path / "out"
    ctx_out.mkdir()
    import logging

    FramesStage().run(
        StageContext(
            item=Item(id="fcut", sha256="0" * 64, path=joined),
            inputs={},
            out=ctx_out,
            config={"fps": 5.0},
            seed=0,
            log=logging.getLogger("test"),
        )
    )
    cuts = read_json(ctx_out / FRAMES_JSON)["cuts"]
    assert len(cuts) == 1
    assert cuts[0] == pytest.approx(1.0, abs=0.15)


def test_real_frames_without_ffmpeg_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import stages.frames.real as real

    monkeypatch.setattr(real, "find_tool", lambda name: None)
    item = make_item(tmp_path)
    stages = build_stages(VIDEO_PIPELINE, {**PipelineConfig.all_mock().impl, "frames": "real"})
    result = mock_runner(tmp_path / "work", stages).run(item)
    assert result.stages["frames"].status == "failed"
    assert "ffmpeg" in result.stages["frames"].error

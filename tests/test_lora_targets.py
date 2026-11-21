"""The LoRA target builder (D-9.26).

The v4 adapter was trained on a still portrait plus micro-jitter and it learned
exactly that: a still image that shakes. The fix is a target that contains real
motion and real texture, so the pieces that build that target are worth a test:
the takes must be split correctly, the variant must start with the exact
conditioning seed, and the motion of the variant must be the motion of the
rendered take -- not of a photograph.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.lora_targets import (
    Take,
    build_variants,
    fit_seed,
    motion_diffs,
    read_video,
    split_takes,
)


def _frames(n: int, value: int = 10, step: int = 0) -> list[np.ndarray]:
    return [np.full((8, 8, 3), value + i * step, dtype=np.uint8) for i in range(n)]


def test_split_takes_keeps_whole_takes() -> None:
    frames = _frames(10)
    takes = split_takes(frames, 4, 5)
    # The last, incomplete window is dropped -- a partial take is not a target.
    assert [t.start for t in takes] == [0, 4]
    assert [len(t.frames) for t in takes] == [4, 4]
    assert takes[0].index == 0


def test_split_takes_stops_at_the_end_of_the_video() -> None:
    takes = split_takes(_frames(5), 4, 5)
    assert len(takes) == 1
    assert len(takes[0].frames) == 4


def test_motion_diffs_detect_movement() -> None:
    still = motion_diffs(_frames(4, value=10))
    assert all(d == 0.0 for d in still)
    moving = motion_diffs(_frames(4, value=10, step=1))
    assert moving == [1.0, 1.0, 1.0]


def test_variant_starts_with_the_seed_and_keeps_the_take() -> None:
    frames = _frames(6, value=30)
    take = Take(index=0, start=0, frames=frames[:4])
    seed = np.full((8, 8, 3), 200, dtype=np.uint8)
    variants = build_variants([take], seed)
    assert len(variants) == 1
    variant = variants[0]
    assert len(variant) == 5
    assert np.array_equal(variant[0], seed)
    assert np.array_equal(variant[1], frames[0])
    assert np.array_equal(variant[4], frames[3])


def test_variant_motion_survives_the_seed() -> None:
    frames = _frames(4, value=30, step=1)
    take = Take(index=0, start=0, frames=frames)
    seed = np.zeros((8, 8, 3), dtype=np.uint8)
    variants = build_variants([take], seed)
    diffs = motion_diffs([np.asarray(f) for f in variants[0]])
    # The seed-to-frame-0 jump is the conditioning step; every later step is
    # the take's own motion.
    assert diffs[0] > 0.0
    assert diffs[1:] == [1.0, 1.0, 1.0]


def test_read_video_rejects_a_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        read_video("definitely-not-a-video.mp4")


def test_fit_seed_matches_the_render_job_crop() -> None:
    import tempfile

    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/p.png"
        Image.new("RGB", (512, 512), (10, 20, 30)).save(path)
        seed = fit_seed(path, 832, 480)
    assert seed.shape == (480, 832, 3)
    assert seed.dtype == np.uint8
    # A flat image stays flat whatever the crop is.
    assert set(np.unique(seed).tolist()) == {10, 20, 30}

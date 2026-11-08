import numpy as np

from stages.generate.deflicker import deflicker, smooth_flicker, steady_wall


def _scene() -> np.ndarray:
    rng = np.random.default_rng(1)
    base = np.full((120, 200, 3), 120.0)
    base += rng.normal(0, 12, (120, 200, 1))  # fine detail
    return base


def _flickering() -> list[np.ndarray]:
    gains = 1 + np.random.default_rng(2).normal(0, 0.03, (40, 3))
    base = _scene()
    return [np.clip(base * g, 0, 255).astype(np.uint8) for g in gains]


def _spread(frames: list[np.ndarray]) -> np.ndarray:
    return np.std(np.mean(np.stack(frames), axis=(1, 2)), axis=0)


def test_wall_step_removes_colour_change_and_keeps_detail() -> None:
    frames = _flickering()
    out = steady_wall(frames)
    assert _spread(out).max() < 0.1 * _spread(frames).min()
    detail = np.std(frames[20][:, :, 0])  # one channel: the colour shift itself is removed
    assert abs(np.std(out[20][:, :, 0]) - detail) < 0.15 * detail


def test_flicker_step_reduces_fast_change() -> None:
    frames = _flickering()
    assert _spread(smooth_flicker(frames, window=8)).max() < 0.5 * _spread(frames).min()


def test_change_on_the_subject_does_not_steer_the_wall() -> None:
    frames = [np.clip(_scene(), 0, 255).astype(np.uint8) for _ in range(8)]
    for i, f in enumerate(frames):  # a bright subject in the centre, moving
        f[40:120, 80 + 4 * i : 120 + 4 * i] = 230
    out = steady_wall(frames)
    wall = (out[3][:, :30].astype(float), frames[3][:, :30].astype(float))
    assert np.mean(np.abs(wall[0] - wall[1])) < 0.5


def test_steady_clip_is_unchanged() -> None:
    frame = np.clip(_scene(), 0, 255).astype(np.uint8)
    out = deflicker([frame] * 10)
    assert np.mean(np.abs(out[5].astype(float) - frame.astype(float))) < 0.5

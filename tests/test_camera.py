"""Tests for the director's-cut camera (stages/generate/camera.py)."""

from typing import Any

import numpy as np

from formats.wscript import Dialogue, Pov
from stages.generate.blocking import Blocking, plan_blocking
from stages.generate.camera import plan_camera
from stages.generate.places import build_place


def _scene_continuity() -> dict[str, Any]:
    return {
        "__location__": "L01",
        "characters": {"C01": {"location": "L01"}, "C02": {"location": "L01"}},
    }


def test_camera_spans_scene_with_wide_at_start() -> None:
    place = build_place(
        {
            "id": "L01",
            "name": "the fort gate",
            "description": "stone arch, iron chain, monsoon path",
        },
        7,
    )
    cont = _scene_continuity()
    stmts: list[Any] = [Dialogue("C01", "a"), Dialogue("C02", "b")]
    timing: list[dict[str, Any]] = [
        {"index": 0, "start": 0.0, "end": 4.0},
        {"index": 1, "start": 5.0, "end": 9.0},
    ]
    bl: Blocking = plan_blocking(stmts, cont, place, timing, 12.0)
    kfs = plan_camera(stmts, bl, place, timing, 12.0)
    assert kfs[0]["t"] == 0.0
    assert kfs[-1]["t"] == 12.0
    assert all("position" in kf and "rotation" in kf and "fov_deg" in kf for kf in kfs)
    # rotations are unit quaternions
    for kf in kfs:
        q = np.array(kf["rotation"])
        assert abs(np.linalg.norm(q) - 1.0) < 1e-6


def test_pov_over_puts_camera_behind_speaker() -> None:
    place = build_place(
        {
            "id": "L01",
            "name": "the fort gate",
            "description": "stone arch, iron chain, monsoon path",
        },
        7,
    )
    cont = _scene_continuity()
    stmts: list[Any] = [Pov("over", "C01")]
    bl: Blocking = plan_blocking(stmts, cont, place, None, 12.0)
    kfs = plan_camera(stmts, bl, place, None, 12.0)
    # one keyframe at t=0 (Pov at position 0 -> t spread over duration; single stmt -> t = 12/2 = 6)
    assert len(kfs) >= 2  # t=0 wide + the pov placement
    pov = next(kf for kf in kfs if kf["t"] > 0)
    # camera is behind C01's spot, i.e. closer to C01 than to C02 in x/z
    cp = np.array(pov["position"])
    c1 = np.array(bl.spots["C01"])
    c2 = np.array(bl.spots["C02"])
    assert np.linalg.norm(cp[[0, 2]] - c1[[0, 2]]) < np.linalg.norm(cp[[0, 2]] - c2[[0, 2]])

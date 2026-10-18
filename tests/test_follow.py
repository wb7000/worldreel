from __future__ import annotations

import numpy as np

from eval.geometry import quat_to_matrix
from stages.package.follow import FollowConfig, look_quaternion, plan_follow, seen_score


def _room() -> tuple[np.ndarray, np.ndarray]:
    """Walls at x, z = +-3 (no floor or ceiling, so some pixels stay empty): the +z wall is
    observed (confidence 1.0), the others guessed (0.1)."""
    g = np.linspace(-3, 3, 61)
    hy = np.linspace(0, 2.5, 26)
    walls, conf = [], []
    for y in hy:
        for s in g:
            walls += [[s, y, 3.0], [s, y, -3.0], [3.0, y, s], [-3.0, y, s]]
            conf += [1.0, 0.1, 0.1, 0.1]
    return np.array(walls), np.array(conf)


def test_look_quaternion_points_minus_z_at_target() -> None:
    q = look_quaternion(np.array([0.0, 1, 0]), np.array([0.0, 1, -5]), np.array([0.0, 1, 0]))
    m = quat_to_matrix(q)
    assert np.allclose(m @ [0, 0, -1], [0, 0, -1], atol=1e-6)


def test_seen_score_prefers_looking_at_the_observed_wall() -> None:
    pts, conf = _room()
    up = np.array([0.0, 1.0, 0.0])
    cfg = FollowConfig()
    toward_seen, clear1 = seen_score(
        np.array([0.0, 1.5, 0.0]), np.array([0.0, 1.0, 2.0]), up, pts, conf, cfg
    )
    toward_guess, clear2 = seen_score(
        np.array([0.0, 1.5, 0.0]), np.array([0.0, 1.0, -2.0]), up, pts, conf, cfg
    )
    assert clear1 and clear2
    assert toward_seen > 0.5 and toward_guess < 0.1


def test_planner_follows_from_the_side_that_sees_observed_space_and_is_smooth() -> None:
    pts, conf = _room()
    track = np.array([[x, 0.0, 0.0] for x in np.linspace(-1, 1, 6)])
    ups = np.tile([0.0, 1.0, 0.0], (6, 1))
    eyes, targets, scores = plan_follow(track, ups, 1.75, pts, conf)
    # The camera should sit on the -z side looking towards +z (the observed wall).
    assert np.all(eyes[:, 2] < targets[:, 2])
    # Better than the baseline that always sits behind (+z side, looking at the guessed wall).
    behind = [
        seen_score(
            t + np.array([0, 1.0, 1.75]),
            t + np.array([0, 0.96, 0]),
            ups[0],
            pts,
            conf,
            FollowConfig(),
        )[0]
        for t in track
    ]
    assert scores.mean() > 3 * np.mean(behind)
    # Smooth: no big jumps between consecutive samples.
    assert np.max(np.linalg.norm(np.diff(eyes, axis=0), axis=1)) < 1.0

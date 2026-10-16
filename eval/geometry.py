"""Camera-trajectory metrics. Monocular reconstruction is only defined up to a similarity
transform (rotation, translation, scale), so estimates are aligned to ground truth with Umeyama's
closed-form Sim(3) fit before measuring (Umeyama 1991, "Least-squares estimation of
transformation parameters between two point patterns")."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Sim3:
    scale: float
    rotation: F64  # (3, 3)
    translation: F64  # (3,)

    def apply(self, points: F64) -> F64:
        return self.scale * points @ self.rotation.T + self.translation

    def apply_rotation(self, rotations: F64) -> F64:
        """Rotate camera-to-world rotation matrices (N, 3, 3) into the aligned frame."""
        out: F64 = np.einsum("ij,njk->nik", self.rotation, rotations)
        return out


def umeyama(src: F64, dst: F64, *, with_scale: bool = True) -> Sim3:
    """Best Sim(3) mapping src points (N, 3) onto dst points (N, 3) in the least-squares sense."""
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("need at least 3 corresponding points of shape (N, 3)")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1
    rot = u @ sign @ vt
    var_s = (xs**2).sum() / len(src)
    scale = float(np.trace(np.diag(d) @ sign) / var_s) if with_scale and var_s > 0 else 1.0
    trans = mu_d - scale * rot @ mu_s
    return Sim3(scale, rot, trans)


def rotation_angle_deg(r: F64) -> float:
    """Angle of a rotation matrix in degrees."""
    cos = (np.trace(r) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


@dataclass(frozen=True)
class TrajectoryErrors:
    ate_rmse: float  # metres, after Sim(3) alignment
    ate_rel: float  # ate_rmse / ground-truth trajectory length
    rot_mean_deg: float  # mean absolute orientation error after alignment
    rpe_trans_rel: float  # mean relative translation error between consecutive frames (fraction)
    rpe_rot_deg: float  # mean relative rotation error between consecutive frames
    scale: float  # alignment scale (est units → metres)
    frames: int


def trajectory_errors(
    est_pos: F64, est_rot: F64, gt_pos: F64, gt_rot: F64
) -> tuple[TrajectoryErrors, Sim3]:
    """Positions (N, 3) and camera-to-world rotations (N, 3, 3) of the same frames.

    The alignment uses each camera's position *and* points one tenth of the trajectory length
    along its three axes. Positions alone leave the rotation poorly determined when the path is
    close to a line (a short arc), which would show up as a spurious constant orientation error.
    """
    gt_len = float(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1).sum()) or 1.0
    est_len = float(np.linalg.norm(np.diff(est_pos, axis=0), axis=1).sum()) or 1.0
    axes = np.eye(3)
    src = np.concatenate([est_pos] + [est_pos + 0.1 * est_len * est_rot @ a for a in axes])
    dst = np.concatenate([gt_pos] + [gt_pos + 0.1 * gt_len * gt_rot @ a for a in axes])
    sim = umeyama(src, dst)
    aligned_pos = sim.apply(est_pos)
    aligned_rot = sim.apply_rotation(est_rot)
    err = np.linalg.norm(aligned_pos - gt_pos, axis=1)
    ate = float(np.sqrt((err**2).mean()))
    length = float(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1).sum())
    rot_err = [rotation_angle_deg(g.T @ e) for g, e in zip(gt_rot, aligned_rot, strict=True)]
    rpe_t, rpe_r = [], []
    for i in range(len(gt_pos) - 1):
        gt_rel_r = gt_rot[i].T @ gt_rot[i + 1]
        est_rel_r = aligned_rot[i].T @ aligned_rot[i + 1]
        gt_rel_t = gt_rot[i].T @ (gt_pos[i + 1] - gt_pos[i])
        est_rel_t = aligned_rot[i].T @ (aligned_pos[i + 1] - aligned_pos[i])
        step = np.linalg.norm(gt_rel_t)
        if step > 1e-9:
            rpe_t.append(float(np.linalg.norm(est_rel_t - gt_rel_t) / step))
        rpe_r.append(rotation_angle_deg(gt_rel_r.T @ est_rel_r))
    return (
        TrajectoryErrors(
            ate_rmse=ate,
            ate_rel=ate / length if length > 0 else float("inf"),
            rot_mean_deg=float(np.mean(rot_err)),
            rpe_trans_rel=float(np.mean(rpe_t)) if rpe_t else 0.0,
            rpe_rot_deg=float(np.mean(rpe_r)) if rpe_r else 0.0,
            scale=sim.scale,
            frames=len(gt_pos),
        ),
        sim,
    )


def quat_to_matrix(q: npt.ArrayLike) -> F64:
    x, y, z, w = np.asarray(q, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(r: F64) -> list[float]:
    """Rotation matrix → unit quaternion [x, y, z, w] (Shepperd's method)."""
    m = r
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = (
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (
            (m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        )
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
        )
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (
            (m[1, 0] - m[0, 1]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
        )
    q = np.array([x, y, z, w], dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return [float(c) for c in q]

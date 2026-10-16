"""Test-time localisation of a held-out image inside a reconstruction.

Novel-view metrics through a single global Sim(3) alignment mix two errors: the quality of the
reconstruction and any residual pose bias (measured separately by the trajectory metrics). The
standard way to isolate the former (as in BARF, Lin et al. 2021, and NoPe-NeRF, Bian et al. 2023,
which optimise test poses before computing image metrics) is to first locate the test camera
within the reconstruction. Here that is done classically: SIFT matches between the test image
and nearby reconstructed frames, mapped to triangulated points through the stored observations,
then PnP with RANSAC and Levenberg-Marquardt refinement.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from eval.geometry import matrix_to_quat, quat_to_matrix

_FLIP = np.diag([1.0, -1.0, -1.0])


def localize(
    image: npt.NDArray[np.uint8],
    frames: list[npt.NDArray[np.uint8]],
    frame_ids: list[int],
    points: npt.NDArray[np.floating],
    observations: npt.NDArray[np.floating],
    k: dict[str, float],
    *,
    max_obs_px: float = 1.5,
) -> tuple[list[float], list[float], int] | None:
    """Pose (position, rotation in WorldReel axes, reconstruction frame) of `image`, plus the
    PnP inlier count; None if it cannot be localised. `frames` are RGB images of the
    reconstruction's frames `frame_ids`; `observations` rows are (frame, point, u, v)."""
    sift = cv2.SIFT.create(nfeatures=4000)
    kq, dq = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None)
    if dq is None:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    obj, img = [], []
    for fid, frame in zip(frame_ids, frames, strict=True):
        kf, df = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), None)
        if df is None:
            continue
        obs = observations[observations[:, 0] == fid]
        if len(obs) == 0:
            continue
        for pair in matcher.knnMatch(dq, df, k=2):
            if len(pair) < 2 or pair[0].distance >= 0.75 * pair[1].distance:
                continue
            u, v = kf[pair[0].trainIdx].pt
            d = np.hypot(obs[:, 2] - u, obs[:, 3] - v)
            nearest = int(np.argmin(d))
            if d[nearest] <= max_obs_px:
                obj.append(points[int(obs[nearest, 1])])
                img.append(kq[pair[0].queryIdx].pt)
    if len(obj) < 12:
        return None
    kmat = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]], np.float64)
    obj_cv = np.asarray(obj, np.float64) @ _FLIP.T  # WorldReel → OpenCV axes
    pts2d = np.asarray(img, np.float64)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_cv,
        pts2d,
        kmat,
        None,
        iterationsCount=500,
        reprojectionError=2.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < 12:
        return None
    idx = inliers.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(obj_cv[idx], pts2d[idx], kmat, None, rvec, tvec)
    r_cw = np.asarray(cv2.Rodrigues(rvec)[0], np.float64)
    r_wc = r_cw.T
    center = -r_wc @ tvec.ravel()
    position = [float(c) for c in _FLIP @ center]
    rotation = matrix_to_quat(_FLIP @ r_wc @ _FLIP)
    return position, rotation, len(idx)


def pose_error(
    pose_a: tuple[list[float], list[float]], pose_b: tuple[list[float], list[float]]
) -> tuple[float, float]:
    """(position distance, rotation angle in degrees) between two poses in the same frame."""
    dist = float(np.linalg.norm(np.subtract(pose_a[0], pose_b[0])))
    r = quat_to_matrix(pose_a[1]).T @ quat_to_matrix(pose_b[1])
    angle = float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))
    return dist, angle

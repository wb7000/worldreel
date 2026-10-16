"""Depth metrics (the standard set from Eigen et al. 2014: AbsRel, RMSE, δ < 1.25)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class DepthErrors:
    abs_rel: float
    rmse: float
    delta1: float  # fraction of pixels with max(est/gt, gt/est) < 1.25
    completeness: float  # fraction of ground-truth pixels that have an estimate
    count: int


def depth_errors(est: npt.NDArray[np.floating], gt: npt.NDArray[np.floating]) -> DepthErrors:
    """Compare depth maps of equal shape; zero means "no value" in either map."""
    if est.shape != gt.shape:
        raise ValueError(f"shape mismatch {est.shape} vs {gt.shape}")
    gt_valid = gt > 0
    both = gt_valid & (est > 0)
    n = int(both.sum())
    completeness = float(n / max(1, int(gt_valid.sum())))
    if n == 0:
        return DepthErrors(float("inf"), float("inf"), 0.0, completeness, 0)
    e = est[both].astype(np.float64)
    g = gt[both].astype(np.float64)
    ratio = np.maximum(e / g, g / e)
    return DepthErrors(
        abs_rel=float(np.mean(np.abs(e - g) / g)),
        rmse=float(np.sqrt(np.mean((e - g) ** 2))),
        delta1=float(np.mean(ratio < 1.25)),
        completeness=completeness,
        count=n,
    )

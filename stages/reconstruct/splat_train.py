"""3D Gaussian Splatting training (Kerbl et al. 2023) with the gsplat library (Apache-2.0).

GPU only: imports torch and gsplat lazily so the rest of the package works without them.
Camera convention: OpenCV (x right, y down, z forward), world-to-camera view matrices, as
measured for gsplat on Kaggle (DECISIONS D-3b.2).

The loop follows the paper's recipe: Gaussians initialised from SfM points with isotropic scales
from the 3 nearest neighbours, Adam with per-parameter learning rates and an exponentially
decaying position learning rate, loss = 0.8·L1 + 0.2·(1 - SSIM), and gsplat's DefaultStrategy for
densification (clone / split / prune, opacity reset). Colours are plain RGB (no spherical
harmonics) in v1.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class SplatConfig:
    steps: int = 7000
    sh_free_rgb: bool = True
    lr_means: float = 1.6e-4  # times scene extent, decays to 1/100 over training
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_colors: float = 2.5e-3
    ssim_weight: float = 0.2
    init_opacity: float = 0.1
    refine_start: int = 500
    refine_stop: int = 5000
    refine_every: int = 100
    reset_every: int = 3000
    checkpoint_every: int = 2000
    background_points: int = 100000  # far sphere for sky/background (0 = off)
    background_radius: float = 10.0  # times the scene extent
    # gsplat prunes Gaussians larger than prune_scale3d x scene extent. Keep gsplat's default
    # (0.1): raising it to protect the background let large foreground blobs survive and veil
    # whole views (seed 13, DECISIONS D-3b.4). Background Gaussians start below the limit instead.
    prune_scale3d: float = 0.1
    background_max_scale: float = 0.05  # initial background Gaussian size, times the extent
    seed: int = 0


@dataclass
class SplatResult:
    params: dict[str, npt.NDArray[np.float32]]
    stats: dict[str, Any] = field(default_factory=dict)


def _ssim(x: Any, y: Any) -> Any:
    """SSIM between image batches [B, 3, H, W] in [0, 1] (11x11 Gaussian window, sigma 1.5)."""
    import torch
    import torch.nn.functional as nnf

    coords = torch.arange(11, device=x.device, dtype=x.dtype) - 5
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    g = g / g.sum()
    window = (g[:, None] @ g[None, :]).expand(3, 1, 11, 11).contiguous()

    def blur(z: Any) -> Any:
        return nnf.conv2d(z, window, padding=5, groups=3)

    mx, my = blur(x), blur(y)
    sxx = blur(x * x) - mx * mx
    syy = blur(y * y) - my * my
    sxy = blur(x * y) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return s.mean()


def _knn_scale(points: npt.NDArray[np.float64], k: int = 3) -> npt.NDArray[np.float64]:
    """Mean distance to the k nearest neighbours (brute force in chunks; fine for ≤ 100k)."""
    import torch

    p = torch.as_tensor(points, dtype=torch.float32, device="cuda")
    out = torch.empty(len(p), device="cuda")
    for i in range(0, len(p), 4096):
        d = torch.cdist(p[i : i + 4096], p)
        d, _ = torch.topk(d, k + 1, largest=False)
        out[i : i + 4096] = d[:, 1:].mean(1)
    return np.asarray(out.clamp_min(1e-4).cpu().numpy(), dtype=np.float64)


def train_splats(
    images: list[npt.NDArray[np.uint8]],
    k: npt.NDArray[np.float64],
    viewmats: npt.NDArray[np.float64],
    init_points: npt.NDArray[np.float64],
    init_colors: npt.NDArray[np.uint8],
    cfg: SplatConfig,
    *,
    checkpoint_dir: Path | None = None,
    log: Any = print,
    ignore: list[npt.NDArray[np.bool_] | None] | None = None,
) -> SplatResult:
    """`ignore[i]` (H x W bool) marks pixels of frame i the loss must skip, e.g. moving people."""
    import torch
    from gsplat import DefaultStrategy, rasterization

    torch.manual_seed(cfg.seed)
    dev = torch.device("cuda")
    h, w = images[0].shape[:2]
    gt = torch.as_tensor(np.stack(images), device=dev).float() / 255.0  # [N, H, W, 3]
    vm = torch.as_tensor(viewmats, dtype=torch.float32, device=dev)
    kk = torch.as_tensor(k, dtype=torch.float32, device=dev)
    masks = (
        [None if m is None else torch.as_tensor(m, device=dev)[..., None] for m in ignore]
        if ignore is not None
        else [None] * len(images)
    )

    centers = -np.einsum("nji,nj->ni", viewmats[:, :3, :3], viewmats[:, :3, 3])
    # Scene extent from the 95th percentile of point distances, not the maximum: a few stray SfM
    # points inflated it threefold on seed 13 (15.9 vs ~5), which scales the prune limit and the
    # position learning rate (DECISIONS D-3b.5).
    centre = np.median(init_points, axis=0)
    extent = float(np.percentile(np.linalg.norm(init_points - centre, axis=1), 95)) * 1.1
    extent = max(extent, float(np.linalg.norm(centers - centers.mean(0), axis=1).max()))

    # Background sphere: SfM finds no points in the sky or far background, so without a place
    # to put them the optimiser fits sky colour with "floaters" close to the cameras, which then
    # block novel views (measured on seed 11, DECISIONS D-3b.4). Seeding Gaussians on a far
    # sphere, a common practice for unbounded scenes, gives the background its own geometry.
    if cfg.background_points > 0:
        rng_bg = np.random.default_rng(cfg.seed + 1)
        dirs = rng_bg.normal(size=(cfg.background_points, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        far = centre + dirs * extent * cfg.background_radius
        mean_rgb = np.stack(images).reshape(-1, 3).mean(0)
        init_points = np.concatenate([init_points, far])
        init_colors = np.concatenate(
            [init_colors, np.tile(mean_rgb.round().astype(np.uint8), (len(far), 1))]
        )
    n = len(init_points)
    sizes = _knn_scale(init_points)
    if cfg.background_points > 0:
        sizes[-cfg.background_points :] = np.minimum(
            sizes[-cfg.background_points :], cfg.background_max_scale * extent
        )
    scales = np.log(sizes)[:, None].repeat(3, 1)
    colors = np.clip(init_colors.astype(np.float64) / 255.0, 1e-3, 1 - 1e-3)
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(
                torch.as_tensor(init_points, dtype=torch.float32, device=dev)
            ),
            "scales": torch.nn.Parameter(torch.as_tensor(scales, dtype=torch.float32, device=dev)),
            "quats": torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]], device=dev).repeat(n, 1)),
            "opacities": torch.nn.Parameter(
                torch.full((n,), math.log(cfg.init_opacity / (1 - cfg.init_opacity)), device=dev)
            ),
            "colors": torch.nn.Parameter(
                torch.logit(torch.as_tensor(colors, dtype=torch.float32, device=dev))
            ),
        }
    )
    lrs = {
        "means": cfg.lr_means * extent,
        "scales": cfg.lr_scales,
        "quats": cfg.lr_quats,
        "opacities": cfg.lr_opacities,
        "colors": cfg.lr_colors,
    }
    optimizers = {key: torch.optim.Adam([params[key]], lr=lr, eps=1e-15) for key, lr in lrs.items()}
    decay = 0.01 ** (1.0 / cfg.steps)
    strategy = DefaultStrategy(
        refine_start_iter=cfg.refine_start,
        refine_stop_iter=cfg.refine_stop,
        refine_every=cfg.refine_every,
        reset_every=cfg.reset_every,
        pause_refine_after_reset=len(images),
        prune_scale3d=cfg.prune_scale3d,
    )
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=extent)

    rng = np.random.default_rng(cfg.seed)
    losses: list[float] = []
    t0 = time.time()
    for step in range(cfg.steps):
        i = int(rng.integers(len(images)))
        render, _, info = rasterization(
            params["means"],
            params["quats"],
            torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]),
            torch.sigmoid(params["colors"]),
            vm[i : i + 1],
            kk[None],
            w,
            h,
            packed=False,
        )
        strategy.step_pre_backward(params, optimizers, state, step, info)
        pred = render[0]
        target = gt[i]
        if masks[i] is not None:
            # Ignored pixels take the (detached) prediction as their target, so they add no error
            # or gradient to L1 or SSIM. People move, so the static model must not explain them.
            mask = masks[i]
            assert mask is not None  # narrowed by the check above
            target = torch.where(mask, pred.detach(), target)
        l1 = (pred - target).abs().mean()
        ssim = _ssim(pred.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None])
        loss = (1 - cfg.ssim_weight) * l1 + cfg.ssim_weight * (1 - ssim)
        loss.backward()
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        for group in optimizers["means"].param_groups:
            group["lr"] *= decay
        losses.append(float(loss))
        if step % 500 == 0 or step == cfg.steps - 1:
            log(
                f"step {step:5d} loss {np.mean(losses[-100:]):.4f} gaussians {len(params['means'])}"
            )
        if checkpoint_dir and step and step % cfg.checkpoint_every == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {k2: v.detach().cpu() for k2, v in params.items()} | {"step": step},
                checkpoint_dir / "latest.pt",
            )

    out = {key: v.detach().cpu().numpy().astype(np.float32) for key, v in params.items()}
    return SplatResult(
        out,
        {
            "steps": cfg.steps,
            "seconds": round(time.time() - t0, 1),
            "gaussians": len(out["means"]),
            "initial_points": n,
            "final_loss": float(np.mean(losses[-100:])),
            "loss_every_500": [
                float(np.mean(losses[max(0, s - 100) : s + 1])) for s in range(0, cfg.steps, 500)
            ],
            "scene_extent": extent,
            "config": asdict(cfg),
        },
    )


def render_splats(
    params: dict[str, npt.NDArray[np.float32]],
    k: npt.NDArray[np.float64],
    viewmat: npt.NDArray[np.float64],
    width: int,
    height: int,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]:
    """Render one view → (RGB uint8 image, alpha)."""
    import torch
    from gsplat import rasterization

    dev = torch.device("cuda")
    p = {key: torch.as_tensor(v, device=dev) for key, v in params.items()}
    with torch.no_grad():
        img, alpha, _ = rasterization(
            p["means"],
            p["quats"],
            torch.exp(p["scales"]),
            torch.sigmoid(p["opacities"]),
            torch.sigmoid(p["colors"]),
            torch.as_tensor(viewmat, dtype=torch.float32, device=dev)[None],
            torch.as_tensor(k, dtype=torch.float32, device=dev)[None],
            width,
            height,
            packed=False,
            backgrounds=torch.tensor([background], device=dev),
        )
    rgb = (img[0].clamp(0, 1) * 255).round().byte().cpu().numpy()
    return rgb, alpha[0, :, :, 0].cpu().numpy()

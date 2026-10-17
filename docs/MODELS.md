# Model register

Every pretrained model, dataset-derived checkpoint or library with model weights used by WorldReel
is listed here **before** it is used. Weights are never committed.

Rules:
- Only open-licensed weights. Record the exact license name and a link to the license text.
- Record the version / commit / checkpoint name exactly as published. Do not guess.
- If something cannot be verified, put `UNVERIFIED` in the License column and do not ship it.

| Stage | Model | Version / checkpoint | License | Source URL | VRAM / CPU notes | Verified on | Status |
|---|---|---|---|---|---|---|---|
| reconstruct (splats) | gsplat library (no pretrained weights; trained per scene) | 1.5.3 (PyPI) | Apache-2.0 | https://github.com/nerfstudio-project/gsplat | 1x T4 16 GB: ~100–420 s / scene, 0.3–0.5 M Gaussians | 2026-10-17, Kaggle | in-use |

Status values: `candidate`, `in-use`, `rejected` (say why in `DECISIONS.md`).

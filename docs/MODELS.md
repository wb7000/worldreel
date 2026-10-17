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
| actors (detection) | DETR ResNet-50 (`facebook/detr-resnet-50`, revision `no_timm`) | Hugging Face transformers 5.16.1 on Kaggle | Apache-2.0 | https://huggingface.co/facebook/detr-resnet-50 | runs on 1x T4; config needs `dilation` null -> False under strict validation | 2026-10-17, Kaggle ("person" = id 1) | in-use (keyframes) |
| actors (video masks) | SAM 2.1 hiera_small video predictor | checkpoint `092824/sam2.1_hiera_small.pt` (184.4 MB), config `configs/sam2.1/sam2.1_hiera_s.yaml` | Apache-2.0 | https://github.com/facebookresearch/sam2 | 1x T4; frames offloaded to CPU | 2026-10-17, Kaggle (IoU 0.92-0.94 vs GT from a GT box, 8 frames) | in-use |

Status values: `candidate`, `in-use`, `rejected` (say why in `DECISIONS.md`).

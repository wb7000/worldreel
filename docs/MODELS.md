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
| fill / retrieval (people filter) | YuNet face detector (OpenCV model zoo) | `face_detection_yunet_2023mar.onnx`, 232,589 bytes, SHA-256 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4; run with OpenCV 5.0.0 `FaceDetectorYN` | MIT (folder licence in opencv_zoo) | https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet | CPU, instant; stored in git-ignored `data/models/` | 2026-10-18, laptop (loads; no false positive on synthetic scenes; detection of real faces not yet measured) | in-use (reference filter) |
| fill (generative texture) | LaMa big-lama, ONNX port `Carve/LaMa-ONNX` `lama_fp32.onnx` | 208,044,816 bytes, SHA-256 1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6, opset 17, fixed 512x512; onnxruntime 1.31.0 (MIT) | Apache-2.0 (code: advimman/lama LICENSE; weights: model cards smartywu/big-lama and Carve/LaMa-ONNX) | https://huggingface.co/Carve/LaMa-ONNX | CPU ~12 s per 512x512; git-ignored `data/models/` | 2026-10-18, laptop (input [0,1], mask 1 = hole, output [0,255], measured) | experimental (opt-in `--generative`) |

Status values: `candidate`, `in-use`, `rejected` (say why in `DECISIONS.md`).

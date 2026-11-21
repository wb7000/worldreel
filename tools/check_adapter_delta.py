"""Is the v5 'best' adapter a no-op?

PEFT initialises LoRA B to zero, so a snapshot taken before the first optimiser
step has delta = B@A = 0 -> the adapter is exactly the base model. The v5
report says best == baseline (step 0), so lora_adapter_best should be that
zero-delta snapshot. Verify it from the weights on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/kaggle/character_lora_v5/out")


def summarise(path: Path) -> tuple[int, float, float]:
    sd = load_file(str(path))
    n_zero = 0
    total = 0
    max_b = 0.0
    for key, tensor in sd.items():
        if not key.endswith("lora_B.weight"):
            continue
        total += 1
        norm = float(tensor.float().norm())
        max_b = max(max_b, norm)
        if norm == 0.0:
            n_zero += 1
    return total, max_b, float(n_zero)


for name in sorted(p.name for p in root.iterdir() if p.is_dir()):
    weight = root / name / "pytorch_lora_weights.safetensors"
    if not weight.exists():
        continue
    total, max_b, n_zero = summarise(weight)
    print(
        f"{name:24s} B tensors={total:4d} zero={int(n_zero):4d} max|B|={max_b:.6e} "
        f"delta={'ZERO (no-op == base model)' if max_b == 0.0 else 'non-zero'}"
    )

# Cross-check: does lora_adapter_best differ from the other snapshots at all?
best = root / "lora_adapter_best" / "pytorch_lora_weights.safetensors"
if best.exists():
    b = load_file(str(best))
    for other in sorted(root.glob("lora_adapter_step*")):
        w = other / "pytorch_lora_weights.safetensors"
        if not w.exists():
            continue
        o = load_file(str(w))
        if set(b) != set(o):
            print(f"{other.name}: different key sets")
            continue
        diff = max(
            float((b[k].float() - o[k].float()).abs().max())
            for k in b
            if b[k].shape == o[k].shape
        )
        print(f"max |best - {other.name}| = {diff:.6e}")
        print(f"torch={torch.__version__}")

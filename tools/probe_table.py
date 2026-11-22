"""Print the probe results as one table."""

from __future__ import annotations

import json
from pathlib import Path

path = Path("runs/kaggle/character_render_lora_probe/out/probe.json")
data = json.loads(path.read_text(encoding="utf-8"))
for r in data["results"]:
    print(
        f"{r['name']:14s} refs={r['num_references']} seed={r['seed']} "
        f"id_mean={r['identity_mean']:.4f} id_min={r['identity_min']:.4f} "
        f"first={r['identity_first']:.4f} last={r['identity_last']:.4f} "
        f"drift={r['identity_drift_per_frame']:+.5f}/frame "
        f"self={r['self_consistency_mean']:.4f}/{r['self_consistency_min']:.4f} "
        f"sharp={r['sharpness_mean']:.1f} motion={r['motion_mean']:.3f} "
        f"still={r['still_frames']}/{r['still_total']} faces={r['faces_found']}/{r['frames']} "
        f"{r['seconds']:.0f}s"
    )

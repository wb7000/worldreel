# Job: check the Kaggle GPU environment for Phase 3b and measure gsplat's camera convention.
# Runs after bootstrap.py (log, save_json, OUT, CODE are defined there).
import subprocess
import sys

report: dict[str, object] = {}
report["nvidia_smi"] = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
    capture_output=True,
    text=True,
).stdout.strip()
log(f"GPU: {report['nvidia_smi']}")

import torch

report["python"] = sys.version.split()[0]
report["torch"] = torch.__version__
report["torch_cuda"] = torch.version.cuda
report["cuda_available"] = torch.cuda.is_available()
report["device_capability"] = (
    list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
)
log(
    f"python {report['python']}, torch {report['torch']} (CUDA {report['torch_cuda']}), available {report['cuda_available']}"
)

t = __import__("time").time()
pip = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "gsplat==1.5.3"], capture_output=True, text=True
)
report["pip_gsplat"] = {
    "returncode": pip.returncode,
    "seconds": round(__import__("time").time() - t, 1),
    "stderr_tail": pip.stderr[-1500:],
}
log(f"pip install gsplat: rc={pip.returncode} in {report['pip_gsplat']['seconds']}s")

try:
    t = __import__("time").time()
    import gsplat
    from gsplat import rasterization

    report["gsplat"] = gsplat.__version__
    dev = torch.device("cuda")
    w, h, f = 64, 48, 50.0
    K = torch.tensor([[[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]]], device=dev)
    view = torch.eye(4, device=dev)[None]

    def render(center: list[float]) -> torch.Tensor:
        means = torch.tensor([center], device=dev)
        quats = torch.tensor([[1.0, 0, 0, 0]], device=dev)
        scales = torch.full((1, 3), 0.3, device=dev)
        opac = torch.tensor([1.0], device=dev)
        colors = torch.tensor([[1.0, 0.0, 0.0]], device=dev)
        _img, alpha, _ = rasterization(means, quats, scales, opac, colors, view, K, w, h)
        return alpha[0, :, :, 0]

    a_front = render([0.0, 0.0, 5.0])
    a_back = render([0.0, 0.0, -5.0])
    a_down = render([0.0, 1.5, 5.0])  # +y in camera space
    ys, xs = torch.nonzero(a_down > 0.5, as_tuple=True)
    report["first_call_seconds"] = round(__import__("time").time() - t, 1)
    report["convention"] = {
        "visible_at_plus_z": float(a_front.max()),
        "visible_at_minus_z": float(a_back.max()),
        "plus_y_appears_at_row": float(ys.float().mean()) if len(ys) else None,
        "image_height": h,
        "interpretation": "OpenCV (x right, y down, z forward) if +z is visible and +y renders below the centre row",
    }
    log(f"convention: {report['convention']}")
except Exception as exc:
    report["gsplat_error"] = repr(exc)
    log(f"gsplat failed: {exc!r}")

# Our own code imports and a quick CPU self-test of the formats package.
try:
    from formats.wreel import FORMAT_VERSION

    report["worldreel_import"] = f"ok (wreel {FORMAT_VERSION})"
except Exception as exc:
    report["worldreel_import"] = repr(exc)
log(f"worldreel import: {report['worldreel_import']}")
save_json("env_check.json", report)
log("done")

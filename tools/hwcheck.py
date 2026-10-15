"""Record the hardware of the current machine.

Usage: ``python -m tools.hwcheck [--out hardware.local.json]``

Uses only the standard library plus whatever system tools exist (``nvidia-smi``, PowerShell CIM,
``/proc``, ``sysctl``). Missing tools are reported as missing, never guessed.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class GpuInfo:
    name: str
    vendor: str
    vram_mb: int | None = None
    cuda: bool = False


@dataclass
class HardwareReport:
    timestamp: str
    os: str
    python: str
    cpu: str
    logical_cpus: int | None
    ram_gb: float | None
    gpus: list[GpuInfo] = field(default_factory=list)
    has_cuda_gpu: bool = False
    tools: dict[str, bool] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _run(cmd: list[str]) -> str | None:
    """Run a command and return stdout, or None if it is missing or fails."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def parse_nvidia_smi(output: str) -> list[GpuInfo]:
    """Parse ``nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits``."""
    gpus: list[GpuInfo] = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0]:
            continue
        vram = int(float(parts[1])) if parts[1].replace(".", "", 1).isdigit() else None
        gpus.append(GpuInfo(name=parts[0], vendor="NVIDIA", vram_mb=vram, cuda=True))
    return gpus


def _vendor_from_name(name: str) -> str:
    lowered = name.lower()
    for vendor in ("nvidia", "amd", "intel", "apple"):
        if vendor in lowered:
            return vendor.upper() if vendor == "amd" else vendor.capitalize()
    if "radeon" in lowered:
        return "AMD"
    return "unknown"


def _other_gpus() -> list[GpuInfo]:
    """Non-CUDA GPUs as reported by the OS. VRAM is not reported (unreliable for iGPUs)."""
    names: list[str] = []
    system = platform.system()
    if system == "Windows":
        out = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_VideoController).Name",
            ]
        )
        names = [n.strip() for n in (out or "").splitlines() if n.strip()]
    elif system == "Linux":
        out = _run(["lspci"])
        names = [
            line.split(": ", 1)[-1]
            for line in (out or "").splitlines()
            if "VGA" in line or "3D controller" in line
        ]
    elif system == "Darwin":
        out = _run(["system_profiler", "SPDisplaysDataType"])
        names = [
            line.split(":", 1)[1].strip()
            for line in (out or "").splitlines()
            if "Chipset Model" in line
        ]
    return [GpuInfo(name=n, vendor=_vendor_from_name(n)) for n in names]


def _cpu_name() -> str:
    system = platform.system()
    if system == "Windows":
        out = _run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]
        )
        if out and out.strip():
            return out.strip().splitlines()[0].strip()
    elif system == "Linux":
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    elif system == "Darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out.strip()
    return platform.processor() or "unknown"


def _ram_gb() -> float | None:
    system = platform.system()
    if sys.platform == "win32":

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return round(float(status.ullTotalPhys) / 1024**3, 1)
        return None
    if system == "Linux":
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            for line in meminfo.read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024**2, 1)
    if system == "Darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        if out and out.strip().isdigit():
            return round(int(out.strip()) / 1024**3, 1)
    return None


def collect() -> HardwareReport:
    smi = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    cuda_gpus = parse_nvidia_smi(smi) if smi else []
    cuda_names = {g.name for g in cuda_gpus}
    other = [g for g in _other_gpus() if g.name not in cuda_names]
    return HardwareReport(
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        os=f"{platform.system()} {platform.release()} ({platform.version()})",
        python=sys.version.split()[0],
        cpu=_cpu_name(),
        logical_cpus=os.cpu_count(),
        ram_gb=_ram_gb(),
        gpus=cuda_gpus + other,
        has_cuda_gpu=bool(cuda_gpus),
        tools={t: shutil.which(t) is not None for t in ("nvidia-smi", "ffmpeg", "blender", "git")},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("hardware.local.json"))
    args = parser.parse_args(argv)
    report = collect()
    text = report.to_json()
    args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not report.has_cuda_gpu:
        print("\nNo CUDA GPU found: GPU stages must use mocks/CPU locally or run on cloud GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

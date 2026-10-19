# Job: fine-tune the script model with Soup and evaluate it (D-7.5).
# Runs after bootstrap.py (log, save_json, OUT, CODE, INPUT, WORK are defined there).
# Attach the data set: python -m tools.kaggle_runner run training/kaggle/jobs/script_sft.py
#                          --scene data:scriptsft
# Kaggle's Python is 3.13; Soup supports 3.10-3.12, so Soup runs in a uv-made Python 3.12 env.
import shutil
import subprocess
import sys
import zipfile

SOUP = "soup-cli[train]==0.76.1"
ENV = WORK / "py312"
PY = ENV / "bin" / "python"
report: dict[str, object] = {}


def sh(cmd: list[str], name: str, timeout: int = 4 * 3600) -> int:
    log(f"$ {' '.join(map(str, cmd))[:300]}")
    with (OUT / f"{name}.log").open("w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout).returncode
    tail = (OUT / f"{name}.log").read_text(encoding="utf-8", errors="replace")[-1500:]
    log(f"{name}: rc={rc}\n{tail}")
    report[name] = rc
    save_json("report.json", report)
    return rc


# 1. data
data = WORK / "sft"
if not data.exists():
    z = next(INPUT.rglob("data.zip"), None)
    if z is not None:
        with zipfile.ZipFile(z) as zf:
            zf.extractall(data)
    else:
        shutil.copytree(next(INPUT.rglob("train.jsonl")).parent, data)
log(f"data: {sorted(p.name for p in data.iterdir())}")
shutil.copy(data / "manifest.json", OUT / "data_manifest.json")

# 2. Python 3.12 environment with Soup (+ what our own modules import)
sh([sys.executable, "-m", "pip", "install", "-q", "uv"], "pip_uv")
uv = [sys.executable, "-m", "uv"]
sh([*uv, "venv", "--python", "3.12", str(ENV)], "uv_venv")
if sh([*uv, "pip", "install", "--python", str(PY), SOUP, "numpy", "jsonschema", "cryptography"],
      "uv_soup") != 0:  # fmt: skip
    raise SystemExit("Soup install failed")
subprocess.run([str(PY), "-m", "pip", "--version"], check=False)
versions = subprocess.run(
    [str(PY), "-c", "import sys,torch,transformers,peft;print(sys.version.split()[0],torch.__version__,"
     "torch.cuda.is_available(),transformers.__version__,peft.__version__)"],
    capture_output=True, text=True,
)  # fmt: skip
report["env"] = versions.stdout.strip() or versions.stderr[-500:]
log(f"env (python torch cuda transformers peft): {report['env']}")

# 3. train
cfg = (CODE / "training" / "configs" / "script_model.yaml").read_text(encoding="utf-8")
cfg = cfg.replace("TRAIN_JSONL", str(data / "train.jsonl")).replace(
    "OUTPUT_DIR", str(WORK / "adapter")
)
(WORK / "soup.yaml").write_text(cfg, encoding="utf-8")
shutil.copy(WORK / "soup.yaml", OUT / "soup.yaml")
soup = ENV / "bin" / "soup"
if sh([str(soup), "train", "--config", str(WORK / "soup.yaml")], "soup_train") != 0:
    raise SystemExit("soup train failed")

# 4. find the adapter Soup wrote and keep a copy (small: LoRA r=16)
cfgs = sorted((WORK / "adapter").rglob("adapter_config.json"), key=lambda p: len(p.parts))
log(f"adapter configs: {[str(c) for c in cfgs]}")
adapter = cfgs[0].parent
shutil.copytree(adapter, OUT / "adapter", dirs_exist_ok=True)
for f in (WORK / "adapter").glob("*.json"):
    shutil.copy(f, OUT / f"soup_{f.name}")

# 5. evaluate base vs tuned
sh([str(PY), "-m", "training.script_model.evaluate", "--data", str(data), "--adapter", str(adapter),
    "--out", str(OUT)], "evaluate")  # fmt: skip
log("done")

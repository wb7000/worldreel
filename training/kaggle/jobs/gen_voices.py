# Job: Phase 8.2 — voice every line of the sample script, flat (D-8.1 baseline) and with A-10
# emotional delivery; speaker embeddings; voice identity, distinctness and expressiveness.
# Runs after bootstrap.py (log, save_json, OUT, CODE, WORK are defined there).
#   python -m tools.kaggle_runner run training/kaggle/jobs/gen_voices.py --wait
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from formats.wscript import read_wscript
from stages.generate.voices import (
    centroid_cosines,
    nearest_centroid_accuracy,
    pitch_spread_semitones,
    script_lines,
)

report: dict[str, object] = {}
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
UV = [sys.executable, "-m", "uv"]
PARLER = (
    "git+https://github.com/huggingface/parler-tts.git@d108732cd57788ec86bc857d99a6cabd66663d68"
)


def env(name: str, packages: list[str]) -> Path:
    py = Path(f"/tmp/{name}/bin/python")
    subprocess.run([*UV, "venv", "--python", "3.12", f"/tmp/{name}"], check=True)
    r = subprocess.run([*UV, "pip", "install", "--python", str(py), *packages],
                       capture_output=True, text=True)  # fmt: skip
    log(f"{name}: install rc={r.returncode} {r.stderr[-300:] if r.returncode else ''}")
    return py


def run(py: Path, name: str, code: str, args: dict[str, object]) -> None:
    script = WORK / f"{name}.py"
    script.write_text(textwrap.dedent(code), encoding="utf-8")
    (WORK / f"{name}_args.json").write_text(json.dumps(args), encoding="utf-8")
    r = subprocess.run([str(py), str(script), str(WORK / f"{name}_args.json"), str(OUT)],
                       capture_output=True, text=True, timeout=4 * 3600)  # fmt: skip
    (OUT / f"{name}.log").write_text(r.stdout + "\n" + r.stderr[-20000:], encoding="utf-8")
    log(f"{name}: rc={r.returncode} {r.stderr[-800:] if r.returncode else ''}")
    if r.returncode:
        raise SystemExit(f"{name} failed")


lines = script_lines(read_wscript(CODE / "eval" / "scripts" / "lantern_keeper.wscript"))
items = []
for n, ln in enumerate(lines):
    for mode in ("flat", "expressive"):
        items.append({"id": f"{mode}/{n:03d}_{ln.character}", "mode": mode, "text": ln.text,
                      "desc": getattr(ln, mode), "seed": ln.seed})  # fmt: skip
(OUT / "flat").mkdir(exist_ok=True)
(OUT / "expressive").mkdir(exist_ok=True)
save_json("lines.json", [ln.__dict__ for ln in lines])
log(f"{len(lines)} lines, {len(items)} clips")

tts = env("tts", [PARLER, "soundfile"])
run(
    tts,
    "tts",
    """
    import json, sys, time, torch, soundfile as sf
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    a = json.load(open(sys.argv[1])); out = sys.argv[2]; rev = "0392b9451a60"
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-mini-v1", revision=rev).to("cuda")
    tok = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1", revision=rev)
    t0, total = time.time(), 0.0
    for it in a["items"]:
        torch.manual_seed(it["seed"])
        ids = tok(it["desc"], return_tensors="pt").input_ids.to("cuda")
        pids = tok(it["text"], return_tensors="pt").input_ids.to("cuda")
        audio = model.generate(input_ids=ids, prompt_input_ids=pids).cpu().numpy().squeeze()
        sf.write(f"{out}/{it['id']}.wav", audio, model.config.sampling_rate, subtype="PCM_16")
        total += len(audio) / model.config.sampling_rate
    json.dump({"seconds": round(time.time() - t0, 1), "audio_seconds": round(total, 1),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)},
              open(f"{out}/tts.json", "w"))
""",
    {"items": items},
)

spk = env("spk", ["speechbrain", "torchaudio", "soundfile"])
run(
    spk,
    "spk",
    """
    import json, sys, numpy as np, torch, torchaudio, soundfile as sf
    from speechbrain.inference.speaker import EncoderClassifier
    a = json.load(open(sys.argv[1])); out = sys.argv[2]
    enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/tmp/ecapa", run_opts={"device": "cuda:0"}, revision="0f99f2d0ebe8")
    embs = []
    for it in a["items"]:
        x, sr = sf.read(f"{out}/{it['id']}.wav", dtype="float32")
        wav = torchaudio.functional.resample(torch.from_numpy(x).unsqueeze(0), sr, 16000)
        embs.append(enc.encode_batch(wav).squeeze().cpu().numpy())
    np.save(f"{out}/embeddings.npy", np.stack(embs))
""",
    {"items": items},
)

emb = np.load(OUT / "embeddings.npy")
for mode in ("flat", "expressive"):
    idx = [i for i, it in enumerate(items) if it["mode"] == mode]
    labels = [items[i]["id"].split("_")[-1] for i in idx]
    spreads, durations = [], []
    for i in idx:
        sr, x = wavfile.read(OUT / f"{items[i]['id']}.wav")
        spreads.append(pitch_spread_semitones(x, sr))
        durations.append(len(x) / sr)
    report[mode] = {
        "voice_id_accuracy": round(nearest_centroid_accuracy(emb[idx], labels), 4),
        "centroid_cosines": centroid_cosines(emb[idx], labels),
        "pitch_spread_mean_st": round(float(np.mean(spreads)), 3),
        "pitch_spread_median_st": round(float(np.median(spreads)), 3),
        "audio_seconds": round(float(np.sum(durations)), 1),
        "per_line_spread": [round(v, 3) for v in spreads],
    }
    log(f"{mode}: {json.dumps({k: v for k, v in report[mode].items() if k != 'per_line_spread'})}")
report["tts"] = json.loads((OUT / "tts.json").read_text())
save_json("report.json", report)
log("done")

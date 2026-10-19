# Job: Phase 8.2 retry — A-10 v2 (D-8.2): design one synthetic reference voice per character
# with Parler-TTS (description only), pick the most distinct set by ECAPA, then speak all 101
# lines with Chatterbox from those references: constant 0.5/0.5 (baseline) and A-10 delivery.
# Runs after bootstrap.py (log, save_json, OUT, CODE, WORK are defined there).
#   python -m tools.kaggle_runner run training/kaggle/jobs/gen_voices2.py --wait
import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from formats.wscript import read_wscript
from stages.generate.bible import expand
from stages.generate.voices import (
    REFERENCE_CANDIDATES,
    REFERENCE_TEXT,
    best_reference_set,
    centroid_cosines,
    chatterbox_params,
    flat_description,
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
ECAPA_SHA = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"


def env(name: str, python: str, packages: list[str]) -> Path:
    py = Path(f"/tmp/{name}/bin/python")
    if not py.exists():
        subprocess.run([*UV, "venv", "--python", python, f"/tmp/{name}"], check=True)
        r = subprocess.run([*UV, "pip", "install", "--python", str(py), *packages],
                           capture_output=True, text=True)  # fmt: skip
        log(f"{name}: install rc={r.returncode} {r.stderr[-400:] if r.returncode else ''}")
        freeze = subprocess.run([*UV, "pip", "freeze", "--python", str(py)],
                                capture_output=True, text=True)  # fmt: skip
        (OUT / f"{name}_freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    return py


def run(py: Path, name: str, code: str, args: dict[str, object]) -> None:
    script = WORK / f"{name}.py"
    script.write_text(textwrap.dedent(code), encoding="utf-8")
    (WORK / f"{name}_args.json").write_text(json.dumps(args), encoding="utf-8")
    r = subprocess.run([str(py), str(script), str(WORK / f"{name}_args.json"), str(OUT)],
                       capture_output=True, text=True, timeout=5 * 3600)  # fmt: skip
    (OUT / f"{name}.log").write_text(r.stdout[-20000:] + "\n" + r.stderr[-20000:], encoding="utf-8")
    log(f"{name}: rc={r.returncode} {r.stderr[-800:] if r.returncode else ''}")
    if r.returncode:
        raise SystemExit(f"{name} failed")


EMBED = """
    import json, sys, numpy as np, torch, torchaudio, soundfile as sf
    from huggingface_hub import snapshot_download
    from speechbrain.inference.speaker import EncoderClassifier
    a = json.load(open(sys.argv[1])); out = sys.argv[2]
    local = snapshot_download("speechbrain/spkrec-ecapa-voxceleb", revision=a["sha"])
    enc = EncoderClassifier.from_hparams(source=local, savedir="/tmp/ecapa",
                                         run_opts={"device": "cuda:0"})
    embs = []
    for f in a["files"]:
        x, sr = sf.read(f"{out}/{f}", dtype="float32")
        x = x if x.ndim == 1 else x.mean(1)
        wav = torchaudio.functional.resample(torch.from_numpy(x).unsqueeze(0), sr, 16000)
        embs.append(enc.encode_batch(wav).squeeze().cpu().numpy())
    np.save(f"{out}/{a['name']}.npy", np.stack(embs))
"""

script = read_wscript(CODE / "eval" / "scripts" / "lantern_keeper.wscript")
chars = script.header["dictionary"]["characters"]
idents = {c: expand(v["identity"]["seed"], v["identity"]["requested"]) for c, v in chars.items()}
(OUT / "refs").mkdir(exist_ok=True)
(OUT / "constant").mkdir(exist_ok=True)
(OUT / "expressive").mkdir(exist_ok=True)

# 1. design candidate reference voices (Parler, description only, never a named speaker)
refs = [
    {"file": f"refs/{c}_{k}.wav", "cid": c, "desc": flat_description(idents[c]),
     "seed": idents[c].seed * 10 + k}
    for c in sorted(idents) for k in range(REFERENCE_CANDIDATES)
]  # fmt: skip
parler = env("tts", "3.12", [PARLER, "soundfile"])
run(
    parler,
    "design",
    """
    import json, sys, torch, soundfile as sf
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    a = json.load(open(sys.argv[1])); out = sys.argv[2]; rev = "0392b9451a60"
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-mini-v1", revision=rev).to("cuda")
    tok = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1", revision=rev)
    for r in a["refs"]:
        torch.manual_seed(r["seed"])
        enc = tok(r["desc"], return_tensors="pt").to("cuda")
        penc = tok(a["text"], return_tensors="pt").to("cuda")
        audio = model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                               prompt_input_ids=penc.input_ids,
                               prompt_attention_mask=penc.attention_mask).cpu().numpy().squeeze()
        sf.write(f"{out}/{r['file']}", audio, model.config.sampling_rate, subtype="PCM_16")
""",
    {"refs": refs, "text": REFERENCE_TEXT},
)

spk = env("spk", "3.12", ["speechbrain", "torchaudio", "soundfile", "huggingface_hub"])
run(
    spk,
    "embed_refs",
    EMBED,
    {"files": [r["file"] for r in refs], "sha": ECAPA_SHA, "name": "ref_emb"},
)
ref_emb = np.load(OUT / "ref_emb.npy")
cands = {c: ref_emb[[i for i, r in enumerate(refs) if r["cid"] == c]] for c in sorted(idents)}
choice, worst_cos = best_reference_set(cands)
designed = {}
for c, k in choice.items():
    f = f"refs/{c}_{k}.wav"
    designed[c] = {"file": f, "sha256": hashlib.sha256((OUT / f).read_bytes()).hexdigest()}
save_json("designed_voices.json", designed)
report["reference_choice"] = {"choice": choice, "max_cosine_between_characters": worst_cos}
log(f"references: {choice}, max cosine {worst_cos}")

# 2. speak every line with Chatterbox from the designed reference
lines = script_lines(script)
items = []
for n, ln in enumerate(lines):
    ex, cfg = chatterbox_params(ln.tension, ln.emotions)
    for mode, (e, g) in (("constant", (0.5, 0.5)), ("expressive", (ex, cfg))):
        items.append({"file": f"{mode}/{n:03d}_{ln.character}.wav", "text": ln.text,
                      "ref": designed[ln.character]["file"], "ex": e, "cfg": g,
                      "seed": ln.seed + n})  # fmt: skip
save_json("lines.json", [{**ln.__dict__, "chatterbox": chatterbox_params(ln.tension, ln.emotions)}
                         for ln in lines])  # fmt: skip
# setuptools: resemble-perth imports pkg_resources and silently drops its watermarker without it
cb = env("cb", "3.11", ["chatterbox-tts==0.1.7", "soundfile", "setuptools<81"])
run(
    cb,
    "speak",
    """
    import hashlib, json, sys, time, torch, soundfile as sf
    from chatterbox.tts import ChatterboxTTS
    a = json.load(open(sys.argv[1])); out = sys.argv[2]
    allowed = {v["sha256"] for v in a["designed"].values()}
    import perth
    if perth.PerthImplicitWatermarker is None:  # never generate unwatermarked speech
        raise SystemExit("Perth watermarker unavailable")
    model = ChatterboxTTS.from_pretrained(device="cuda")
    t0, total = time.time(), 0.0
    for it in a["items"]:
        ref = f"{out}/{it['ref']}"
        # Safety guard (D-8.2): only references our designer made may be spoken from.
        if hashlib.sha256(open(ref, "rb").read()).hexdigest() not in allowed:
            raise SystemExit(f"refusing undesigned reference {ref}")
        torch.manual_seed(it["seed"])
        wav = model.generate(it["text"], audio_prompt_path=ref,
                             exaggeration=it["ex"], cfg_weight=it["cfg"])
        x = wav.squeeze(0).cpu().numpy()
        sf.write(f"{out}/{it['file']}", x, model.sr, subtype="PCM_16")
        total += len(x) / model.sr
    json.dump({"seconds": round(time.time() - t0, 1), "audio_seconds": round(total, 1),
               "sr": model.sr, "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)},
              open(f"{out}/speak.json", "w"))
""",
    {"items": items, "designed": designed},
)

run(
    spk,
    "embed_lines",
    EMBED,
    {"files": [it["file"] for it in items], "sha": ECAPA_SHA, "name": "line_emb"},
)
emb = np.load(OUT / "line_emb.npy")
for mode in ("constant", "expressive"):
    idx = [i for i, it in enumerate(items) if it["file"].startswith(mode)]
    labels = [items[i]["file"].split("_")[-1][:-4] for i in idx]
    spreads = []
    for i in idx:
        sr, x = wavfile.read(OUT / items[i]["file"])
        spreads.append(pitch_spread_semitones(x, sr))
    report[mode] = {
        "voice_id_accuracy": round(nearest_centroid_accuracy(emb[idx], labels), 4),
        "centroid_cosines": centroid_cosines(emb[idx], labels),
        "pitch_spread_mean_st": round(float(np.mean(spreads)), 3),
        "per_line_spread": [round(v, 3) for v in spreads],
    }
    log(f"{mode}: {json.dumps({k: v for k, v in report[mode].items() if k != 'per_line_spread'})}")
report["speak"] = json.loads((OUT / "speak.json").read_text())
save_json("report.json", report)
log("done")

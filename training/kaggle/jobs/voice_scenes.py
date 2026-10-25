# Job: voice a whole directed script with the accepted pipeline (A-12, D-8.7c; plan D-8.9).
# Per scene: chunks of <= 2 speakers (stages.generate.voicing); each chunk rendered by Dia in one
# pass (temperature 1.3, cfg 3.0, float32, 3 takes), Whisper keeps the cleanest take and gives
# word timestamps, the take is split into turns, every turn is voice-converted (ChatterboxVC) to
# its designed voice and put back at its time; chunks are joined with short pauses.
# Outputs: scenes/<scene>.wav and timing.json (start / end of every line in scene time).
#   python -m tools.kaggle_runner run training/kaggle/jobs/voice_scenes.py \
#       --scene data:voicesdesigned --scene data:lanterndirected
import hashlib
import json
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from eval.conversation_turns import boundaries
from formats.wscript import read_wscript
from stages.generate.voices import REFERENCE_TEXT, _words, word_error_rate
from stages.generate.voicing import scene_chunks

report: dict[str, object] = {}
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
UV = [sys.executable, "-m", "uv"]
SEEDS = [7, 8, 9]
GAP_S = 0.5  # pause between chunks
CB = ["chatterbox-tts==0.1.7", "soundfile", "setuptools<81"]


def env(name: str, python: str, packages: list[str]) -> Path:
    py = Path(f"/tmp/{name}/bin/python")
    if not py.exists():
        subprocess.run([*UV, "venv", "--python", python, f"/tmp/{name}"], check=True)
        r = subprocess.run([*UV, "pip", "install", "--python", str(py), *packages],
                           capture_output=True, text=True)  # fmt: skip
        log(f"{name}: install rc={r.returncode} {r.stderr[-400:] if r.returncode else ''}")
    return py


def run(py: Path, name: str, code: str, args: dict[str, object]) -> None:
    script = WORK / f"step_{name}.py"  # never named like a package (D-8.5b)
    script.write_text(textwrap.dedent(code), encoding="utf-8")
    (WORK / f"{name}_args.json").write_text(json.dumps(args), encoding="utf-8")
    r = subprocess.run([str(py), str(script), str(WORK / f"{name}_args.json"), str(OUT)],
                       capture_output=True, text=True, timeout=8 * 3600)  # fmt: skip
    (OUT / f"{name}.log").write_text(r.stdout[-20000:] + "\n" + r.stderr[-20000:], encoding="utf-8")
    log(f"{name}: rc={r.returncode} {r.stderr[-600:] if r.returncode else ''}")
    if r.returncode:
        raise SystemExit(f"{name} failed")


# Inputs: designed voices + the directed script
designed_json = next(INPUT.rglob("designed_voices.json"))
designed = json.loads(designed_json.read_text())
(OUT / "refs").mkdir(exist_ok=True)
for v in designed.values():
    shutil.copy(designed_json.parent / v["file"], OUT / "refs" / Path(v["file"]).name)
    v["file"] = f"refs/{Path(v['file']).name}"
    assert hashlib.sha256((OUT / v["file"]).read_bytes()).hexdigest() == v["sha256"]
script_dir = WORK / "directed"
if not script_dir.exists():
    z = next((p for p in INPUT.rglob("data.zip") if "lanterndirected" in str(p)), None)
    if z is not None:
        with zipfile.ZipFile(z) as zf:
            zf.extractall(script_dir)
    else:
        shutil.copytree(
            next(p for p in INPUT.rglob("wscript.json") if "lanterndirected" in str(p)).parent,
            script_dir,
        )
script = read_wscript(script_dir)
log(
    f"script: {script.header['title']}, {len(script.scenes)} scenes, delivery for {len(script.delivery)}"
)
for d in ("takes", "turns", "turns_vc", "scenes"):
    (OUT / d).mkdir(exist_ok=True)

# Re-voice only these scenes (D-8.9b: S001 and S006 hit Dia's ~16 s cap); empty = all scenes
ONLY = ["S001", "S006"]
scene_ids = [sid for sid in script.scenes if not ONLY or sid in ONLY]
chunks = [c for sid in scene_ids for c in scene_chunks(script, sid)]
items = []
for c in chunks:
    key = f"{c.scene}_{c.number:02d}"
    parts, sr = [], 44100
    for cid in c.speakers:
        sr, x = wavfile.read(OUT / designed[cid]["file"])
        parts += [x.astype(np.int16), np.zeros(int(0.3 * sr), np.int16)]
    prompt = f"refs/prompt_{key}.wav"
    wavfile.write(OUT / prompt, sr, np.concatenate(parts))
    items.append({"key": key, "prompt": prompt, "text": c.dia_text(REFERENCE_TEXT)})
log(f"{len(chunks)} chunks, {sum(len(c.turns) for c in chunks)} lines")

dia = env("dia", "3.12", ["git+https://github.com/nari-labs/dia.git", "soundfile"])
run(
    dia,
    "dia_render",
    """
    import json, sys, torch, numpy as np, soundfile as sf
    from dia.model import Dia
    a = json.load(open(sys.argv[1])); out = sys.argv[2]
    m = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype="float32")
    for it in a["items"]:
        for s in a["seeds"]:
            torch.manual_seed(s)
            x = m.generate(it["text"], audio_prompt=f"{out}/{it['prompt']}", use_torch_compile=False,
                           cfg_scale=3.0, temperature=1.3, top_p=0.95, cfg_filter_top_k=50)
            sf.write(f"{out}/takes/{it['key']}_s{s}.wav", np.asarray(x, dtype=np.float32), 44100, subtype="PCM_16")
        print(it["key"], flush=True)
""",
    {"items": items, "seeds": SEEDS},
)

asr = env("asr", "3.12", ["openai-whisper", "soundfile"])
takes = sorted(str(p.relative_to(OUT)) for p in (OUT / "takes").glob("*.wav"))
run(
    asr,
    "asr",
    """
    import json, sys, whisper
    a = json.load(open(sys.argv[1])); out = sys.argv[2]
    m = whisper.load_model("small.en", device="cuda")
    res = {}
    for f in a["takes"]:
        r = m.transcribe(f"{out}/{f}", language="en", fp16=True, word_timestamps=True)
        res[f] = {"text": r["text"], "words": [[w["word"], w["start"], w["end"]]
                  for s in r["segments"] for w in s.get("words", [])]}
    json.dump(res, open(f"{out}/asr.json", "w"))
""",
    {"takes": takes},
)
heard = json.loads((OUT / "asr.json").read_text())

vc_items, plan = [], {}
for c in chunks:
    key = f"{c.scene}_{c.number:02d}"
    spoken = " ".join(t.spoken for t in c.turns)
    cands = sorted((word_error_rate(spoken, heard[f]["text"]), f) for s in SEEDS
                   if (f := f"takes/{key}_s{s}.wav") in heard)  # fmt: skip
    wer, best = cands[0]
    sr, x = wavfile.read(OUT / best)
    hyp = [(ww[0], float(s0), float(e0)) for w, s0, e0 in heard[best]["words"] if (ww := _words(w))]
    cuts = boundaries([_words(t.spoken) for t in c.turns], hyp, len(x) / sr)
    turns = []
    for k, (t, (s0, e0)) in enumerate(zip(c.turns, cuts, strict=True)):
        f = f"turns/{key}_{k}.wav"
        wavfile.write(OUT / f, sr, x[int(s0 * sr) : int(e0 * sr)])
        vc_items.append(
            {"src": f, "dst": f"turns_vc/{key}_{k}.wav", "ref": designed[t.speaker]["file"]}
        )
        turns.append({"index": t.index, "speaker": t.speaker, "start": s0, "end": e0})
    plan[key] = {"take": best, "wer": round(wer, 3), "wer_all": [round(w, 3) for w, _ in cands],
                 "seconds": round(len(x) / sr, 2), "turns": turns}  # fmt: skip
report["plan"] = plan
save_json("report.json", report)

cb = env("cb", "3.11", CB)
run(
    cb,
    "vc",
    """
    import json, sys, soundfile as sf, perth
    from chatterbox.vc import ChatterboxVC
    a = json.load(open(sys.argv[1])); out = sys.argv[2]
    assert perth.PerthImplicitWatermarker is not None
    m = ChatterboxVC.from_pretrained("cuda")
    for it in a["items"]:
        w = m.generate(f"{out}/{it['src']}", target_voice_path=f"{out}/{it['ref']}")
        sf.write(f"{out}/{it['dst']}", w.squeeze(0).cpu().numpy(), m.sr, subtype="PCM_16")
    json.dump({"sr": m.sr}, open(f"{out}/vc.json", "w"))
""",
    {"items": vc_items},
)

# Assemble each scene: chunks in order, converted turns at their times, GAP_S between chunks
vc_sr = json.loads((OUT / "vc.json").read_text())["sr"]
timing: dict[str, list[dict[str, object]]] = {}
for sid in scene_ids:
    pieces: list[np.ndarray] = []
    t0 = 0.0
    timing[sid] = []
    for c in [c for c in chunks if c.scene == sid]:
        key = f"{c.scene}_{c.number:02d}"
        p = plan[key]
        mix = np.zeros(int((p["seconds"] + 1.0) * vc_sr), np.float64)
        end = 0
        for k, t in enumerate(p["turns"]):
            sr_t, seg = wavfile.read(OUT / f"turns_vc/{key}_{k}.wav")
            seg = seg.astype(np.float64) / 32768
            if sr_t != vc_sr:
                seg = resample_poly(seg, vc_sr, sr_t)
            s = int(t["start"] * vc_sr)
            n = min(len(seg), len(mix) - s)
            mix[s : s + n] += seg[:n]
            end = max(end, s + n)
            timing[sid].append({"index": t["index"], "speaker": t["speaker"],
                                "start": round(t0 + t["start"], 3),
                                "end": round(t0 + (s + n) / vc_sr, 3)})  # fmt: skip
        mix = mix[:end]
        pieces += [mix, np.zeros(int(GAP_S * vc_sr))]
        t0 += len(mix) / vc_sr + GAP_S
    scene = np.concatenate(pieces) if pieces else np.zeros(1)
    peak = np.abs(scene).max()
    if peak > 0.99:
        scene = scene * 0.99 / peak
    wavfile.write(OUT / "scenes" / f"{sid}.wav", vc_sr, (scene * 32767).astype(np.int16))
    log(f"{sid}: {len(scene) / vc_sr:.1f} s")
save_json("timing.json", timing)
report["scene_seconds"] = {sid: round(t[-1]["end"], 2) if t else 0 for sid, t in timing.items()}
save_json("report.json", report)
log("done")

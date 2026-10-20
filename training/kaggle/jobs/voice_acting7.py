# Job: round 7 (D-8.7) — clean the source. Dia renders each whole exchange with calmer settings
# (temperature 1.3, cfg 3.0, float32) in 5 takes; Whisper picks the cleanest and gives word
# timestamps; the take is split into turns; every turn is voice-converted with ChatterboxVC to
# its character's designed voice and put back at the same start time.
#   python -m tools.kaggle_runner run training/kaggle/jobs/voice_acting7.py --scene data:voicesdesigned
import hashlib
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from eval.conversation_turns import boundaries
from stages.generate.voices import REFERENCE_TEXT, _words, word_error_rate

report: dict[str, object] = {}
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
UV = [sys.executable, "-m", "uv"]
SEEDS = [7, 8, 9, 10, 11]
CB = ["chatterbox-tts==0.1.7", "soundfile", "setuptools<81"]
DIA_NV = {"(laugh)": "(laughs)", "(chuckle)": "(chuckle)", "(sigh)": "(sighs)"}


def env(name: str, python: str, packages: list[str]) -> Path:
    py = Path(f"/tmp/{name}/bin/python")
    if not py.exists():
        subprocess.run([*UV, "venv", "--python", python, f"/tmp/{name}"], check=True)
        r = subprocess.run([*UV, "pip", "install", "--python", str(py), *packages],
                           capture_output=True, text=True)  # fmt: skip
        log(f"{name}: install rc={r.returncode} {r.stderr[-400:] if r.returncode else ''}")
    return py


def run(py: Path, name: str, code: str, args: dict[str, object]) -> None:
    # step names never equal a package name (a script called dia.py shadowed `dia`, D-8.5b)
    script = WORK / f"step_{name}.py"
    script.write_text(textwrap.dedent(code), encoding="utf-8")
    (WORK / f"{name}_args.json").write_text(json.dumps(args), encoding="utf-8")
    r = subprocess.run([str(py), str(script), str(WORK / f"{name}_args.json"), str(OUT)],
                       capture_output=True, text=True, timeout=4 * 3600)  # fmt: skip
    (OUT / f"{name}.log").write_text(r.stdout[-20000:] + "\n" + r.stderr[-20000:], encoding="utf-8")
    log(f"{name}: rc={r.returncode} {r.stderr[-600:] if r.returncode else ''}")
    if r.returncode:
        raise SystemExit(f"{name} failed")


designed_json = next(INPUT.rglob("designed_voices.json"))
designed = json.loads(designed_json.read_text())
(OUT / "refs").mkdir(exist_ok=True)
for v in designed.values():
    shutil.copy(designed_json.parent / v["file"], OUT / "refs" / Path(v["file"]).name)
    v["file"] = f"refs/{Path(v['file']).name}"
    assert hashlib.sha256((OUT / v["file"]).read_bytes()).hexdigest() == v["sha256"]
exchanges = json.loads((CODE / "eval" / "conversations.json").read_text())["exchanges"]
for d in ("takes", "turns", "turns_vc", "final"):
    (OUT / d).mkdir(exist_ok=True)

items = []
for ex in exchanges:
    speakers = list(dict.fromkeys(t["cid"] for t in ex["turns"]))
    parts = []
    for c in speakers:
        sr, x = wavfile.read(OUT / designed[c]["file"])
        parts += [x.astype(np.int16), np.zeros(int(0.3 * sr), np.int16)]
    prompt = f"refs/{ex['id']}_prompt.wav"
    wavfile.write(OUT / prompt, sr, np.concatenate(parts))
    tags = {c: f"[S{k + 1}]" for k, c in enumerate(speakers)}
    spoken = [t["spoken"] for t in ex["turns"]]
    body = " ".join(f"{tags[t['cid']]} {s}" for t, s in zip(ex["turns"], spoken, strict=True))
    for a, b in DIA_NV.items():
        body = body.replace(a, b)
    items.append({"id": ex["id"], "prompt": prompt,
                  "text": " ".join(f"{tags[c]} {REFERENCE_TEXT}" for c in speakers) + " " + body})  # fmt: skip

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
            sf.write(f"{out}/takes/{it['id']}_s{s}.wav", np.asarray(x, dtype=np.float32), 44100, subtype="PCM_16")
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
for ex in exchanges:
    spoken = " ".join(t["spoken"] for t in ex["turns"])
    cands = sorted((word_error_rate(spoken, heard[f]["text"]), f) for s in SEEDS
                   if (f := f"takes/{ex['id']}_s{s}.wav") in heard)  # fmt: skip
    wer, best = cands[0]
    sr, x = wavfile.read(OUT / best)
    shutil.copy(OUT / best, OUT / "final" / f"{ex['id']}_raw.wav")
    hyp = [(ww[0], float(s), float(e)) for w, s, e in heard[best]["words"] if (ww := _words(w))]
    cuts = boundaries([_words(t["spoken"]) for t in ex["turns"]], hyp, len(x) / sr)
    turns = []
    for i, (t, (s0, e0)) in enumerate(zip(ex["turns"], cuts, strict=True)):
        f = f"turns/{ex['id']}_{i}.wav"
        wavfile.write(OUT / f, sr, x[int(s0 * sr) : int(e0 * sr)])
        vc_items.append(
            {"src": f, "dst": f"turns_vc/{ex['id']}_{i}.wav", "ref": designed[t["cid"]]["file"]}
        )
        turns.append({"cid": t["cid"], "start": s0, "end": e0})
    plan[ex["id"]] = {"take": best, "wer": round(wer, 3), "wer_all": [round(c[0], 3) for c in cands],
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

# Reassemble the converted turns at their original start times (timing kept)
vc_sr = json.loads((OUT / "vc.json").read_text())["sr"]
for ex in exchanges:
    p = plan[ex["id"]]
    total = int(p["seconds"] * vc_sr) + vc_sr
    mix = np.zeros(total, np.float64)
    for i, t in enumerate(p["turns"]):
        sr_t, seg = wavfile.read(OUT / f"turns_vc/{ex['id']}_{i}.wav")
        seg = seg.astype(np.float64) / 32768
        if sr_t != vc_sr:
            seg = resample_poly(seg, vc_sr, sr_t)
        s = int(t["start"] * vc_sr)
        n = min(len(seg), total - s)
        mix[s : s + n] += seg[:n]
    mix = mix[: int(p["seconds"] * vc_sr) + int(0.3 * vc_sr)]
    peak = np.abs(mix).max()
    if peak > 0.99:
        mix = mix * 0.99 / peak
    wavfile.write(OUT / "final" / f"{ex['id']}_vc.wav", vc_sr, (mix * 32767).astype(np.int16))
save_json("report.json", report)
log("done")

# Job: Phase 8.1 — check the generation models on the T4 before building on them (D-8.0).
# Runs after bootstrap.py (log, save_json, OUT, CODE, WORK are defined there).
#   python -m tools.kaggle_runner run training/kaggle/jobs/gen_env.py --wait
# Each tool gets its own uv-made Python 3.12 environment under /tmp (Parler-TTS pins older
# transformers; Kaggle's own Python is 3.13). Outputs: report.json, voice clips, portraits.
import json
import subprocess
import sys
import textwrap
from pathlib import Path

report: dict[str, object] = {}
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
UV = [sys.executable, "-m", "uv"]

# Two characters of the sample script, from their identity vectors (stages.generate.bible).
sys.path.insert(0, str(CODE))
from formats.wscript import read_wscript
from stages.generate.bible import expand

dic = read_wscript(CODE / "eval" / "scripts" / "lantern_keeper.wscript").header["dictionary"]
people = {cid: dic["characters"][cid] for cid in ("C01", "C02")}
LINES = [
    "We have four days before the restoration crews start pulling those stones apart.",
    "This survey is from 1962. It shows a chapel here, and a passage under it.",
    "Then nobody has needed it since.",
]


def env(name: str, packages: list[str]) -> Path:
    py = Path(f"/tmp/{name}/bin/python")
    if not py.exists():
        subprocess.run([*UV, "venv", "--python", "3.12", f"/tmp/{name}"], check=True)
    r = subprocess.run([*UV, "pip", "install", "--python", str(py), *packages],
                       capture_output=True, text=True)  # fmt: skip
    report[f"{name}_install"] = {"rc": r.returncode, "err": r.stderr[-1500:]}
    freeze = subprocess.run(
        [*UV, "pip", "freeze", "--python", str(py)], capture_output=True, text=True
    )
    (OUT / f"{name}_freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    log(f"{name}: install rc={r.returncode}")
    return py


def run(py: Path, name: str, code: str, args: dict[str, object]) -> dict[str, object]:
    script = WORK / f"{name}.py"
    script.write_text(textwrap.dedent(code), encoding="utf-8")
    (WORK / f"{name}_args.json").write_text(json.dumps(args), encoding="utf-8")
    r = subprocess.run([str(py), str(script), str(WORK / f"{name}_args.json"), str(OUT)],
                       capture_output=True, text=True, timeout=3 * 3600)  # fmt: skip
    (OUT / f"{name}.log").write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
    try:
        res = json.loads((OUT / f"{name}.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        res = {"error": r.stderr[-2000:]}
    res["rc"] = r.returncode
    report[name] = res
    save_json("report.json", report)
    log(f"{name}: rc={r.returncode} {json.dumps(res)[:600]}")
    return res


def voice_description(v: dict[str, float], presentation: float) -> str:
    """Identity voice vector -> Parler-TTS description (no named speakers: synthetic only)."""
    who = "female" if presentation < 0.35 else "male" if presentation > 0.65 else "androgynous"
    pitch = [
        "very low-pitched",
        "low-pitched",
        "moderate pitch",
        "high-pitched",
        "very high-pitched",
    ]
    speed = ["very slowly", "slowly", "at a moderate speed", "quickly", "very quickly"]
    tone = (
        "breathy"
        if v["breathiness"] > 0.65
        else "rough, gravelly"
        if v["roughness"] > 0.65
        else "clear"
    )
    return (
        f"A {who} speaker with a {pitch[min(int(v['pitch'] * 5), 4)]} voice speaks "
        f"{speed[min(int(v['rate'] * 5), 4)]} in a {tone} voice, "
        "with a very close recording that has almost no background noise."
    )


voices = {}
for cid, c in people.items():
    ident = expand(c["identity"]["seed"], c["identity"]["requested"])
    voices[cid] = voice_description(ident.voice, ident.appearance["presentation"])
report["voice_descriptions"] = voices

# 1. TTS: Parler-TTS Mini v1
tts_py = env("tts", ["git+https://github.com/huggingface/parler-tts.git", "soundfile"])
run(
    tts_py,
    "tts",
    """
    import json, sys, time, torch, soundfile as sf
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    a = json.load(open(sys.argv[1])); out = sys.argv[2]
    rev = "0392b9451a60"
    t0 = time.time()
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1", revision=rev).to("cuda")
    tok = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1", revision=rev)
    res = {"load_s": round(time.time() - t0, 1), "sr": model.config.sampling_rate, "clips": []}
    torch.cuda.reset_peak_memory_stats()
    for cid, desc in a["voices"].items():
        for i, line in enumerate(a["lines"]):
            torch.manual_seed(a["seeds"][cid])
            t = time.time()
            ids = tok(desc, return_tensors="pt").input_ids.to("cuda")
            pids = tok(line, return_tensors="pt").input_ids.to("cuda")
            audio = model.generate(input_ids=ids, prompt_input_ids=pids).cpu().numpy().squeeze()
            sf.write(f"{out}/voice_{cid}_{i}.wav", audio, model.config.sampling_rate)
            res["clips"].append({"cid": cid, "i": i, "s": round(time.time() - t, 1),
                                 "audio_s": round(len(audio) / model.config.sampling_rate, 2)})
    res["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    json.dump(res, open(f"{out}/tts.json", "w"))
""",
    {
        "voices": voices,
        "lines": LINES,
        "seeds": {cid: c["identity"]["seed"] for cid, c in people.items()},
    },
)

# 2. Speaker embeddings: SpeechBrain ECAPA on the clips
spk_py = env("spk", ["speechbrain", "torchaudio", "soundfile"])
run(
    spk_py,
    "spk",
    """
    import json, sys, glob, time, torch, torchaudio
    from speechbrain.inference.speaker import EncoderClassifier
    out = sys.argv[2]
    t0 = time.time()
    enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                         savedir="/tmp/ecapa", run_opts={"device": "cuda"})
    files = sorted(glob.glob(f"{out}/voice_*.wav")); embs = {}
    for f in files:
        wav, sr = torchaudio.load(f)
        wav = torchaudio.functional.resample(wav, sr, 16000)
        embs[f.split("/")[-1]] = torch.nn.functional.normalize(enc.encode_batch(wav).squeeze(), dim=0)
    names = list(embs); cos = [[round(float(embs[a] @ embs[b]), 3) for b in names] for a in names]
    json.dump({"load_s": round(time.time() - t0, 1), "files": names, "cosine": cos}, open(f"{out}/spk.json", "w"))
""",
    {},
)

# 3. Portraits: Kandinsky 2.2 prior + decoder (fp16)
img_py = env("img", ["diffusers", "transformers", "accelerate", "safetensors"])
prompts = {
    cid: f"studio portrait photo of a {c['identity']['requested']}, {c['description']}, "
    "head and shoulders, facing the camera, neutral grey background, soft light"
    for cid, c in people.items()
}
report["portrait_prompts"] = prompts
run(
    img_py,
    "img",
    """
    import json, sys, time, torch
    from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
    a = json.load(open(sys.argv[1])); out = sys.argv[2]
    t0 = time.time()
    prior = KandinskyV22PriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior",
        revision="9fc51ad5732a", torch_dtype=torch.float16).to("cuda")
    dec = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder",
        revision="9ae140d347fe", torch_dtype=torch.float16).to("cuda")
    res = {"load_s": round(time.time() - t0, 1), "images": []}
    torch.cuda.reset_peak_memory_stats()
    for cid, p in a["prompts"].items():
        t = time.time()
        g = torch.Generator("cuda").manual_seed(a["seeds"][cid])
        emb, neg = prior(p, negative_prompt="cartoon, drawing, blurry, text", generator=g).to_tuple()
        img = dec(image_embeds=emb, negative_image_embeds=neg, height=512, width=512,
                  num_inference_steps=50, generator=g).images[0]
        img.save(f"{out}/portrait_{cid}.png")
        res["images"].append({"cid": cid, "s": round(time.time() - t, 1)})
    res["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    json.dump(res, open(f"{out}/img.json", "w"))
""",
    {"prompts": prompts, "seeds": {cid: c["identity"]["seed"] for cid, c in people.items()}},
)
log("done")

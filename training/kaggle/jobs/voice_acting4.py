# Job: voice-acting round 4 — conversation, not reading (D-8.5). Six two-person exchanges
# (eval/conversations.json) rendered three ways:
#   R1 Turbo, written text, line by line (round-3 style);
#   R2 Turbo, spoken text (hesitations, nonverbals) + manner tag, line by line;
#   R3 Dia, the whole exchange in one pass (spoken text), both designed voices as the prompt.
# 3 takes each; Whisper keeps the cleanest. Lines of R1/R2 are joined with short gaps.
#   python -m tools.kaggle_runner run training/kaggle/jobs/voice_acting4.py --scene data:voicesdesigned
import hashlib
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from stages.generate.voices import REFERENCE_TEXT, word_error_rate

report: dict[str, object] = {}
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
UV = [sys.executable, "-m", "uv"]
SEEDS = [7, 8, 9]
CB = ["chatterbox-tts==0.1.7", "soundfile", "setuptools<81"]
MANNER_TAG = {
    "mischievous": "[sarcastic]", "annoyed": "[angry]", "teasing": "[happy]",
    "laughing despite herself": "[happy]", "overjoyed": "[happy]", "disbelieving": "[surprised]",
    "irritated": "[angry]", "sarcastic": "[sarcastic]", "pleading, close to tears": "[crying]",
    "panicking": "[fear]", "pleading": "[crying]",
}  # fmt: skip
EMOTION_TAG = {"excited": "[happy]", "joyful": "[happy]", "angry": "[angry]", "afraid": "[fear]",
               "crying": "[crying]", "comic_sarcastic": "[sarcastic]", "dry_humour": "[sarcastic]"}  # fmt: skip
TURBO_NV = {"(laugh)": "[laugh]", "(chuckle)": "[chuckle]", "(sigh)": "[sigh]"}
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
    script = WORK / f"{name}.py"
    script.write_text(textwrap.dedent(code), encoding="utf-8")
    (WORK / f"{name}_args.json").write_text(json.dumps(args), encoding="utf-8")
    r = subprocess.run([str(py), str(script), str(WORK / f"{name}_args.json"), str(OUT)],
                       capture_output=True, text=True, timeout=4 * 3600)  # fmt: skip
    (OUT / f"{name}.log").write_text(r.stdout[-20000:] + "\n" + r.stderr[-20000:], encoding="utf-8")
    log(f"{name}: rc={r.returncode} {r.stderr[-600:] if r.returncode else ''}")
    if r.returncode:
        raise SystemExit(f"{name} failed")


def swap(text: str, table: dict[str, str]) -> str:
    for a, b in table.items():
        text = text.replace(a, b)
    return text


designed_json = next(INPUT.rglob("designed_voices.json"))
designed = json.loads(designed_json.read_text())
(OUT / "refs").mkdir(exist_ok=True)
for v in designed.values():
    shutil.copy(designed_json.parent / v["file"], OUT / "refs" / Path(v["file"]).name)
    v["file"] = f"refs/{Path(v['file']).name}"
    assert hashlib.sha256((OUT / v["file"]).read_bytes()).hexdigest() == v["sha256"]
exchanges = json.loads((CODE / "eval" / "conversations.json").read_text())["exchanges"]
for d in ("R1", "R2", "R3", "final"):
    (OUT / d).mkdir(exist_ok=True)

turn_items, dia_items = [], []
for ex in exchanges:
    speakers = list(dict.fromkeys(t["cid"] for t in ex["turns"]))
    for i, t in enumerate(ex["turns"]):
        tag = EMOTION_TAG.get(t["emotion"], "")
        turn_items.append({"id": f"R1/{ex['id']}_{i}", "ref": designed[t["cid"]]["file"],
                           "text": f"{tag} {t['written']}".strip(), "plain": t["written"]})  # fmt: skip
        mtag = MANNER_TAG.get(t["manner"], tag)
        turn_items.append({"id": f"R2/{ex['id']}_{i}", "ref": designed[t["cid"]]["file"],
                           "text": f"{mtag} {swap(t['spoken'], TURBO_NV)}".strip(),
                           "plain": t["spoken"]})  # fmt: skip
    # Dia: two speakers in one pass; prompt = both designed voices saying the reference sentence
    prompt_file = f"refs/{ex['id']}_prompt.wav"
    parts = []
    for c in speakers:
        sr, x = wavfile.read(OUT / designed[c]["file"])
        parts += [x.astype(np.int16), np.zeros(int(0.3 * sr), np.int16)]
    wavfile.write(OUT / prompt_file, sr, np.concatenate(parts))
    tags = {c: f"[S{k + 1}]" for k, c in enumerate(speakers)}
    prompt_text = " ".join(f"{tags[c]} {REFERENCE_TEXT}" for c in speakers)
    body = " ".join(f"{tags[t['cid']]} {swap(t['spoken'], DIA_NV)}" for t in ex["turns"])
    dia_items.append({"id": f"R3/{ex['id']}", "prompt": prompt_file, "text": f"{prompt_text} {body}",
                      "plain": " ".join(t["spoken"] for t in ex["turns"])})  # fmt: skip
save_json("items.json", {"turns": turn_items, "dia": dia_items})

cb = env("cb", "3.11", CB)
run(
    cb,
    "turbo",
    """
    import json, sys, torch, soundfile as sf, perth
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    a = json.load(open(sys.argv[1])); out = sys.argv[2]
    assert perth.PerthImplicitWatermarker is not None
    m = ChatterboxTurboTTS.from_pretrained(device="cuda")
    for it in a["items"]:
        for s in a["seeds"]:
            torch.manual_seed(s)
            w = m.generate(it["text"], audio_prompt_path=f"{out}/{it['ref']}")
            sf.write(f"{out}/{it['id']}_s{s}.wav", w.squeeze(0).cpu().numpy(), m.sr, subtype="PCM_16")
""",
    {"items": turn_items, "seeds": SEEDS},
)
dia = env("dia", "3.12", ["git+https://github.com/nari-labs/dia.git", "soundfile"])
run(
    dia,
    "dia_render",  # not "dia": the script file must not shadow the dia package
    """
    import json, sys, torch, numpy as np, soundfile as sf
    from dia.model import Dia
    a = json.load(open(sys.argv[1])); out = sys.argv[2]
    m = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype="float16")
    for it in a["items"]:
        for s in a["seeds"]:
            torch.manual_seed(s)
            x = m.generate(it["text"], audio_prompt=f"{out}/{it['prompt']}", use_torch_compile=False,
                           cfg_scale=4.0, temperature=1.8, top_p=0.90, cfg_filter_top_k=50)
            sf.write(f"{out}/{it['id']}_s{s}.wav", np.asarray(x, dtype=np.float32), 44100, subtype="PCM_16")
""",
    {"items": dia_items, "seeds": SEEDS},
)

asr = env("asr", "3.12", ["openai-whisper", "soundfile"])
takes = sorted(
    str(p.relative_to(OUT)) for d in ("R1", "R2", "R3") for p in (OUT / d).glob("*_s*.wav")
)
run(
    asr,
    "asr",
    """
    import json, sys, whisper
    a = json.load(open(sys.argv[1])); out = sys.argv[2]
    m = whisper.load_model("small.en", device="cuda")
    json.dump({f: m.transcribe(f"{out}/{f}", language="en", fp16=True)["text"] for f in a["takes"]},
              open(f"{out}/asr.json", "w"))
""",
    {"takes": takes},
)
heard = json.loads((OUT / "asr.json").read_text())


def best(item_id: str, plain: str) -> tuple[str, float]:
    cands = [
        (word_error_rate(plain, heard[f]), f)
        for s in SEEDS
        if (f := f"{item_id}_s{s}.wav") in heard
    ]
    wer, f = min(cands)
    return f, round(wer, 3)


chosen: dict[str, object] = {}
for ex in exchanges:
    for rname in ("R1", "R2"):
        pieces, wers = [], []
        sr = 0
        for i in range(len(ex["turns"])):
            item = next(t for t in turn_items if t["id"] == f"{rname}/{ex['id']}_{i}")
            f, wer = best(item["id"], item["plain"])
            sr, x = wavfile.read(OUT / f)
            pieces += [x, np.zeros(int(0.35 * sr), x.dtype)]
            wers.append(wer)
        wavfile.write(OUT / "final" / f"{ex['id']}_{rname}.wav", sr, np.concatenate(pieces))
        chosen[f"{rname}/{ex['id']}"] = {"wer_per_turn": wers}
    item = next(d for d in dia_items if d["id"] == f"R3/{ex['id']}")
    f, wer = best(item["id"], item["plain"])
    shutil.copy(OUT / f, OUT / "final" / f"{ex['id']}_R3.wav")
    sr, x = wavfile.read(OUT / f)
    chosen[f"R3/{ex['id']}"] = {"wer": wer, "file": f, "seconds": round(len(x) / sr, 2)}
report["chosen"] = chosen
save_json("report.json", report)
log("done")

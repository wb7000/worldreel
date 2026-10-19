# Job: voice-acting round 3 (D-8.4). Same 12 lines (eval/emotion_lines.json).
#   D2: Chatterbox Turbo with its emotion tags ([happy] [crying] [angry] [fear] [whispering]
#       [sarcastic] [surprised]) + nonverbals, in the designed voice;
#   B2: Dia with nonverbals never at the start of a line (start-of-line "(sighs)" broke it in
#       D-8.3d), converted to the designed voice with ChatterboxVC (BV2).
# 3 takes per line, Whisper keeps the cleanest usable one. A-11 shaping runs afterwards locally.
#   python -m tools.kaggle_runner run training/kaggle/jobs/voice_acting3.py --scene data:voicesdesigned
import hashlib
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

from scipy.io import wavfile

from stages.generate.voices import REFERENCE_TEXT, take_ok

report: dict[str, object] = {}
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
UV = [sys.executable, "-m", "uv"]
SEEDS = [7, 8, 9]
CB = ["chatterbox-tts==0.1.7", "soundfile", "setuptools<81"]
TURBO = {
    "excited": ["[happy]"], "joyful": ["[happy]"], "comic_sarcastic": ["[sarcastic]"],
    "despair": [], "crying": ["[crying]"], "angry": ["[angry]"], "afraid": ["[fear]"],
    "whisper_tense": ["[whispering]"], "shocked_grief": ["[surprised]"], "tender": [],
    "neutral": [], "dry_humour": ["[sarcastic]"],
}  # fmt: skip
TURBO_NV = {
    "laugh": "[laugh]",
    "chuckle": "[chuckle]",
    "sigh": "[sigh]",
    "gasp": "[gasp]",
    "cry": "[sniff]",
}
DIA_NV = {
    "laugh": "(laughs)",
    "sigh": "(sighs)",
    "chuckle": "(chuckle)",
    "gasp": "(gasps)",
    "cry": "(sniffs)",
}


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


def with_nonverbal(text: str, tags: list[str], table: dict[str, str]) -> str:
    """Nonverbals after the first sentence (or at the end) — never at the very start."""
    nv = " ".join(table[t] for t in tags if t in table)
    if not nv:
        return text
    first, sep, rest = text.partition(". ")
    return f"{first}. {nv} {rest}" if sep else f"{text} {nv}"


designed_json = next(INPUT.rglob("designed_voices.json"))
designed = json.loads(designed_json.read_text())
(OUT / "refs").mkdir(exist_ok=True)
for v in designed.values():
    shutil.copy(designed_json.parent / v["file"], OUT / "refs" / Path(v["file"]).name)
    v["file"] = f"refs/{Path(v['file']).name}"
    assert hashlib.sha256((OUT / v["file"]).read_bytes()).hexdigest() == v["sha256"]
lines = json.loads((CODE / "eval" / "emotion_lines.json").read_text())["lines"]
for d in ("D2", "B2", "BV2"):
    (OUT / d).mkdir(exist_ok=True)
items = [
    {"id": ln["id"], "text": ln["text"], "ref": designed[ln["cid"]]["file"],
     "turbo": " ".join([*TURBO.get(ln["emotion"], []), with_nonverbal(ln["text"], ln["tags"], TURBO_NV)]),
     "dia": f"[S1] {REFERENCE_TEXT} [S1] {with_nonverbal(ln['text'], ln['tags'], DIA_NV)}"}
    for ln in lines
]  # fmt: skip
save_json("items.json", items)
args = {"items": items, "seeds": SEEDS}

cb = env("cb", "3.11", CB)
run(
    cb,
    "D2_turbo",
    """
    import json, sys, torch, soundfile as sf, perth
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    a = json.load(open(sys.argv[1])); out = sys.argv[2]
    assert perth.PerthImplicitWatermarker is not None
    m = ChatterboxTurboTTS.from_pretrained(device="cuda")
    for it in a["items"]:
        for s in a["seeds"]:
            torch.manual_seed(s)
            w = m.generate(it["turbo"], audio_prompt_path=f"{out}/{it['ref']}")
            sf.write(f"{out}/D2/{it['id']}_s{s}.wav", w.squeeze(0).cpu().numpy(), m.sr, subtype="PCM_16")
""",
    args,
)
dia = env("dia", "3.12", ["git+https://github.com/nari-labs/dia.git", "soundfile"])
run(
    dia,
    "B2_dia",
    """
    import json, sys, torch, numpy as np, soundfile as sf
    from dia.model import Dia
    a = json.load(open(sys.argv[1])); out = sys.argv[2]
    m = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype="float16")
    for it in a["items"]:
        for s in a["seeds"]:
            torch.manual_seed(s)
            x = m.generate(it["dia"], audio_prompt=f"{out}/{it['ref']}", use_torch_compile=False,
                           cfg_scale=4.0, temperature=1.8, top_p=0.90, cfg_filter_top_k=50)
            sf.write(f"{out}/B2/{it['id']}_s{s}.wav", np.asarray(x, dtype=np.float32), 44100, subtype="PCM_16")
""",
    args,
)

asr = env("asr", "3.12", ["openai-whisper", "soundfile"])
takes = sorted(str(p.relative_to(OUT)) for d in ("D2", "B2") for p in (OUT / d).glob("*_s*.wav"))
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
asr_text = json.loads((OUT / "asr.json").read_text())
chosen: dict[str, dict[str, object]] = {}
for d in ("D2", "B2"):
    for it in items:
        cands = []
        for seed in SEEDS:
            f = f"{d}/{it['id']}_s{seed}.wav"
            if f in asr_text:
                sr, x = wavfile.read(OUT / f)
                ok, wer = take_ok(it["text"], asr_text[f], len(x) / sr)
                cands.append((not ok, wer, f, round(len(x) / sr, 2)))
        if cands:
            bad, wer, f, sec = min(cands)
            chosen[f"{d}/{it['id']}"] = {"file": f, "usable": not bad, "wer": wer, "seconds": sec,
                                         "takes_usable": sum(not c[0] for c in cands)}  # fmt: skip
report["chosen"] = chosen
save_json("report.json", report)

vc_items = [{"src": chosen[f"B2/{it['id']}"]["file"], "dst": f"BV2/{it['id']}.wav", "ref": it["ref"]}
            for it in items if f"B2/{it['id']}" in chosen]  # fmt: skip
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
""",
    {"items": vc_items},
)
save_json("report.json", report)
log("done")

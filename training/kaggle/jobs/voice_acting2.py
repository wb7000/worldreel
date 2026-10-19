# Job: voice-acting round 2 (D-8.3c). Same 12 lines. A Chatterbox, D Chatterbox Turbo (native
# tags), B Dia and C Maya1 (acting) converted to the designed voice with ChatterboxVC (BV, CV).
# Every system: 3 takes per line, Whisper picks the cleanest (take guard). Blind folder W/X/Y/Z.
#   python -m tools.kaggle_runner run training/kaggle/jobs/voice_acting2.py --scene data:voicesdesigned
import hashlib
import json
import random
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from formats.wscript import read_wscript
from stages.generate.bible import expand
from stages.generate.voices import REFERENCE_TEXT, pitch_spread_semitones, take_ok

report: dict[str, object] = {}
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
UV = [sys.executable, "-m", "uv"]
SEEDS = [7, 8, 9]
CB = ["chatterbox-tts==0.1.7", "soundfile", "setuptools<81"]


def env(name: str, python: str, packages: list[str]) -> Path:
    py = Path(f"/tmp/{name}/bin/python")
    if not py.exists():
        subprocess.run([*UV, "venv", "--python", python, f"/tmp/{name}"], check=True)
        r = subprocess.run([*UV, "pip", "install", "--python", str(py), *packages],
                           capture_output=True, text=True)  # fmt: skip
        log(f"{name}: install rc={r.returncode} {r.stderr[-400:] if r.returncode else ''}")
        fr = subprocess.run(
            [*UV, "pip", "freeze", "--python", str(py)], capture_output=True, text=True
        )
        (OUT / f"{name}_freeze.txt").write_text(fr.stdout, encoding="utf-8")
    return py


def run(py: Path, name: str, code: str, args: dict[str, object], must: bool = False) -> bool:
    script = WORK / f"{name}.py"
    script.write_text(textwrap.dedent(code), encoding="utf-8")
    (WORK / f"{name}_args.json").write_text(json.dumps(args), encoding="utf-8")
    r = subprocess.run([str(py), str(script), str(WORK / f"{name}_args.json"), str(OUT)],
                       capture_output=True, text=True, timeout=4 * 3600)  # fmt: skip
    (OUT / f"{name}.log").write_text(r.stdout[-20000:] + "\n" + r.stderr[-20000:], encoding="utf-8")
    log(f"{name}: rc={r.returncode} {r.stderr[-600:] if r.returncode else ''}")
    report[f"{name}_rc"] = r.returncode
    if r.returncode and must:
        raise SystemExit(f"{name} failed")
    return r.returncode == 0


designed_json = next(INPUT.rglob("designed_voices.json"))
designed = json.loads(designed_json.read_text())
(OUT / "refs").mkdir(exist_ok=True)
for v in designed.values():
    shutil.copy(designed_json.parent / v["file"], OUT / "refs" / Path(v["file"]).name)
    v["file"] = f"refs/{Path(v['file']).name}"
    assert hashlib.sha256((OUT / v["file"]).read_bytes()).hexdigest() == v["sha256"]
lines = json.loads((CODE / "eval" / "emotion_lines.json").read_text())["lines"]
chars = read_wscript(CODE / "eval" / "scripts" / "lantern_keeper.wscript").header["dictionary"][
    "characters"
]
idents = {c: expand(v["identity"]["seed"], v["identity"]["requested"]) for c, v in chars.items()}
for d in ("A", "B", "C", "D", "BV", "CV", "listen"):
    (OUT / d).mkdir(exist_ok=True)

DIA_TAG = {
    "laugh": "(laughs)",
    "sigh": "(sighs)",
    "chuckle": "(chuckle)",
    "gasp": "(gasps)",
    "cry": "(sniffs)",
}
MAYA_TAGS = {
    "excited": ["<excited>", "<laugh>"], "despair": ["<sigh>"], "comic_sarcastic": ["<sarcastic>", "<chuckle>"],
    "crying": ["<cry>"], "angry": ["<angry>"], "afraid": ["<gasp>"], "whisper_tense": ["<whisper>"],
    "tender": [], "joyful": ["<laugh>"], "neutral": [], "dry_humour": ["<sarcastic>"],
    "shocked_grief": ["<gasp>", "<cry>"],
}  # fmt: skip


def maya_description(c: str) -> str:
    a, v = idents[c].appearance, idents[c].voice
    who = (
        "female"
        if a["presentation"] < 0.35
        else "male"
        if a["presentation"] > 0.65
        else "androgynous"
    )
    pitch = ["very low", "low", "normal", "high", "very high"][min(int(v["pitch"] * 5), 4)]
    pace = ["very slow", "slow", "conversational", "quick", "very quick"][
        min(int(v["rate"] * 5), 4)
    ]
    timbre = (
        "breathy" if v["breathiness"] > 0.65 else "gravelly" if v["roughness"] > 0.65 else "clear"
    )
    age = int((18 + 67 * a["age"]) // 10) * 10
    return f"Realistic {who} voice in the {age}s age. {pitch.capitalize()} pitch, {timbre} timbre, {pace} pacing."


items = []
for ln in lines:
    first, _, rest = ln["text"].partition(". ")
    nv = [DIA_TAG[t] for t in ln["tags"] if t in DIA_TAG]
    dia_text = (f"{first}. {' '.join(nv)} {rest}" if rest and "laugh" in ln["tags"]
                else f"{' '.join(nv)} {ln['text']}").strip()  # fmt: skip
    items.append({
        "id": ln["id"], "cid": ln["cid"], "text": ln["text"], "tags": ln["tags"],
        "ref": designed[ln["cid"]]["file"],
        "ex": round(0.3 + 0.65 * ln["intensity"], 3), "cfg": 0.3 if ln["intensity"] >= 0.5 else 0.5,
        "dia": f"[S1] {REFERENCE_TEXT} [S1] {dia_text}",
        "maya_desc": maya_description(ln["cid"]),
        "maya_text": " ".join([*MAYA_TAGS.get(ln["emotion"], []), ln["text"]]),
    })  # fmt: skip
save_json("items.json", items)
args = {"items": items, "seeds": SEEDS}

cb = env("cb", "3.11", CB)
run(
    cb,
    "A_chatterbox",
    """
    import json, sys, torch, soundfile as sf, perth
    from chatterbox.tts import ChatterboxTTS
    a = json.load(open(sys.argv[1])); out = sys.argv[2]
    assert perth.PerthImplicitWatermarker is not None
    m = ChatterboxTTS.from_pretrained(device="cuda")
    for it in a["items"]:
        for s in a["seeds"]:
            torch.manual_seed(s)
            w = m.generate(it["text"], audio_prompt_path=f"{out}/{it['ref']}", exaggeration=it["ex"], cfg_weight=it["cfg"])
            sf.write(f"{out}/A/{it['id']}_s{s}.wav", w.squeeze(0).cpu().numpy(), m.sr, subtype="PCM_16")
""",
    args,
)
run(
    cb,
    "D_turbo",
    """
    import json, re, sys, torch, soundfile as sf, perth
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    a = json.load(open(sys.argv[1])); out = sys.argv[2]
    assert perth.PerthImplicitWatermarker is not None
    m = ChatterboxTurboTTS.from_pretrained(device="cuda")
    # Which [tags] does the Turbo tokenizer know? (the card lists [laugh] [chuckle] [cough] "and more")
    vocab = {}
    for attr in ("tokenizer", "tokenizer_t3"):
        t = getattr(m, attr, None)
        if t is not None and hasattr(t, "get_vocab"):
            vocab = t.get_vocab()
    tags = sorted(k for k in vocab if re.fullmatch(r"\\[[a-z_ ]+\\]", k))
    json.dump({"tags": tags}, open(f"{out}/turbo_tags.json", "w"))
    use = {"laugh": "[laugh]", "chuckle": "[chuckle]", "sigh": "[sigh]", "gasp": "[gasp]", "cry": "[cry]"}
    for it in a["items"]:
        add = [use[t] for t in it["tags"] if t in use and (use[t] in tags or t in ("laugh", "chuckle"))]
        first, _, rest = it["text"].partition(". ")
        text = f"{first}. {' '.join(add)} {rest}".strip() if rest and add else f"{' '.join(add)} {it['text']}".strip()
        for s in a["seeds"]:
            torch.manual_seed(s)
            w = m.generate(text, audio_prompt_path=f"{out}/{it['ref']}")
            sf.write(f"{out}/D/{it['id']}_s{s}.wav", w.squeeze(0).cpu().numpy(), m.sr, subtype="PCM_16")
""",
    args,
)

dia = env("dia", "3.12", ["git+https://github.com/nari-labs/dia.git", "soundfile"])
run(
    dia,
    "B_dia",
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
            sf.write(f"{out}/B/{it['id']}_s{s}.wav", np.asarray(x, dtype=np.float32), 44100, subtype="PCM_16")
""",
    args,
)

maya = env("maya", "3.12", ["transformers", "accelerate", "snac", "soundfile"])
run(
    maya,
    "C_maya",
    """
    import json, sys, torch, soundfile as sf
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from snac import SNAC
    a = json.load(open(sys.argv[1])); out = sys.argv[2]
    m = AutoModelForCausalLM.from_pretrained("maya-research/maya1", revision="b1b9b38ca9de", torch_dtype=torch.float16, device_map="cuda")
    tok = AutoTokenizer.from_pretrained("maya-research/maya1", revision="b1b9b38ca9de")
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to("cuda")
    SOS, EOS, OFF = 128257, 128258, 128266
    d = lambda i: tok.decode([i])
    for it in a["items"]:
        p = d(128259) + tok.bos_token + f'<description="{it["maya_desc"]}"> {it["maya_text"]}' + d(128009) + d(128260) + d(128261) + d(SOS)
        for s in a["seeds"]:
            torch.manual_seed(s)
            inp = tok(p, return_tensors="pt", add_special_tokens=False).to("cuda")
            with torch.inference_mode():
                o = m.generate(**inp, max_new_tokens=2048, min_new_tokens=28, temperature=0.4, top_p=0.9,
                               repetition_penalty=1.1, do_sample=True, eos_token_id=EOS, pad_token_id=tok.pad_token_id)
            ids = o[0, inp["input_ids"].shape[1]:].tolist()
            ids = ids[: ids.index(EOS)] if EOS in ids else ids
            codes = [t for t in ids if OFF <= t <= 156937]; n = len(codes) // 7; codes = codes[: n * 7]
            l1, l2, l3 = [], [], []
            for i in range(n):
                q = [(x - OFF) % 4096 for x in codes[i * 7:(i + 1) * 7]]
                l1.append(q[0]); l2 += [q[1], q[4]]; l3 += [q[2], q[3], q[5], q[6]]
            with torch.inference_mode():
                z = snac.quantizer.from_codes([torch.tensor(l, device="cuda").unsqueeze(0) for l in (l1, l2, l3)])
                audio = snac.decoder(z)[0, 0].float().cpu().numpy()[2048:]
            sf.write(f"{out}/C/{it['id']}_s{s}.wav", audio, 24000, subtype="PCM_16")
""",
    args,
)

# Take guard: Whisper transcribes every take; keep the cleanest usable take per system and line
asr = env("asr", "3.12", ["openai-whisper", "soundfile"])
takes = sorted(str(p.relative_to(OUT)) for s in "ABCD" for p in (OUT / s).glob("*_s*.wav"))
run(
    asr,
    "asr",
    """
    import json, sys, whisper
    a = json.load(open(sys.argv[1])); out = sys.argv[2]
    m = whisper.load_model("small.en", device="cuda")
    res = {f: m.transcribe(f"{out}/{f}", language="en", fp16=True)["text"] for f in a["takes"]}
    json.dump(res, open(f"{out}/asr.json", "w"))
""",
    {"takes": takes},
    must=True,
)
asr_text = json.loads((OUT / "asr.json").read_text())
chosen: dict[str, dict[str, object]] = {}
for s in "ABCD":
    for it in items:
        cands = []
        for seed in SEEDS:
            f = f"{s}/{it['id']}_s{seed}.wav"
            if f not in asr_text:
                continue
            sr, x = wavfile.read(OUT / f)
            ok, wer = take_ok(it["text"], asr_text[f], len(x) / sr)
            cands.append((not ok, wer, f, len(x) / sr))
        if cands:
            bad, wer, f, sec = min(cands)
            chosen[f"{s}/{it['id']}"] = {"file": f, "usable": not bad, "wer": wer, "seconds": round(sec, 2),
                                         "takes_usable": sum(not c[0] for c in cands), "takes": len(cands)}  # fmt: skip
report["chosen"] = chosen

# Voice conversion of the chosen Dia / Maya1 takes to the designed voice
vc_items = [{"src": chosen[f"{s}/{it['id']}"]["file"], "dst": f"{s}V/{it['id']}.wav", "ref": it["ref"]}
            for s in "BC" for it in items if f"{s}/{it['id']}" in chosen]  # fmt: skip
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

# Score: ECAPA cosine to the designed reference + SER (reported), pitch spread before/after VC
finals = {
    f"{s}/{it['id']}": chosen[f"{s}/{it['id']}"]["file"]
    for s in "AD"
    for it in items
    if f"{s}/{it['id']}" in chosen
}
finals |= {
    f"{s}/{it['id']}": chosen[f"{s}/{it['id']}"]["file"]
    for s in "BC"
    for it in items
    if f"{s}/{it['id']}" in chosen
}
finals |= {
    f"{s}V/{it['id']}": f"{s}V/{it['id']}.wav"
    for s in "BC"
    for it in items
    if (OUT / f"{s}V/{it['id']}.wav").exists()
}
spk = env(
    "spk", "3.12", ["speechbrain", "torchaudio", "soundfile", "huggingface_hub", "transformers"]
)
run(
    spk,
    "score",
    """
    import json, sys, torch, torchaudio, soundfile as sf
    from huggingface_hub import snapshot_download
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.inference.interfaces import foreign_class
    a = json.load(open(sys.argv[1])); out = sys.argv[2]
    ecapa = EncoderClassifier.from_hparams(source=snapshot_download("speechbrain/spkrec-ecapa-voxceleb",
        revision="0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"), savedir="/tmp/ecapa", run_opts={"device": "cuda:0"})
    ser = foreign_class(source=snapshot_download("speechbrain/emotion-recognition-wav2vec2-IEMOCAP", revision="117a9c3dff08"),
        pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier", savedir="/tmp/ser", run_opts={"device": "cuda:0"})
    def load(f):
        x, sr = sf.read(f"{out}/{f}", dtype="float32"); x = x if x.ndim == 1 else x.mean(1)
        return torchaudio.functional.resample(torch.from_numpy(x).unsqueeze(0), sr, 16000)
    emb = lambda f: torch.nn.functional.normalize(ecapa.encode_batch(load(f)).squeeze(), dim=0)
    refs = {r: emb(r) for r in a["refs"]}
    res = {}
    for k, f in a["finals"].items():
        e = emb(f); lab = ser.classify_batch(load(f).to("cuda:0"))[3][0]
        res[k] = {"cos": {r: round(float(e @ v), 4) for r, v in refs.items()}, "ser": lab}
    json.dump(res, open(f"{out}/scores.json", "w"))
""",
    {"finals": finals, "refs": sorted({it["ref"] for it in items})},
    must=True,
)
sc = json.loads((OUT / "scores.json").read_text())
summary: dict[str, object] = {}
for s in ("A", "B", "BV", "C", "CV", "D"):
    rows = []
    for it in items:
        k = f"{s}/{it['id']}"
        if k not in sc:
            continue
        cos = sc[k]["cos"]
        own = cos[it["ref"]]
        sr, x = wavfile.read(OUT / finals[k])
        rows.append({"id": it["id"], "cos_own": own, "nearest_ok": own == max(cos.values()),
                     "spread": round(pitch_spread_semitones(x, sr), 3), "ser": sc[k]["ser"],
                     "usable": chosen.get(f"{s.rstrip('V')}/{it['id']}", {}).get("usable")})  # fmt: skip
    if rows:
        summary[s] = {
            "nearest_ref_accuracy": round(sum(r["nearest_ok"] for r in rows) / len(rows), 3),
            "cos_own_mean": round(float(np.mean([r["cos_own"] for r in rows])), 3),
            "pitch_spread_mean": round(float(np.mean([r["spread"] for r in rows])), 3),
            "usable_takes": sum(bool(r["usable"]) for r in rows),
            "per_line": rows,
        }
        log(f"{s}: {json.dumps({k: v for k, v in summary[s].items() if k != 'per_line'})}")
report["summary"] = summary

# Blind folder: per line A, D, BV, CV shuffled as W/X/Y/Z
rng = random.Random(2027)
key = {}
for it in items:
    order = [s for s in ("A", "D", "BV", "CV") if f"{s}/{it['id']}" in finals]
    rng.shuffle(order)
    for letter, s in zip("WXYZ", order, strict=False):
        shutil.copy(OUT / finals[f"{s}/{it['id']}"], OUT / "listen" / f"{it['id']}_{letter}.wav")
        key[f"{it['id']}_{letter}"] = s
report["blind_key"] = key
save_json("report.json", report)
log("done")

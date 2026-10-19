# Job: A-10 v3 voice-acting bake-off (D-8.3). The same 12 emotion lines (eval/emotion_lines.json,
# intended emotion written by hand) rendered by three open TTS models:
#   A Chatterbox 0.1.7 from the designed reference voice, exaggeration from intensity;
#   B Dia-1.6B-0626 from the designed reference (audio + transcript) with nonverbal tags;
#   C Maya1 from a voice description with inline emotion tags.
# Scored: speech-emotion recognition (SpeechBrain IEMOCAP: neu / ang / hap / sad) against the
# intended class; ECAPA cosine to the character's designed reference; plus a blind folder for
# the owner (letters shuffled per line; the key is only in report.json).
#   python -m tools.kaggle_runner run training/kaggle/jobs/voice_acting.py --scene data:voicesdesigned
import hashlib
import json
import random
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

from formats.wscript import read_wscript
from stages.generate.bible import expand
from stages.generate.voices import REFERENCE_TEXT

report: dict[str, object] = {}
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
UV = [sys.executable, "-m", "uv"]
ECAPA_SHA = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
SER_SHA = "117a9c3dff08"


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
                       capture_output=True, text=True, timeout=3 * 3600)  # fmt: skip
    (OUT / f"{name}.log").write_text(r.stdout[-20000:] + "\n" + r.stderr[-20000:], encoding="utf-8")
    log(f"{name}: rc={r.returncode} {r.stderr[-600:] if r.returncode else ''}")
    report[f"{name}_rc"] = r.returncode
    if r.returncode and must:
        raise SystemExit(f"{name} failed")
    return r.returncode == 0


# Designed voices (D-8.2b) and the test lines
designed_json = next(INPUT.rglob("designed_voices.json"))
designed = json.loads(designed_json.read_text())
refdir = OUT / "refs"
refdir.mkdir(exist_ok=True)
for v in designed.values():
    shutil.copy(designed_json.parent / v["file"], refdir / Path(v["file"]).name)
    v["file"] = f"refs/{Path(v['file']).name}"
    assert hashlib.sha256((OUT / v["file"]).read_bytes()).hexdigest() == v["sha256"]
lines = json.loads((CODE / "eval" / "emotion_lines.json").read_text())["lines"]
chars = read_wscript(CODE / "eval" / "scripts" / "lantern_keeper.wscript").header["dictionary"][
    "characters"
]
idents = {c: expand(v["identity"]["seed"], v["identity"]["requested"]) for c, v in chars.items()}
for d in ("A", "B", "C"):
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
    age = 18 + 67 * a["age"]
    pitch = ["very low", "low", "normal", "high", "very high"][min(int(v["pitch"] * 5), 4)]
    pace = ["very slow", "slow", "conversational", "quick", "very quick"][
        min(int(v["rate"] * 5), 4)
    ]
    timbre = (
        "breathy" if v["breathiness"] > 0.65 else "gravelly" if v["roughness"] > 0.65 else "clear"
    )
    return (f"Realistic {who} voice in the {int(age // 10) * 10}s age. {pitch.capitalize()} pitch, "
            f"{timbre} timbre, {pace} pacing.")  # fmt: skip


items = []
for ln in lines:
    first, _, rest = ln["text"].partition(". ")
    nv = [DIA_TAG[t] for t in ln["tags"] if t in DIA_TAG]
    dia_text = (f"{first}. {' '.join(nv)} {rest}" if rest and "laugh" in ln["tags"]
                else f"{' '.join(nv)} {ln['text']}").strip()  # fmt: skip
    items.append({
        "id": ln["id"], "cid": ln["cid"], "text": ln["text"],
        "ref": designed[ln["cid"]]["file"],
        "ex": round(0.3 + 0.65 * ln["intensity"], 3), "cfg": 0.3 if ln["intensity"] >= 0.5 else 0.5,
        "dia": f"[S1] {REFERENCE_TEXT} [S1] {dia_text}",
        "maya_desc": maya_description(ln["cid"]),
        "maya_text": " ".join([*MAYA_TAGS.get(ln["emotion"], []), ln["text"]]),
    })  # fmt: skip
save_json("items.json", items)

# A: Chatterbox
cb = env("cb", "3.11", ["chatterbox-tts==0.1.7", "soundfile", "setuptools<81"])
run(
    cb,
    "A_chatterbox",
    """
    import json, sys, torch, soundfile as sf, perth
    from chatterbox.tts import ChatterboxTTS
    a = json.load(open(sys.argv[1])); out = sys.argv[2]
    assert perth.PerthImplicitWatermarker is not None, "watermarker missing"
    m = ChatterboxTTS.from_pretrained(device="cuda")
    for it in a["items"]:
        torch.manual_seed(7)
        wav = m.generate(it["text"], audio_prompt_path=f"{out}/{it['ref']}",
                         exaggeration=it["ex"], cfg_weight=it["cfg"])
        sf.write(f"{out}/A/{it['id']}.wav", wav.squeeze(0).cpu().numpy(), m.sr, subtype="PCM_16")
""",
    {"items": items},
)

# B: Dia
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
        torch.manual_seed(7)
        audio = m.generate(it["dia"], audio_prompt=f"{out}/{it['ref']}", use_torch_compile=False,
                           cfg_scale=4.0, temperature=1.8, top_p=0.90, cfg_filter_top_k=50)
        sf.write(f"{out}/B/{it['id']}.wav", np.asarray(audio, dtype=np.float32), 44100, subtype="PCM_16")
""",
    {"items": items},
)

# C: Maya1 (fp16: the T4 has no fast bf16)
maya = env("maya", "3.12", ["transformers", "accelerate", "snac", "soundfile"])
run(
    maya,
    "C_maya",
    """
    import json, sys, torch, soundfile as sf
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from snac import SNAC
    a = json.load(open(sys.argv[1])); out = sys.argv[2]
    m = AutoModelForCausalLM.from_pretrained("maya-research/maya1", revision="b1b9b38ca9de",
                                             torch_dtype=torch.float16, device_map="cuda")
    tok = AutoTokenizer.from_pretrained("maya-research/maya1", revision="b1b9b38ca9de")
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to("cuda")
    SOS, EOS, OFF = 128257, 128258, 128266
    def prompt(desc, text):
        d = lambda i: tok.decode([i])
        return d(128259) + tok.bos_token + f'<description="{desc}"> {text}' + d(128009) + d(128260) + d(128261) + d(SOS)
    for it in a["items"]:
        torch.manual_seed(7)
        inp = tok(prompt(it["maya_desc"], it["maya_text"]), return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            o = m.generate(**inp, max_new_tokens=2048, min_new_tokens=28, temperature=0.4, top_p=0.9,
                           repetition_penalty=1.1, do_sample=True, eos_token_id=EOS, pad_token_id=tok.pad_token_id)
        ids = o[0, inp["input_ids"].shape[1]:].tolist()
        ids = ids[: ids.index(EOS)] if EOS in ids else ids
        codes = [t for t in ids if OFF <= t <= 156937]
        n = len(codes) // 7; codes = codes[: n * 7]
        l1, l2, l3 = [], [], []
        for i in range(n):
            s = [(x - OFF) % 4096 for x in codes[i * 7:(i + 1) * 7]]
            l1.append(s[0]); l2 += [s[1], s[4]]; l3 += [s[2], s[3], s[5], s[6]]
        with torch.inference_mode():
            z = snac.quantizer.from_codes([torch.tensor(l, device="cuda").unsqueeze(0) for l in (l1, l2, l3)])
            audio = snac.decoder(z)[0, 0].float().cpu().numpy()[2048:]
        sf.write(f"{out}/C/{it['id']}.wav", audio, 24000, subtype="PCM_16")
""",
    {"items": items},
)

# Scoring: SER + ECAPA
spk = env(
    "spk", "3.12", ["speechbrain", "torchaudio", "soundfile", "huggingface_hub", "transformers"]
)
files = [
    f"{s}/{it['id']}.wav" for s in "ABC" for it in items if (OUT / s / f"{it['id']}.wav").exists()
]
refs = sorted({it["ref"] for it in items})
run(
    spk,
    "score",
    """
    import json, sys, numpy as np, torch, torchaudio, soundfile as sf
    from huggingface_hub import snapshot_download
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.inference.interfaces import foreign_class
    a = json.load(open(sys.argv[1])); out = sys.argv[2]
    ecapa = EncoderClassifier.from_hparams(source=snapshot_download("speechbrain/spkrec-ecapa-voxceleb",
        revision=a["ecapa"]), savedir="/tmp/ecapa", run_opts={"device": "cuda:0"})
    ser_dir = snapshot_download("speechbrain/emotion-recognition-wav2vec2-IEMOCAP", revision=a["ser"])
    ser = foreign_class(source=ser_dir, pymodule_file="custom_interface.py",
                        classname="CustomEncoderWav2vec2Classifier", savedir="/tmp/ser",
                        run_opts={"device": "cuda:0"})
    def load(f):
        x, sr = sf.read(f"{out}/{f}", dtype="float32"); x = x if x.ndim == 1 else x.mean(1)
        return torchaudio.functional.resample(torch.from_numpy(x).unsqueeze(0), sr, 16000)
    res = {"ref_emb": {r: ecapa.encode_batch(load(r)).squeeze().cpu().numpy().tolist() for r in a["refs"]}, "clips": {}}
    for f in a["files"]:
        w = load(f)
        e = ecapa.encode_batch(w).squeeze().cpu().numpy().tolist()
        prob, score, idx, lab = ser.classify_batch(w.to("cuda:0"))
        res["clips"][f] = {"emb": e, "ser": lab[0], "ser_prob": prob.squeeze().cpu().numpy().tolist(),
                           "seconds": round(w.shape[1] / 16000, 2)}
    json.dump(res, open(f"{out}/scores.json", "w"))
""",
    {"files": files, "refs": refs, "ecapa": ECAPA_SHA, "ser": SER_SHA},
    must=True,
)

sc = json.loads((OUT / "scores.json").read_text())
ref_emb = {r: np.array(v) / np.linalg.norm(v) for r, v in sc["ref_emb"].items()}
summary: dict[str, dict[str, object]] = {}
SER_MAP = {"neu": "neutral", "ang": "angry", "hap": "happy", "sad": "sad"}
for s in "ABC":
    hits = tot = idok = 0
    per = {}
    for ln, it in zip(lines, items, strict=True):
        f = f"{s}/{it['id']}.wav"
        if f not in sc["clips"]:
            continue
        c = sc["clips"][f]
        e = np.array(c["emb"]) / np.linalg.norm(c["emb"])
        cos = {r: float(e @ v) for r, v in ref_emb.items()}
        own = cos[it["ref"]]
        idok += own == max(cos.values())
        heard = SER_MAP.get(c["ser"], c["ser"])
        if ln["ser"]:
            tot += 1
            hits += heard == ln["ser"]
        per[it["id"]] = {"heard": heard, "intended": ln["ser"], "cos_own_ref": round(own, 3),
                         "seconds": c["seconds"]}  # fmt: skip
    n = len(per)
    summary[s] = {"clips": n, "ser_agreement": round(hits / tot, 3) if tot else None,
                  "ser_lines": tot, "nearest_ref_accuracy": round(idok / n, 3) if n else None,
                  "per_line": per}  # fmt: skip
    log(f"{s}: {json.dumps({k: v for k, v in summary[s].items() if k != 'per_line'})}")
report["summary"] = summary

# Blind listening folder: per line, the three systems as X/Y/Z in random order
rng = random.Random(2026)
(OUT / "listen").mkdir(exist_ok=True)
key = {}
for it in items:
    order = [s for s in "ABC" if (OUT / s / f"{it['id']}.wav").exists()]
    rng.shuffle(order)
    for letter, s in zip("XYZ", order, strict=False):
        shutil.copy(OUT / s / f"{it['id']}.wav", OUT / "listen" / f"{it['id']}_{letter}.wav")
        key[f"{it['id']}_{letter}"] = s
report["blind_key"] = key
save_json("report.json", report)
log("done")

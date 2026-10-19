"""Synthetic voices from the character bible, with per-line emotional delivery (A-10).

A line's TTS description = the character's fixed identity part (gender, pitch, speed, timbre
from the voice vector) + a delivery part from the script (tension, manner note, line cues).
Only descriptions are used — never a named or recorded speaker (no cloning, brief §5 / §7).
Pure Python: the TTS itself runs on Kaggle (`training/kaggle/jobs/gen_voices.py`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from formats.wscript import Dialogue, SceneHeader, Script, Tension, expand_text
from stages.generate.bible import Identity, expand

PITCH = ("very low-pitched", "low-pitched", "moderate-pitched", "high-pitched", "very high-pitched")
SPEED = ("very slowly", "slowly", "at a moderate speed", "quickly", "very quickly")
RECORDING = "with a very close recording that has almost no background noise."

# manner note words -> delivery words (A-10)
MANNER = (
    (("whisper", "quiet", "under his breath", "under her breath", "barely audible", "softly"),
     "speaks softly, almost whispering"),
    (("dry",), "dry and deadpan"),
    (("gentl", "careful", "kind"), "gentle and warm"),
    (("smil", "laugh", "grin"), "cheerful and warm"),
    (("angr", "shout", "furious"), "angry and loud"),
    (("sad", "tear", "cry", "sob"), "sad and shaken"),
    (("yawn", "tired"), "tired and sleepy"),
    (("hast", "urgent"), "urgent and breathless"),
)  # fmt: skip
CUES = (
    (re.compile(r"\b(no, no|please|come on|hurry|fast|run|go)\b", re.I), "afraid"),
    (re.compile(r"\b(dead|died|drowned|lost|fifty years|gone)\b", re.I), "sad"),
    (re.compile(r"\b(thank you|family|darling|love)\b", re.I), "tender"),
)


def identity_part(ident: Identity) -> str:
    v, presentation = ident.voice, ident.appearance["presentation"]
    who = "female" if presentation < 0.35 else "male" if presentation > 0.65 else "androgynous"
    if v["breathiness"] > 0.65:
        timbre = "breathy"
    elif v["roughness"] > 0.65:
        timbre = "rough, gravelly"
    else:
        timbre = "clear"
    return (
        f"A {who} speaker with a {PITCH[min(int(v['pitch'] * 5), 4)]}, {timbre} voice "
        f"speaks {SPEED[min(int(v['rate'] * 5), 4)]}"
    )


def flat_description(ident: Identity) -> str:
    """The D-8.1 baseline: the same description for every line of a character."""
    return f"{identity_part(ident)} {RECORDING}"


def delivery(text: str, manner: str | None, tension: int) -> tuple[str, list[str]]:
    """Delivery words for one line, and the emotion tags for its sound event."""
    if tension <= 2:
        words = ["calm, slightly expressive"]
    elif tension <= 5:
        words = ["expressive and animated"]
    elif tension <= 8:
        words = ["very expressive and tense"]
    else:
        words = ["urgent, very expressive and animated"]
    emotions: list[str] = []
    m = (manner or "").lower()
    for keys, desc in MANNER:
        if any(k in m for k in keys):
            words.append(desc)
            emotions.append(desc.split(" and ")[0].split(",")[0].replace("speaks ", ""))
    for rx, emo in CUES:
        if rx.search(text):
            emotions.append(emo)
    if "!" in text:
        emotions.append("excited")
    if text.rstrip().endswith("?"):
        emotions.append("questioning")
    tags = [e for e in emotions if e not in words]
    extra = [e for e in tags if e in ("afraid", "sad", "tender", "excited", "questioning")]
    if extra:
        words.append(" and ".join(dict.fromkeys(extra)))
    return ", ".join(words), list(dict.fromkeys(emotions)) or ["neutral"]


@dataclass(frozen=True)
class Line:
    scene: str
    index: int  # statement index within the scene
    character: str
    text: str  # IDs expanded to names, ready to speak
    tension: int
    emotions: tuple[str, ...]
    flat: str  # baseline description
    expressive: str  # A-10 description
    seed: int


def script_lines(script: Script) -> list[Line]:
    d = script.header["dictionary"]
    idents = {
        cid: expand(int(c["identity"]["seed"]), str(c["identity"].get("requested", "")))
        for cid, c in d["characters"].items()
        if "identity" in c
    }
    out: list[Line] = []
    for entry in script.header["scenes"]:
        sid = entry["id"]
        tension = int(entry.get("tension", 5))
        for i, s in enumerate(script.scenes[sid]):
            if isinstance(s, SceneHeader):
                tension = int(dict(s.params).get("tension", tension))
            elif isinstance(s, Tension):
                tension = s.value
            elif isinstance(s, Dialogue) and s.speaker in idents:
                ident = idents[s.speaker]
                text = expand_text(s.text, d)
                how, emotions = delivery(text, s.manner, tension)
                out.append(
                    Line(
                        sid,
                        i,
                        s.speaker,
                        text,
                        tension,
                        tuple(emotions),
                        flat_description(ident),
                        f"{identity_part(ident)}, {how}, {RECORDING}",
                        ident.seed,
                    )
                )
    return out


# --------------------------------------------------------------------------- metrics


def pitch_track(x: np.ndarray, sr: int) -> np.ndarray:
    """f0 in Hz per 10 ms voiced frame (autocorrelation, 60-400 Hz). Simple and dependency-free;
    only used to compare delivery styles of the same voices, not as an absolute pitch."""
    x = x.astype(np.float64)
    x = x / (np.abs(x).max() + 1e-9)
    fr, hop = int(0.04 * sr), int(0.01 * sr)
    lo, hi = int(sr / 400), int(sr / 60)
    out = []
    win = np.hanning(fr)
    for s in range(0, len(x) - fr, hop):
        w = x[s : s + fr] * win
        if np.sqrt((w**2).mean()) < 0.05:
            continue
        ac = np.correlate(w, w, "full")[fr - 1 :]
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.3 * ac[0]:
            out.append(sr / k)
    return np.array(out)


def pitch_spread_semitones(x: np.ndarray, sr: int) -> float:
    """Standard deviation of f0 in semitones over voiced frames (expressiveness proxy, A-10)."""
    f0 = pitch_track(x, sr)
    if len(f0) < 5:
        return 0.0
    return float(np.std(12 * np.log2(f0 / np.median(f0))))


def nearest_centroid_accuracy(emb: np.ndarray, labels: list[str]) -> float:
    """Leave-one-out: is each (L2-normalised) embedding closest to its own character's centroid
    (computed without it)? D-8.0 voice identity metric."""
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    labs = np.array(labels)
    ok = 0
    for i in range(len(emb)):
        best, best_lab = -2.0, ""
        for lab in sorted(set(labels)):
            mask = (labs == lab) & (np.arange(len(emb)) != i)
            if not mask.any():
                continue
            c = emb[mask].mean(0)
            cos = float(emb[i] @ c / np.linalg.norm(c))
            if cos > best:
                best, best_lab = cos, lab
        ok += best_lab == labels[i]
    return ok / len(emb)


def centroid_cosines(emb: np.ndarray, labels: list[str]) -> dict[str, float]:
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    labs = np.array(labels)
    cents = {lab: emb[labs == lab].mean(0) for lab in sorted(set(labels))}
    cents = {k: v / np.linalg.norm(v) for k, v in cents.items()}
    keys = list(cents)
    return {
        f"{a}-{b}": round(float(cents[a] @ cents[b]), 4)
        for i, a in enumerate(keys)
        for b in keys[i + 1 :]
    }

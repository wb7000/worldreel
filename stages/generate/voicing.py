"""Plan how a directed scene is voiced with the accepted pipeline (A-12, D-8.7c).

Dia renders at most two speakers per pass ([S1] / [S2]); a scene is cut into chunks of
consecutive dialogue lines with at most two distinct speakers. Each chunk is rendered as one
conversation from the delivery layer's spoken text, then split into turns and voice-converted
(Kaggle job `training/kaggle/jobs/voice_scenes.py`). Pure Python: planning only.
"""

from __future__ import annotations

from dataclasses import dataclass

from formats.wscript import Dialogue, Script, expand_text

DIA_NV = {"(laugh)": "(laughs)", "(chuckle)": "(chuckle)", "(sigh)": "(sighs)", "(gasp)": "(gasps)",
          "(cry)": "(sniffs)"}  # fmt: skip


@dataclass(frozen=True)
class Turn:
    index: int  # statement index in the scene file
    speaker: str
    spoken: str  # what is said (delivery layer, else the written text)


@dataclass(frozen=True)
class Chunk:
    scene: str
    number: int
    speakers: tuple[str, ...]  # order = [S1], [S2]
    turns: tuple[Turn, ...]

    def dia_text(self, reference_text: str) -> str:
        """Dia prompt: each speaker's reference sentence (the audio prompt), then the turns."""
        tag = {c: f"[S{k + 1}]" for k, c in enumerate(self.speakers)}
        prompt = " ".join(f"{tag[c]} {reference_text}" for c in self.speakers)
        body = " ".join(f"{tag[t.speaker]} {t.spoken}" for t in self.turns)
        for a, b in DIA_NV.items():
            body = body.replace(a, b)
        return f"{prompt} {body}"


def scene_chunks(script: Script, scene_id: str, max_speakers: int = 2) -> list[Chunk]:
    d = script.header["dictionary"]
    layer = script.delivery.get(scene_id, {}).get("lines", {})
    turns = []
    for i, s in enumerate(script.scenes[scene_id]):
        if isinstance(s, Dialogue):
            spoken = layer.get(str(i), {}).get("spoken") or expand_text(s.text, d)
            turns.append(Turn(i, s.speaker, str(spoken)))
    chunks: list[Chunk] = []
    current: list[Turn] = []
    for t in turns:
        speakers = list(dict.fromkeys([*(c.speaker for c in current), t.speaker]))
        if current and len(speakers) > max_speakers:
            chunks.append(_chunk(scene_id, len(chunks), current))
            current = []
        current.append(t)
    if current:
        chunks.append(_chunk(scene_id, len(chunks), current))
    return chunks


def _chunk(scene_id: str, number: int, turns: list[Turn]) -> Chunk:
    return Chunk(scene_id, number, tuple(dict.fromkeys(t.speaker for t in turns)), tuple(turns))

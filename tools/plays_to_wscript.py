"""Convert public-domain stage plays (Project Gutenberg plain text) to `.wscript` (D-7.5).

    python -m tools.plays_to_wscript data/plays/raw --out data/plays/wscript

Paragraph-based and layout-agnostic: a paragraph is a speech when it starts with a known speaker
name (`NAME.` / `NAME [direction] ...` on one line, or `NAME.` alone with the speech in the next
paragraph), otherwise a stage direction. Speakers are names seen in speaker position at least
three times. Acts (or scenes) become `.wsl` scenes; the first stage direction of an act becomes
its location. A short leading `[direction]` becomes the dialogue manner; other bracketed
directions become action lines. Speaker names inside directions become `{C…}` references.
Continuity: everyone who speaks in an act is at that act's location; changes between acts are
declared `offscreen` (plays do not write them as changes).

Prefaces, the Project Gutenberg header / licence and anything after the play are dropped; the
converted files stay in git-ignored `data/` and are never published.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any

from formats.wscript import Action, Blank, Dialogue, SceneHeader, Script, Statement, write_wscript
from stages.generate.continuity import check

ACT = re.compile(
    r"^(ACT\s+([IVX]+|ONE|TWO|THREE|FOUR|FIVE|FIRST|SECOND|THIRD|FOURTH|FIFTH|LAST)\b"
    r"|(FIRST|SECOND|THIRD|FOURTH|FIFTH|LAST)\s+ACT\b|SCENE\s+([IVX]+|\d+)\b)",
)
STOP = re.compile(r"^(THE END|END OF THE PLAY|CURTAIN\.?$|SEQUEL\b|\*\*\* END OF)", re.I)
NAME = r"[A-Z][A-Z'\u2019\-]+(?:\.?\s+[A-Z][A-Z'\u2019\-]+){0,3}"  # u2019 = curly apostrophe
SPEAKER_INLINE = re.compile(rf"^({NAME})(?:\.\s*|\s+(?=[\[(_]))(.*)$", re.S)
SPEAKER_ALONE = re.compile(rf"^({NAME})\.?$")
BRACKET = re.compile(r"_?\[([^\]]*)\]_?\.?|\(([^)]*)\)")
NOT_NAMES = {"ACT", "SCENE", "CURTAIN", "THE END", "END", "I", "A", "O", "OH"}


def _is_act(p: str) -> bool:
    return "\n" not in p.strip() and bool(ACT.match(p.strip()))


def _clean(text: str) -> str:
    text = text.replace("\r", "").replace("_", "").replace("{", "{{").replace("}", "}}")
    return re.sub(r"\s+", " ", text).strip()


def _body(raw: str) -> str:
    start = raw.find("*** START OF")
    end = raw.find("*** END OF")
    raw = raw[raw.find("\n", start) + 1 if start >= 0 else 0 : end if end >= 0 else None]
    return raw.replace("\r\n", "\n")


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _speakers(paras: list[str]) -> set[str]:
    counts: Counter[str] = Counter()
    for p in paras:
        first = p.split("\n", 1)[0].strip()
        for rx in (SPEAKER_ALONE, SPEAKER_INLINE):
            m = rx.match(first)
            if m and m.group(1).rstrip(".") not in NOT_NAMES:
                counts[m.group(1).rstrip(".")] += 1
                break
    return {n for n, c in counts.items() if c >= 3}


def _split_speech(text: str) -> list[tuple[str, str]]:
    """('say', text) / ('do', direction) segments of one speech, in order."""
    out: list[tuple[str, str]] = []
    pos = 0
    for m in BRACKET.finditer(text):
        before = text[pos : m.start()].strip(" .,;")
        if before:
            out.append(("say", text[pos : m.start()].strip()))
        inner = (m.group(1) if m.group(1) is not None else m.group(2)).strip()
        if inner:
            out.append(("do", inner))
        pos = m.end()
    if text[pos:].strip(" .,;"):
        out.append(("say", text[pos:].strip()))
    return out


def convert(raw: str, title: str) -> Script | None:
    body = _body(raw)
    # An act heading is a one-line paragraph (a table of contents lists several in one).
    paras = [p.strip() if _is_act(p) else _clean(p) for p in _paragraphs(body)]
    first_act = next((i for i, p in enumerate(paras) if _is_act(p)), None)
    if first_act is None:
        return None
    paras = paras[first_act:]
    stop = next((i for i, p in enumerate(paras) if STOP.match(p)), len(paras))
    paras = paras[:stop]
    names = _speakers(paras)
    if len(names) < 2:
        return None
    ids = {n: f"C{i + 1:02d}" for i, n in enumerate(sorted(names))}
    name_rx = re.compile(
        r"\b(" + "|".join(sorted(map(re.escape, names), key=len, reverse=True)) + r")\b"
    )

    def refs(text: str) -> str:
        return name_rx.sub(lambda m: "{" + ids[m.group(1)] + "}", text)

    acts: list[tuple[str, list[str]]] = []
    for p in paras:
        if _is_act(p):
            acts.append((p.split("\n")[0].strip().rstrip("."), []))
        elif acts:
            acts[-1][1].append(p)

    scenes: dict[str, list[Statement]] = {}
    index: list[dict[str, Any]] = []
    locations: dict[str, Any] = {}
    prev_loc: dict[str, str] = {}
    for n, (heading, body_paras) in enumerate(acts, start=1):
        sid, lid = f"S{n:03d}", f"L{n:02d}"
        stmts: list[Statement] = [SceneHeader(sid, (("loc", lid),))]
        setting = ""
        speakers_here: list[str] = []
        pending: str | None = None  # speaker whose speech is in the next paragraph
        for p in body_paras:
            m_alone = SPEAKER_ALONE.match(p)
            m_inline = SPEAKER_INLINE.match(p)
            if m_alone and m_alone.group(1).rstrip(".") in names:
                pending = m_alone.group(1).rstrip(".")
                continue
            speaker, speech = None, ""
            if pending:
                speaker, speech, pending = pending, p, None
            elif m_inline and m_inline.group(1).rstrip(".") in names:
                speaker, speech = m_inline.group(1).rstrip("."), m_inline.group(2)
            if speaker is None:
                direction = p.strip("[]()_ ")
                if direction:
                    setting = setting or direction
                    stmts.append(Action(refs(direction)))
                continue
            cid = ids[speaker]
            speakers_here.append(cid)
            segments = _split_speech(speech)
            manner = None
            if segments and segments[0][0] == "do" and len(segments[0][1].split()) <= 8:
                manner = refs(segments[0][1]).replace("(", "").replace(")", "")
                segments = segments[1:]
            for kind, text in segments:
                if kind == "do":
                    stmts.append(Action(refs(text)))
                else:
                    text = text.lstrip(" .,;:—-")  # punctuation left after a [direction]
                    if text:
                        stmts.append(Dialogue(cid, text, manner))
                        manner = None
        if not any(isinstance(s, Dialogue) for s in stmts):
            continue
        loc_name = " ".join(setting.split()[:10]) or heading
        locations[lid] = {"name": loc_name, "description": setting[:240]}
        here = sorted(set(speakers_here))
        cont: dict[str, Any] = {"characters": {}}
        for cid in sorted(set(prev_loc) | set(here)):
            cont["characters"][cid] = {"location": lid if cid in here else prev_loc[cid]}
        if index:
            moved = [f"{c}.location" for c in here if prev_loc.get(c) != lid]
            if moved:
                cont["offscreen"] = moved
        for cid in here:
            prev_loc[cid] = lid
        who = ", ".join(n for n, c in ids.items() if c in here)
        index.append(
            {"id": sid, "title": heading.title(),
             "summary": f"{heading.title()}: {' '.join(setting.split()[:14])}… With {who}.",
             "location": lid, "tension": 5, "continuity": cont}
        )  # fmt: skip
        scenes[sid] = [s for s in stmts if not isinstance(s, Blank)]
    if not scenes:
        return None
    header = {
        "format": "wscript",
        "format_version": "1.1",
        "title": title,
        "language": "en",
        "dictionary": {
            "characters": {c: {"name": n.title()} for n, c in ids.items()},
            "locations": locations,
            "props": {},
            "text": {},
        },
        "scenes": index,
    }
    return Script(header=header, scenes=scenes)


def title_of(raw: str) -> str:
    m = re.search(r"^Title:\s*(.+)$", raw, re.M)
    return m.group(1).strip() if m else "Untitled"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m tools.plays_to_wscript")
    p.add_argument("raw", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    for f in sorted(a.raw.glob("pg*.txt")):
        raw = f.read_text(encoding="utf-8", errors="replace")
        script = convert(raw, title_of(raw))
        if script is None:
            print(f"{f.name}: no play found")
            continue
        report, _ = check(script)
        n_dia = sum(isinstance(s, Dialogue) for v in script.scenes.values() for s in v)
        n_act = sum(isinstance(s, Action) for v in script.scenes.values() for s in v)
        dest = a.out / f"{f.stem}.wscript"
        if dest.exists():
            import shutil

            shutil.rmtree(dest)
        try:
            write_wscript(dest, script, as_zip=False)
            status = "ok" if report.ok else f"continuity {len(report.errors)} errors"
        except Exception as exc:  # report and keep going with the other plays
            status = f"INVALID: {str(exc)[:160]}"
        print(
            f"{f.name}: {script.header['title'][:30]!r} scenes={len(script.scenes)} "
            f"speakers={len(script.header['dictionary']['characters'])} dialogue={n_dia} "
            f"actions={n_act} {status}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

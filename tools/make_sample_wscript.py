# ruff: noqa: E501  (scene text: the .wsl format is one statement per line, never wrapped)
"""Build the sample long script `eval/scripts/lantern_keeper.wscript` (D-7.2).

    python -m tools.make_sample_wscript [--out PATH] [--zip] [--force]

"The Lantern Keeper" is an original ten-scene script written for this project (no third-party
text). Only the first scene's continuity is written by hand; every later start state is filled
in from the previous scene's end state (`continuity.propagate`), except the paths a scene
declares `offscreen` (its time jumps).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from formats.wscript import Script, parse_scene, write_wscript
from stages.generate.continuity import propagate

OUT = Path("eval/scripts/lantern_keeper.wscript")

DICTIONARY: dict[str, Any] = {
    "characters": {
        "C01": {"name": "Mira", "description": "archivist in her thirties, careful, stubborn",
                "voice": "warm, quick"},
        "C02": {"name": "Ade", "description": "fort caretaker, seventy-four, wiry, slow to smile",
                "voice": "low, dry"},
        "C03": {"name": "Tomas", "description": "drone pilot in his twenties, talks when nervous",
                "voice": "light, fast"},
        "C04": {"name": "Lena", "description": "Ade's granddaughter, nineteen, direct",
                "voice": "clear, steady"},
    },
    "locations": {
        "L01": {"name": "the fort gate", "description": "stone arch, iron chain, monsoon path"},
        "L02": {"name": "the rampart", "description": "sea wall walk, wet battlements"},
        "L03": {"name": "the chapel ruin", "description": "roofless nave, fig roots, low arch"},
        "L04": {"name": "Ade's hut", "description": "one whitewashed room, oil lamp"},
        "L05": {"name": "the passage", "description": "steps into the hill, iron gate, channel"},
    },
    "props": {
        "P01": {"name": "lantern", "description": "brass storm lantern, cracked glass"},
        "P02": {"name": "iron key", "description": "long, rusted, on a leather loop"},
        "P03": {"name": "drone", "description": "small scanning drone with a red light"},
        "P04": {"name": "green raincoat", "description": "Mira's, too thin for the monsoon"},
        "P05": {"name": "survey map", "description": "1962 fort survey, folded and soft"},
        "P06": {"name": "photograph", "description": "two young men with lanterns, 1962"},
        "P07": {"name": "tablet", "description": "Lena's tablet, cracked corner"},
    },
    "text": {},
}  # fmt: skip

START: dict[str, Any] = {
    "characters": {
        "C01": {"location": "L01", "wearing": ["P04"], "holding": ["P05"], "state": "soaked"},
        "C02": {"location": "L01", "holding": ["P01", "P02"]},
        "C03": {"location": "L02", "holding": ["P03"]},
        "C04": {"location": "L04", "holding": ["P07"]},
    },
    "lighting": "dusk_overcast",
    "weather": "heavy_rain",
    "time": "dusk",
}

S001 = """@scene S001 loc=L01 time=dusk tension=3
~ rain intensity=heavy
> wide
! Monsoon rain hammers the stone arch of {L01}. {C01} hurries up the path, the {P05} folded under her coat.
! {C02} waits under the arch, the {P01} swinging from one hand, the {P02} from the other.
C01: Mr. Ade? I'm Mira, from the archive. They said you'd be expecting me.
C02: They said a team. I see one wet woman with a paper map.
C01: The team is a drone and the boy who flies it. He went ahead to the rampart.
C02 (dry): Then he has already broken the first rule.
C01: Which is?
C02: Nobody walks the walls after dark. The stones move when they are wet.
> over C01
C01: We have four days before the restoration crews start pulling those stones apart. Every night counts.
C02: Every night counts for the stones too. They have stood four hundred years because people listened to them.
! {C01} unfolds the {P05}, shielding it from the rain with her body.
C01: This survey is from 1962. It shows a chapel here, and a passage under it that nobody has drawn since.
% 4
C02: Then nobody has needed it since.
C01: Or nobody was allowed to look.
! A long pause. {C02} turns the {P02} over in his fingers.
C02: Come. The hut is dry. My granddaughter makes tea that could wake the dead, which is the only kind of company this place gets.
= C01 location=L04
= C02 location=L04
! They leave the gate to the rain.
"""

S002 = """@scene S002 loc=L02 time=dusk tension=4
~ wind gusts=strong
> third C03
! On top of {L02}, {C03} crouches behind a battlement, thumbs on the controller. The {P03} hums above him like an angry insect.
C03: Log, day one. Light is almost gone. Wind from the sea, maybe forty kilometres an hour.
! The {P03} dips, recovers, and slides toward the dark mass of the chapel roof.
% 5
~ drone_motor pitch=rising
C03 (under his breath): No, no, no. Not the trees.
! The signal bar drops to zero. The {P03} vanishes behind the broken bell tower.
= C03 holding-=P03
> first C03
! On the controller screen: a frozen frame of a doorway, half buried in vines, and the words SIGNAL LOST.
C03: Okay. That's fine. It's an expensive camera in a haunted chapel. Totally fine.
! He stands, slips on the wet stone, and catches himself on the parapet.
C03: Mira is going to kill me.
= C03 location=L04
"""

S003 = """@scene S003 loc=L04 time=night tension=3
= world time=night lighting=oil_lamp weather=steady_rain
~ rain intensity=steady
> wide
! A one-room hut of whitewashed stone. {C04} pours tea from a dented pot. {C01} dries the {P05} by the lamp. {C02} hangs the {P01} on a nail by the door.
= C02 holding-=P01
C04: Grandfather says you want to scan the whole fort. Every stone.
C01: Every stone, every beam, every crack. If the restoration goes wrong, the scan is all that's left.
! The door bangs open. {C03} stumbles in, soaked to the skin, holding the empty controller.
C03: Before anyone says anything, the wind was not in the forecast.
C01: Tomas. Where is the drone?
C03: Technically? In the chapel. Or on it. The last picture shows a door.
% 5
! {C02} sets down his cup very slowly.
C02: The chapel is closed.
C03: The door looked pretty open to me.
C04: Show me.
! {C03} hands the controller to {C04}. She zooms into the frozen frame.
C04: That's not the chapel door. The chapel door is on the east side. This one goes down.
> over C04
C01 (quietly): The passage on the survey.
C02: Tomorrow. In daylight. With the key, and with me.
C03: The battery dies in six hours. After that the tracker is gone, and so is everything it recorded.
! Silence. Rain on the roof. {C02} looks at the {P01} on its nail.
C02: Then I will fetch it myself. Alone.
C01: Not alone.
C02: Lena, keep them here. I need to check the gate chain before anyone goes anywhere.
! He takes the {P01} from its nail and steps out into the rain.
= C02 holding+=P01 location=L01
"""

S004 = """@scene S004 loc=L04 time=night tension=4
> wide
! {C04} kneels by a tin trunk under the bed and pulls out a cloth bundle.
C04: He doesn't know I have this. It was my grandmother's.
= C04 holding+=P06
! Inside the cloth: the {P06}, brown at the edges. Two young men stand in front of a low arched doorway, holding lanterns.
C04: That's him. The one on the left. Nineteen sixty-two.
C01: The year of the survey.
C04: The other one is his brother. Nobody talks about his brother.
C03 (careful): What happened to him?
C04: He went down that doorway to measure it. The rains came early that year. The passage flooded.
% 6
! {C01} looks from the {P06} to the frozen frame on the controller. The same arch.
C01: So when your grandfather says the chapel is closed...
C04: He means he closed it.
! {C04} hands the {P06} to {C01}.
= C04 holding-=P06
= C01 holding+=P06
C04: If you're going down there, take this. He should see that you know.
C03: We're actually going down there?
C01: Tomas, check what the drone recorded before the signal dropped. Lena, does the old path still reach the chapel?
C04: The west steps. They're slippery, but they hold. I'll show you.
= C01 location=L03
= C03 location=L03
= C04 location=L03
! They pull on their coats and go out into the dark.
"""

S005 = """@scene S005 loc=L03 time=night tension=6
= world lighting=phone_torches
~ rain intensity=heavy
~ thunder distance=far
> third C01
! The chapel has no roof. Rain falls straight into the nave and runs between the flagstones toward the east wall.
! Phone torches sweep over broken pews. {C03} spots the {P03} wedged in a fig tree's roots beside a low arch.
C03: There! It's alive. Look, the light's still blinking.
= C03 holding+=P03
C03: Ninety minutes of footage. And it went inside. It flew right through the doorway before it hit the roots.
C04: Through? That arch has been bricked up since before I was born.
! {C01} crouches at the arch. The lower bricks have fallen in. Cold air breathes out of the gap.
C01: Not anymore.
% 7
~ footsteps surface=stone
= C02 location=L03
! A swinging light on the steps behind them. {C02} stands in the doorway of the nave, the {P01} raised.
C02: Lena.
C04: Grandfather, I can explain.
C02: You brought them here. In this rain. To this door.
! {C01} steps forward and holds out the {P06}.
C01: She brought me this. I think you should be the one to tell me what it is.
! {C02} looks at the {P06} for a long time. The {P01} trembles in his hand.
C02 (barely audible): Where did you find that?
C04: Grandmother kept it. She kept it for you.
"""

S006 = """@scene S006 loc=L03 time=night tension=5
> over C02
! {C02} lowers himself onto a broken pew. The {P01} rests on his knee.
C02: His name was Rafi. He was better at everything. Faster up the walls, better with numbers. The surveyor hired him, not me.
C02: That night the man went home early. Rafi said one more measurement, just to the first bend. I held the lantern at the top of the steps.
C01: And the water came.
C02: It came from below, not above. That is what nobody believed. The sea comes into this hill through a channel older than the fort.
! {C03} glances at the screen of the {P03}, then at {C01}.
C03: Mira. The footage. There's water moving in there, even now. You can see a current.
% 6
C02: I bricked the arch myself. I told the village the chapel was unsafe, and after a while it was true.
! A gust of wind. The arch breathes out again, colder.
C01 (gently): The scan could map it. All of it, without anyone going down. If there's a channel, the restoration has to know before they start drilling.
C02: And Rafi?
C01: If he's down there, we'll know where. And you can stop going down in your head every time it rains.
! {C02} is quiet. Then he takes the {P02} from his belt and holds it out to {C01}.
= C02 holding-=P02
= C01 holding+=P02
C02: The iron gate at the bottom of the steps. This opens it. Not tonight. At low tide, at dawn.
= C01 location=L04
= C02 location=L04
= C03 location=L04
= C04 location=L04
! They climb back up the west steps together, {C02} last, holding the {P01} high.
"""

S007 = """@scene S007 loc=L04 time=night tension=4
= world weather=light_rain lighting=oil_lamp
~ rain intensity=light
> wide
! The hut, after midnight. The {P05} is spread across the table, weighted with teacups. {C03} has the {P03} plugged into a laptop.
C03: The drone got about forty metres in before the roots. The passage drops, then turns north. Toward the sea.
! {C01} draws a pencil line on the {P05}, following his numbers.
C01: The turn is right under the rampart. Where the crews want to put the new drainage.
C04: So if they drill there...
C01: They hit the channel, and the sea comes up the hill. The same thing as nineteen sixty-two, only faster.
% 5
! {C02} sits apart by the door. {C01} slides the {P06} across the table to him.
= C01 holding-=P06
= C02 holding+=P06
! He turns it over in his hands, again and again.
C02: He would have liked you. He liked people who drew lines on maps.
C01: Then help me draw this one right.
C02: Low tide is at five forty. Sleep two hours. All of you.
"""

S008 = """@scene S008 loc=L05 time=dawn tension=7
~ water flow=steady
~ drip interval=irregular
> first C01
! Stone steps lead down into the hill. Grey dawn light reaches the first ten steps and gives up. {C02} goes first with the {P01}. {C01} follows, the {P02} in her fist. {C03} brings up the rear with the {P03} on a short tether.
C02: Count the steps. Forty-one to the gate. Rafi made me count them.
C01: Thirty-nine. Forty. Forty-one.
! An iron gate, orange with rust, across the passage. Beyond it, the sound of moving water.
! {C01} fits the {P02} into the lock. It does not turn.
C01: It's rusted solid.
C02: Turn it toward you first. Then away. It was always stubborn.
~ lock_click
! The lock gives. The gate swings in with a scream of metal. {C01} leaves the {P02} in the lock.
= C01 holding-=P02
% 8
C03: Drone's up. Scanning. Stay behind it, the scanner hates people.
! The {P03} drifts ahead, a small red light painting the walls. Water runs knee deep along the floor, toward the north.
C02: The bend. This is where I lost the light.
! The passage turns. The light of the {P03} finds a ledge above the waterline. On the ledge, something wrapped in rotten canvas, and a rusted lantern beside it.
C03 (very quiet): Mira.
! {C01} raises her hand to stop him. {C02} has already seen.
C02: Rafi.
% 9
~ water flow=rising
C03: Water's coming up. Fast. The tide must have turned early.
C01: Ade, we have to go. We know where he is now. We can bring him out properly.
C02: Fifty years. Fifty years he has been waiting at the bend.
C01: And he can wait one more day, with people who know his name. Please.
! {C02} touches the canvas once. Then he turns and starts back toward the gate.
= C01 location=L03
= C02 location=L03
= C03 location=L03
! They climb, the water loud behind them.
"""

S009 = """@scene S009 loc=L03 time=dawn tension=5
> wide
! {C04} waits at the top of the steps, wrapped in a blanket. The rain has thinned to a mist. They come up out of the ground one by one, soaked to the waist.
C04: You were gone forty minutes. I almost came down.
C02: Good that you didn't.
! {C02} sits on the chapel step. {C04} sits beside him without asking.
C04: Did you find him?
C02: We found him.
! She puts her head on his shoulder. He lets her.
= world weather=clearing lighting=sunrise
% 4
C03: I got all of it. The channel, the bend, the ledge. Every centimetre.
C01: Send it to the restoration office before they wake up. With the drainage plan crossed out.
C01 (to Ade): They'll want to bring him up with the right people. A priest, maybe. Family.
C02: Family is here.
! {C02} takes the {P06} out of his shirt and gives it to {C04}.
= C02 holding-=P06
= C04 holding+=P06
C02: Your grandmother kept this for me. Now you keep it.
"""

S010 = """@scene S010 loc=L02 time=morning tension=2
~ sea distance=far
~ birds
> wide
! Morning on {L02}, two days later. The sea is flat and silver. The restoration crews' trucks are parked at the gate, engines off.
! {C03} holds out the {P07} of {C04}. On the screen, the fort turns slowly in three dimensions, the passage a blue thread under the hill.
C03: That's it. Every wall, every step. You can walk through it at night now and nobody gets wet.
C04: Can you put a light at the bend?
C03: Easy.
! A small light appears in the model, deep under the hill.
C02: A lantern. Make it a lantern.
C03 (smiling): Done.
% 3
C01: The office called. The drainage goes on the south side. And they want you on the restoration team, Ade. Someone who knows where the stones move.
C02: I am seventy-four years old.
C01: That's why.
! {C02} laughs, for the first time. He opens the glass of the {P01} and blows out the flame.
= C02 holding-=P01
! He sets the {P01} on the battlement, facing the sea.
% 2
C04: Grandfather? Will you show me the steps? Properly. Forty-one.
C02: Forty-one. Come. Before the crews start.
> wide
! They walk along the wall, the old man and his granddaughter, the fort whole around them.
"""

# (id, title, summary, location, time of day, tension, scene text, offscreen continuity)
SCENES: list[tuple[str, str, str, str, str, int, str, dict[str, Any]]] = [
    ("S001", "Locked Gate", "Mira arrives in the monsoon; Ade warns her off the walls and the "
     "1962 survey shows a forgotten passage.", "L01", "dusk", 3, S001, {}),
    ("S002", "Signal Lost", "Tomas loses the drone over the chapel ruin; its last frame shows a "
     "doorway.", "L02", "dusk", 4, S002, {}),
    ("S003", "Tea and Rules", "In Ade's hut Lena recognises the doorway as the way down; Ade "
     "refuses, then goes out alone.", "L04", "night", 3, S003, {}),
    ("S004", "The Photograph", "Lena shows a 1962 photograph of Ade and his lost brother; the three "
     "set off for the chapel.", "L04", "night", 4, S004, {}),
    ("S005", "The Open Arch", "They find the drone by a broken arch; Ade arrives and sees the "
     "photograph.", "L03", "night", 6, S005, {}),
    ("S006", "Brothers", "Ade tells how his brother Rafi drowned in the passage and gives Mira the "
     "key for dawn.", "L03", "night", 5, S006, {}),
    ("S007", "Low Tide", "The drone footage shows the channel runs under the planned drainage; "
     "they wait for low tide.", "L04", "night", 4, S007, {}),
    ("S008", "Forty-One Steps", "Mira, Ade and Tomas open the gate, scan the passage and find "
     "Rafi's remains before the tide returns.", "L05", "dawn", 7, S008,
     {"offscreen": ["C01.location", "C02.location", "C03.location", "C04.location",
                    "world.time", "world.lighting", "world.weather"],
      "characters": {c: {"location": loc} for c, loc in
                     (("C01", "L05"), ("C02", "L05"), ("C03", "L05"), ("C04", "L03"))},
      "time": "dawn", "lighting": "grey_dawn", "weather": "drizzle"}),
    ("S009", "Surface", "Back at the chapel at sunrise; the scan goes to the restoration office "
     "and Ade gives the photograph to Lena.", "L03", "dawn", 5, S009, {}),
    ("S010", "The Model", "Two days later on the rampart the fort is a 3D model; Ade puts out the "
     "lantern and joins the restoration.", "L02", "morning", 2, S010,
     {"offscreen": ["C01.location", "C02.location", "C03.location", "C04.location",
                    "world.time", "world.lighting", "world.weather"],
      "characters": {c: {"location": "L02"} for c in ("C01", "C02", "C03", "C04")},
      "time": "morning", "lighting": "morning_sun", "weather": "clear"}),
]  # fmt: skip


def build() -> Script:
    index = [
        {"id": sid, "title": title, "summary": summary, "location": loc, "time_of_day": tod,
         "tension": tension, "continuity": START if i == 0 else offscreen}
        for i, (sid, title, summary, loc, tod, tension, _, offscreen) in enumerate(SCENES)
    ]  # fmt: skip
    header = {
        "format": "wscript",
        "format_version": "1.1",
        "title": "The Lantern Keeper",
        "language": "en",
        "dictionary": DICTIONARY,
        "scenes": index,
    }
    scenes = {sid: parse_scene(text) for sid, *_, text, _ in SCENES}
    return propagate(Script(header=header, scenes=scenes))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m tools.make_sample_wscript")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--zip", action="store_true")
    p.add_argument("--force", action="store_true", help="replace an existing output")
    a = p.parse_args(argv)
    if a.out.exists():
        if not a.force:
            p.error(f"{a.out} exists (use --force to replace it)")
        shutil.rmtree(a.out) if a.out.is_dir() else a.out.unlink()
    write_wscript(a.out, build(), as_zip=a.zip)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

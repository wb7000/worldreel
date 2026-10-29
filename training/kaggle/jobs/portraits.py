# Job: portrait candidates for every character of the sample script (A-14, D-8.10).
# Kandinsky 2.2 (prior 9fc51ad5732a + decoder 9ae140d347fe, Apache-2.0), fp16, 512 x 512, 50 steps,
# CANDIDATES seeded candidates per character (ONLY: the characters to redo, [] = all) from appearance-only prompts. Choosing the set is done locally.
#   python -m tools.kaggle_runner run training/kaggle/jobs/portraits.py
import json
import subprocess
import sys
import textwrap
from pathlib import Path

from formats.wscript import read_wscript
from stages.generate.portraits import NEGATIVE, candidate_seeds, portrait_prompt

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
UV = [sys.executable, "-m", "uv"]
py = Path("/tmp/img/bin/python")
subprocess.run([*UV, "venv", "--python", "3.12", "/tmp/img"], check=True)
r = subprocess.run([*UV, "pip", "install", "--python", str(py), "diffusers", "transformers",
                    "accelerate", "safetensors", "torch"], capture_output=True, text=True)  # fmt: skip
log(f"img install rc={r.returncode} {r.stderr[-300:] if r.returncode else ''}")

ONLY = ["C02"]  # D-8.10: Ade's prompt changed (facial-hair scale); the others are unchanged
chars = read_wscript(CODE / "eval" / "scripts" / "lantern_keeper.wscript").header["dictionary"][
    "characters"
]
items = [
    {"file": f"portraits/{cid}_{k}.png", "cid": cid, "prompt": portrait_prompt(c), "seed": s}
    for cid, c in sorted(chars.items())
    if not ONLY or cid in ONLY
    for k, s in enumerate(candidate_seeds(c))
]
(OUT / "portraits").mkdir(exist_ok=True)
save_json("items.json", items)
step = WORK / "step_portraits.py"
step.write_text(
    textwrap.dedent("""
    import json, sys, time, torch
    from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
    a = json.load(open(sys.argv[1])); out = sys.argv[2]
    prior = KandinskyV22PriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior",
        revision="9fc51ad5732a", torch_dtype=torch.float16).to("cuda")
    dec = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder",
        revision="9ae140d347fe", torch_dtype=torch.float16).to("cuda")
    t0 = time.time()
    for it in a["items"]:
        g = torch.Generator("cuda").manual_seed(it["seed"])
        emb, neg = prior(it["prompt"], negative_prompt=a["negative"], generator=g).to_tuple()
        img = dec(image_embeds=emb, negative_image_embeds=neg, height=512, width=512,
                  num_inference_steps=50, generator=g).images[0]
        img.save(f"{out}/{it['file']}")
        print(it["file"], flush=True)
    json.dump({"seconds": round(time.time() - t0, 1),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)},
              open(f"{out}/portraits.json", "w"))
"""),
    encoding="utf-8",
)
(WORK / "portraits_args.json").write_text(json.dumps({"items": items, "negative": NEGATIVE}))
r = subprocess.run([str(py), str(step), str(WORK / "portraits_args.json"), str(OUT)],
                   capture_output=True, text=True, timeout=3 * 3600)  # fmt: skip
(OUT / "portraits.log").write_text(r.stdout[-20000:] + "\n" + r.stderr[-20000:], encoding="utf-8")
log(f"portraits rc={r.returncode} {r.stderr[-800:] if r.returncode else ''}")
log("done")

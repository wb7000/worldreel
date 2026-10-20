# Job: the director (A-13, D-8.8). Qwen2.5-7B-Instruct (Apache-2.0, 4-bit NF4) directs every scene
# of the test set (6 scenes with hand labels) and of The Lantern Keeper (10 scenes): manner,
# emotion, intensity and spoken text per line, validated and meaning-checked into a delivery layer.
#   python -m tools.kaggle_runner run training/kaggle/jobs/director.py
import json
import subprocess
import sys
import textwrap
from pathlib import Path

from eval.director_eval import _norm, score, test_script
from formats.wscript import read_wscript, write_wscript
from stages.generate.director import build_prompt, parse_answer, to_layer

MODEL = "Qwen/Qwen2.5-14B-Instruct"  # v2 (D-8.8b); v1 was 7B
REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
SOURCE = f"director:{MODEL}@{REVISION[:7]}"
report: dict[str, object] = {}
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
UV = [sys.executable, "-m", "uv"]

py = Path("/tmp/llm/bin/python")
subprocess.run([*UV, "venv", "--python", "3.12", "/tmp/llm"], check=True)
r = subprocess.run([*UV, "pip", "install", "--python", str(py), "transformers", "accelerate",
                    "bitsandbytes", "torch"], capture_output=True, text=True)  # fmt: skip
log(f"llm install rc={r.returncode} {r.stderr[-300:] if r.returncode else ''}")

test, gold = test_script()
sample = read_wscript(CODE / "eval" / "scripts" / "lantern_keeper.wscript")
jobs = [{"key": f"test/{sid}", "messages": build_prompt(test, sid)} for sid in test.scenes]
jobs += [{"key": f"lantern/{sid}", "messages": build_prompt(sample, sid)} for sid in sample.scenes]
(WORK / "director_args.json").write_text(
    json.dumps({"jobs": jobs, "model": MODEL, "rev": REVISION})
)
(WORK / "step_director_llm.py").write_text(
    textwrap.dedent("""
    import json, sys, time, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    a = json.load(open(sys.argv[1])); out = sys.argv[2]
    tok = AutoTokenizer.from_pretrained(a["model"], revision=a["rev"])
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.float16)
    m = AutoModelForCausalLM.from_pretrained(a["model"], revision=a["rev"], quantization_config=q,
                                             device_map="cuda")
    def ask(messages):
        enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True)  # transformers 5: a dict, not a tensor
        ids = enc["input_ids"].to("cuda")
        with torch.inference_mode():
            torch.manual_seed(7)
            o = m.generate(ids, max_new_tokens=3000, do_sample=True, temperature=0.7, top_p=0.9,
                           pad_token_id=tok.eos_token_id)
        return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)
    res = {}
    for j in a["jobs"]:
        t0 = time.time()
        text = ask(j["messages"])
        retried = False
        try:
            s, e = text.find("{"), text.rfind("}")
            json.loads(text[s:e + 1])["lines"]
        except Exception as exc:
            retried = True
            text = ask(j["messages"] + [{"role": "assistant", "content": text},
                       {"role": "user", "content": f"That was not valid JSON ({exc}). Answer again with the JSON only."}])
        res[j["key"]] = {"answer": text, "retried": retried, "seconds": round(time.time() - t0, 1)}
        print(j["key"], res[j["key"]]["seconds"], "s", "retried" if retried else "", flush=True)
    json.dump({"answers": res, "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)},
              open(f"{out}/answers.json", "w"))
"""),
    encoding="utf-8",
)
r = subprocess.run([str(py), str(WORK / "step_director_llm.py"), str(WORK / "director_args.json"), str(OUT)],
                   capture_output=True, text=True, timeout=5 * 3600)  # fmt: skip
(OUT / "director_llm.log").write_text(
    r.stdout[-20000:] + "\n" + r.stderr[-20000:], encoding="utf-8"
)
log(f"director_llm rc={r.returncode} {r.stderr[-800:] if r.returncode else ''}")
if r.returncode:
    raise SystemExit("director failed")

answers = json.loads((OUT / "answers.json").read_text())
report["peak_gb"] = answers["peak_gb"]
problems: dict[str, list[str]] = {}
parse_failures = []
for key, a in answers["answers"].items():
    group, sid = key.split("/")
    script = test if group == "test" else sample
    try:
        parsed = parse_answer(a["answer"])
    except ValueError as exc:
        parse_failures.append(f"{key}: {exc}")
        parsed = []
    layer, probs = to_layer(script, sid, parsed, SOURCE)
    script.delivery[sid] = layer
    problems[key] = probs
report["parse_failures"] = parse_failures
report["retried"] = [k for k, a in answers["answers"].items() if a["retried"]]
report["seconds"] = {k: a["seconds"] for k, a in answers["answers"].items()}
report["problems"] = problems
report["test_score"] = score(test, gold)
lantern = [d for sid in sample.scenes for d in sample.delivery[sid]["lines"].values()]

changed = [
    d["meaning_ok"] and _norm(d["spoken"]) != _norm(t)
    for sid in sample.scenes
    for (i, _, t) in __import__("stages.generate.director", fromlist=["x"]).dialogue_lines(
        sample, sid
    )
    for d in [sample.delivery[sid]["lines"][str(i)]]
]
report["lantern"] = {
    "spoken_changed_ok": round(sum(changed) / len(changed), 3),
    "lines": len(lantern),
    "meaning_ok": round(sum(d["meaning_ok"] for d in lantern) / len(lantern), 3),
    "emotions": {
        e: sum(d["emotion"] == e for d in lantern) for e in sorted({d["emotion"] for d in lantern})
    },
}
write_wscript(OUT / "lantern_keeper_directed.wscript", sample, as_zip=False)
write_wscript(OUT / "director_test.wscript", test, as_zip=False)
log(json.dumps({k: v for k, v in report["test_score"].items() if k != "rows"}))
log(json.dumps(report["lantern"]))
save_json("report.json", report)
log("done")

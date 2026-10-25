# Job: the director (A-13, D-8.8c v3). Qwen2.5-14B-Instruct (Apache-2.0, 4-bit NF4) directs every
# scene of the test set (6 scenes with hand labels) and of The Lantern Keeper (10 scenes):
#   stage 1 — manner, emotion, intensity (and a first spoken try) per line;
#   stage 2 — three spoken takes per line given its manner and emotion; the first take that
#             differs from the written line and passes the meaning check is kept (best of 3).
#   python -m tools.kaggle_runner run training/kaggle/jobs/director.py
import json
import subprocess
import sys
import textwrap
from pathlib import Path

from eval.director_eval import _norm, score, test_script
from formats.wscript import read_wscript, write_wscript
from stages.generate.director import (
    apply_rewrites,
    build_prompt,
    build_rewrite_prompt,
    dialogue_lines,
    parse_answer,
    rewrite_chunks,
    to_layer,
)

MODEL = "Qwen/Qwen2.5-14B-Instruct"  # v2+ (D-8.8b); v1 was 7B
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

STEP = WORK / "step_director_llm.py"
STEP.write_text(
    textwrap.dedent("""
    import json, sys, time, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    a = json.load(open(sys.argv[1])); out_file = sys.argv[2]
    tok = AutoTokenizer.from_pretrained(a["model"], revision=a["rev"])
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.float16)
    m = AutoModelForCausalLM.from_pretrained(a["model"], revision=a["rev"], quantization_config=q,
                                             device_map="cuda")
    def ask(messages):
        enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True)  # transformers 5: a dict
        ids = enc["input_ids"].to("cuda")
        torch.manual_seed(7)
        with torch.inference_mode():
            o = m.generate(ids, max_new_tokens=3500, do_sample=True, temperature=0.7, top_p=0.9,
                           pad_token_id=tok.eos_token_id)
        return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)
    res = {}
    for j in a["jobs"]:
        t0 = time.time()
        text = ask(j["messages"]); retried = False
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
              open(out_file, "w"))
"""),
    encoding="utf-8",
)


def llm(name: str, jobs: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    args = WORK / f"{name}_args.json"
    args.write_text(json.dumps({"jobs": jobs, "model": MODEL, "rev": REVISION}))
    out_file = OUT / f"{name}_answers.json"
    rr = subprocess.run([str(py), str(STEP), str(args), str(out_file)],
                        capture_output=True, text=True, timeout=5 * 3600)  # fmt: skip
    (OUT / f"{name}.log").write_text(
        rr.stdout[-20000:] + "\n" + rr.stderr[-20000:], encoding="utf-8"
    )
    log(f"{name}: rc={rr.returncode} {rr.stderr[-800:] if rr.returncode else ''}")
    if rr.returncode:
        raise SystemExit(f"{name} failed")
    data = json.loads(out_file.read_text())
    report[f"{name}_peak_gb"] = data["peak_gb"]
    report[f"{name}_retried"] = [k for k, v in data["answers"].items() if v["retried"]]
    answers: dict[str, dict[str, object]] = data["answers"]
    return answers


test, gold = test_script()
sample = read_wscript(CODE / "eval" / "scripts" / "lantern_keeper.wscript")
scripts = {"test": test, "lantern": sample}

# Stage 1: manner / emotion / intensity (reused from an attached earlier run when present —
# stage 1 is unchanged since v2; saves ~30 min of GPU)
prev = next(INPUT.rglob("stage1_answers.json"), None)
if prev is not None:
    a1 = json.loads(prev.read_text())["answers"]
    (OUT / "stage1_answers.json").write_text(prev.read_text())
    log(f"stage 1 reused from {prev}")
else:
    a1 = None
a1 = a1 or llm("stage1", [{"key": f"{g}/{sid}", "messages": build_prompt(s, sid)}
                    for g, s in scripts.items() for sid in s.scenes])  # fmt: skip
problems: dict[str, list[str]] = {}
for key, a in a1.items():
    g, sid = key.split("/")
    try:
        parsed = parse_answer(str(a["answer"]))
    except ValueError as exc:
        parsed, problems[key] = [], [f"stage 1 parse: {exc}"]
    layer, probs = to_layer(scripts[g], sid, parsed, SOURCE)
    scripts[g].delivery[sid] = layer
    problems.setdefault(key, []).extend(probs)
report["stage1_test_score"] = {k: v for k, v in score(test, gold).items() if k != "rows"}

# Stage 2: five spoken takes per line, chunks of <= 6 lines, first good take kept (v4)
a2 = llm("stage2", [
    {"key": f"{g}/{sid}/{k}", "messages": build_rewrite_prompt(s, sid, s.delivery[sid], only=chunk)}
    for g, s in scripts.items() for sid in s.scenes
    for k, chunk in enumerate(rewrite_chunks(s, sid))
])  # fmt: skip
for key, a in a2.items():
    g, sid, _chunk = key.split("/")
    try:
        parsed = parse_answer(str(a["answer"]))
    except ValueError as exc:
        parsed = []
        problems.setdefault(f"{g}/{sid}", []).append(f"stage 2 parse: {exc}")
    problems.setdefault(f"{g}/{sid}", []).extend(
        apply_rewrites(scripts[g], sid, scripts[g].delivery[sid], parsed)
    )
report["problems"] = problems
report["test_score"] = score(test, gold)
lantern = [
    (d, t)
    for sid in sample.scenes
    for (i, _, t) in dialogue_lines(sample, sid)
    for d in [sample.delivery[sid]["lines"][str(i)]]
]
report["lantern"] = {
    "lines": len(lantern),
    "spoken_changed_ok": round(
        sum(d["meaning_ok"] and _norm(d["spoken"]) != _norm(t) for d, t in lantern) / len(lantern),
        3,
    ),
    "meaning_ok": round(sum(d["meaning_ok"] for d, _ in lantern) / len(lantern), 3),
    "emotions": {
        e: sum(d["emotion"] == e for d, _ in lantern)
        for e in sorted({d["emotion"] for d, _ in lantern})
    },
}
write_wscript(OUT / "lantern_keeper_directed.wscript", sample, as_zip=False)
write_wscript(OUT / "director_test.wscript", test, as_zip=False)
log("stage1 " + json.dumps(report["stage1_test_score"]))
log("final " + json.dumps({k: v for k, v in report["test_score"].items() if k != "rows"}))
log("lantern " + json.dumps(report["lantern"]))
save_json("report.json", report)
log("done")

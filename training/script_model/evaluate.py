"""Evaluate base vs fine-tuned script model on the held-out prompts (D-7.5). Needs a GPU,
`torch`, `transformers` and `peft` (the Soup environment on Kaggle).

    python -m training.script_model.evaluate --data DIR --adapter DIR --out DIR

For every prompt in `test.jsonl` (held-out plays) and `ood.jsonl` (The Lantern Keeper): greedy
generation (≤ 320 new tokens), scored by `stages.generate.script_model.score`; and the teacher-
forced loss of the true next lines. Also the loss on `val.jsonl`. Writes `metrics.json` and
`samples.jsonl` (prompt tail, gold, both models' outputs).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from stages.generate.script_model import score

MAX_NEW_TOKENS = 320
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main(argv: list[str] | None = None) -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.float16
    ).to("cuda")
    base.eval()
    tuned = PeftModel.from_pretrained(base, str(a.adapter))
    tuned.eval()

    def prompt_ids(msgs: list[dict[str, str]]) -> Any:
        text = tok.apply_chat_template(msgs[:2], tokenize=False, add_generation_prompt=True)
        return tok(text, return_tensors="pt").input_ids.to("cuda")

    def generate(model: Any, msgs: list[dict[str, str]], adapter_on: bool) -> str:
        ids = prompt_ids(msgs)
        ctx = model.disable_adapter() if not adapter_on else _NoAdapterChange()
        with ctx:
            out = model.generate(
                ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        return str(tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True))

    def gold_loss(model: Any, msgs: list[dict[str, str]], adapter_on: bool) -> tuple[float, int]:
        ids = prompt_ids(msgs)
        gold = tok(msgs[2]["content"] + tok.eos_token, return_tensors="pt").input_ids.to("cuda")
        full = torch.cat([ids, gold], dim=1)
        labels = full.clone()
        labels[:, : ids.shape[1]] = -100
        ctx = model.disable_adapter() if not adapter_on else _NoAdapterChange()
        with ctx:
            loss = model(full, labels=labels).loss
        n = int(gold.shape[1])
        return float(loss) * n, n

    torch.set_grad_enabled(False)  # evaluation only
    metrics: dict[str, Any] = {}
    samples: list[dict[str, Any]] = []
    for split in ("test", "ood", "val"):
        rows = _rows(a.data / f"{split}.jsonl")
        if a.limit:
            rows = rows[: a.limit]
        res: dict[str, Any] = {"n": len(rows)}
        for name, on in (("base", False), ("tuned", True)):
            t0 = time.time()
            tot, cnt, syn, ref = 0.0, 0, 0, 0
            for r in rows:
                s, n = gold_loss(tuned, r["messages"], on)
                tot, cnt = tot + s, cnt + n
                if split == "val":
                    continue
                text = generate(tuned, r["messages"], on)
                d = {k: dict.fromkeys(v) for k, v in r["meta"]["dictionary"].items()}
                sc = score(text, d)
                syn, ref = syn + sc.syntax_ok, ref + sc.refs_ok
                if name == "base":
                    samples.append(
                        {"split": split,
                         "meta": {k: v for k, v in r["meta"].items() if k != "dictionary"},
                         "prompt_tail": r["messages"][1]["content"][-600:],
                         "gold": r["messages"][2]["content"], "base": text,
                         "base_score": sc.__dict__}
                    )  # fmt: skip
                else:
                    match = next(
                        x for x in samples
                        if x["split"] == split and x["meta"]["cut"] == r["meta"]["cut"]
                        and x["meta"]["scene"] == r["meta"]["scene"]
                        and x["meta"]["title"] == r["meta"]["title"]
                    )  # fmt: skip
                    match["tuned"], match["tuned_score"] = text, sc.__dict__
            res[name] = {
                "loss": round(tot / cnt, 4),
                "perplexity": round(math.exp(tot / cnt), 2),
                "seconds": round(time.time() - t0, 1),
            }
            if split != "val":
                res[name] |= {
                    "syntax": round(syn / len(rows), 4),
                    "refs": round(ref / len(rows), 4),
                }
            print(split, name, res[name], flush=True)
        metrics[split] = res
    (a.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (a.out / "samples.jsonl").write_text(
        "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples), encoding="utf-8"
    )
    return 0


class _NoAdapterChange:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


if __name__ == "__main__":
    raise SystemExit(main())

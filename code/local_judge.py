"""LLM-as-judge evaluation with a LOCAL open-weights model - no API, no cost.

Scores every system on the SAME random sample of test documents along the four
SummEval dimensions (Fabbri et al., 2021; Gao et al., 2023), 1-5, greedy decoding.

Why local: the judge runs on the same GPU used for training, so the evaluation costs
nothing and - unlike a commercial endpoint whose weights change without notice - it is
exactly reproducible from the model name and revision recorded in the output file.

Usage
  python local_judge.py --dataset samsum
  python local_judge.py --dataset cnn_dm --n 300 --judge Qwen/Qwen2.5-7B-Instruct

Output
  runs/<ds>__ljudge_cache.jsonl   one row per (doc, system); the script is resumable
  runs/<ds>__ljudge.json          per-system means, standard errors and the config

Notes
  * Every summary is scored on its own, so there is no presentation-order bias between
    systems, and the judge never sees which system produced a summary.
  * Runs are skipped, not invented: a system whose predictions are missing is reported
    as missing rather than silently dropped from the comparison.
"""
import argparse
import json
import os
import random
import re
import time

import numpy as np

from common import EVAL_SUBSET_SEED, append_jsonl, load_dataset_splits, read_jsonl, run_name

DIMS = ["consistency", "relevance", "fluency", "coherence"]

SYSTEMS = {
    "frozen": dict(method="frozen"),
    "lora4":  dict(method="lora", rank=4),
    "lora8":  dict(method="lora", rank=8),
    "lora16": dict(method="lora", rank=16),
    "full":   dict(method="full"),
}

# Bump this whenever PROMPT changes: cached rows written under an older version are
# ignored, so a re-run never mixes scores produced by two different instruments.
#   v1 - ended with a worked example whose values were all 3. The judge copied it, and
#        every system scored ~3.0 regardless of quality. Do not reuse v1 rows.
PROMPT_VERSION = 2

PROMPT = """You are a strict evaluator of automatically generated summaries.

Read the source document and the summary, then rate the summary on four dimensions,
each an integer from 1 to 5.

Use the whole scale. Summaries differ a lot in quality: when one is clearly poor give it
a 1 or a 2, and when it is clearly good give it a 4 or a 5. Do not default to the middle
of the scale.

- consistency: are the statements in the summary supported by the source?
  1 = mostly invented or contradicted by the source; 5 = every statement is supported.
- relevance: does the summary capture what matters and leave out the rest?
  1 = misses the main point, or is mostly filler; 5 = captures exactly the key content.
- fluency: grammar, spelling, punctuation and word choice of the individual sentences.
  1 = broken or hard to read; 5 = clean and natural.
- coherence: is the summary organised and readable as a whole?
  1 = disconnected fragments; 5 = well structured and easy to follow.

SOURCE DOCUMENT:
{source}

SUMMARY:
{summary}

First write one short sentence justifying your assessment. Then, on the final line,
write only the scores, as a JSON object with exactly these four integer keys:
{{"consistency": <1-5>, "relevance": <1-5>, "fluency": <1-5>, "coherence": <1-5>}}"""


def truncate_words(text, n):
    w = text.split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


def _as_scores(obj):
    out = {}
    for d in DIMS:
        if d not in obj:
            return None
        try:
            v = int(round(float(obj[d])))
        except (TypeError, ValueError):
            return None
        if not 1 <= v <= 5:
            return None
        out[d] = v
    return out


def parse_scores(txt):
    """Pull the score object out of the model's reply; None if it is unusable.

    Scans every brace-delimited candidate and keeps the LAST one that carries all four
    dimensions, so a preamble - or a reasoning model's <think> block, which may itself
    contain braces and draft scores - cannot be mistaken for the final answer.
    """
    txt = re.sub(r"<think>.*?</think>", " ", txt, flags=re.S | re.I)
    best = None
    for m in re.finditer(r"\{[^{}]*\}", txt, re.S):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        sc = _as_scores(obj)
        if sc:
            best = sc
    return best


class Judge:
    def __init__(self, model_name, revision=None, max_new_tokens=160, batch_size=8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.name = model_name
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        kw = {"revision": revision} if revision else {}
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left", **kw)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
        dtype = torch.bfloat16 if cap >= (8, 0) else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto", **kw)
        self.model.eval()
        self.dtype = str(dtype)
        # the exact commit actually loaded, so the evaluation can be reproduced later
        self.revision = getattr(getattr(self.model, "config", None), "_commit_hash", None) or revision
        print(f"[judge] {model_name} (revision {self.revision}) loaded in {dtype}", flush=True)

    def _render(self, user_msg):
        """Chat template, with a reasoning model's thinking mode switched off.

        Qwen3 and other hybrid-reasoning models emit a <think> block by default, which
        wastes the generation budget and can leave the reply truncated before the JSON.
        Templates that do not know the argument simply reject it, so fall back.
        """
        msgs = [{"role": "user", "content": user_msg}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True,
                                                enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True)

    def score_batch(self, pairs):
        """pairs: list of (source, summary). Returns list of score dicts or None."""
        texts = [self._render(PROMPT.format(source=s, summary=y)) for s, y in pairs]
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=3072).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, temperature=None, top_p=None,
                                      pad_token_id=self.tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        raw = self.tok.batch_decode(gen, skip_special_tokens=True)
        return [(parse_scores(t), t) for t in raw]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=300, help="documents sampled per dataset")
    ap.add_argument("--judge", default="Qwen/Qwen2.5-7B-Instruct",
                    help="hub name or a local directory")
    ap.add_argument("--judge_revision", default=None,
                    help="pin a hub commit/tag so the evaluation stays reproducible")
    ap.add_argument("--out_root", default="runs")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=None,
                    help="judge the models trained with this budget instead of the default")
    ap.add_argument("--seed", type=int, default=None,
                    help="judge the models trained with this seed instead of the default")
    ap.add_argument("--systems", default=None,
                    help="comma-separated subset, e.g. lora8,full (default: all five)")
    ap.add_argument("--show", type=int, default=3,
                    help="print this many raw judge replies, to check it is really judging")
    ap.add_argument("--src_words", type=int, default=600,
                    help="source truncation, to keep the judge prompt inside its context")
    a = ap.parse_args()

    # A "variant" is a training budget / seed other than the default. It is part of the
    # cache key and of the output filename, so scores for models trained differently can
    # never be mixed with - or silently reuse - the scores of the default models.
    tags = []
    if a.epochs is not None:
        tags.append(f"ep{a.epochs:g}")
    if a.seed is not None:
        tags.append(f"seed{a.seed}")
    variant = "_".join(tags)
    extra = {}
    if a.epochs is not None:
        extra["epochs"] = a.epochs
    if a.seed is not None:
        extra["seed"] = a.seed
    wanted = [x.strip() for x in a.systems.split(",")] if a.systems else list(SYSTEMS)

    # ---- collect the systems that actually have predictions ----------------
    preds, missing = {}, []
    for s, kw in SYSTEMS.items():
        if s not in wanted:
            continue
        if s == "frozen" and a.epochs is not None:
            continue          # the zero-shot model is never trained, so it has no budget
        p = os.path.join(a.out_root, run_name(a.dataset, **kw, **extra), "predictions.jsonl")
        rows = read_jsonl(p)
        if rows:
            preds[s] = {r["id"]: r["prediction"] for r in rows}
        else:
            missing.append(s)
            print(f"[missing] {s}: {p}")
    if not preds:
        raise SystemExit("no predictions found - run train_eval.py first")

    # the same documents for every system, chosen by a fixed seed
    ids = sorted(set.intersection(*[set(v) for v in preds.values()]))
    random.Random(EVAL_SUBSET_SEED).shuffle(ids)
    ids = sorted(ids[:a.n])
    sources = {ex["id"]: ex["source"] for ex in load_dataset_splits(a.dataset)["test"]
               if ex["id"] in set(ids)}

    cache_path = os.path.join(a.out_root, f"{a.dataset}__ljudge_cache.jsonl")
    all_rows = read_jsonl(cache_path)
    cache = {(r["id"], r["system"]): r for r in all_rows
             if r.get("judge") == a.judge and r.get("pv") == PROMPT_VERSION
             and (r.get("variant") or "") == variant}
    stale = sum(1 for r in all_rows if r.get("pv") != PROMPT_VERSION)
    if stale:
        print(f"[judge] ignoring {stale} cached rows from an older prompt version")
    todo = [(i, s) for i in ids for s in preds if (i, s) not in cache]
    print(f"[judge] {len(ids)} docs x {len(preds)} systems; {len(todo)} to score, "
          f"{len(cache)} cached", flush=True)

    t0 = time.time()
    if todo:
        judge = Judge(a.judge, revision=a.judge_revision, batch_size=a.batch_size)
        unparsed, shown = 0, 0
        for b in range(0, len(todo), a.batch_size):
            chunk = todo[b:b + a.batch_size]
            pairs = [(truncate_words(sources[i], a.src_words), preds[s][i]) for i, s in chunk]
            for (i, s), (sc, raw) in zip(chunk, judge.score_batch(pairs)):
                if shown < a.show:
                    shown += 1
                    print(f"\n----- raw reply {shown} (system={s}) -----\n{raw.strip()}\n"
                          f"----- parsed: {sc} -----\n", flush=True)
                if sc is None:
                    unparsed += 1
                    continue
                row = {"id": i, "system": s, "judge": a.judge, "pv": PROMPT_VERSION,
                       "variant": variant, **sc}
                append_jsonl(cache_path, [row])
                cache[(i, s)] = row
            print(f"  {min(b + a.batch_size, len(todo))}/{len(todo)}", flush=True)
        if unparsed:
            pct = 100.0 * unparsed / len(todo)
            print(f"[warn] {unparsed} of {len(todo)} replies ({pct:.1f}%) could not be parsed")
            if pct > 20:
                print("[warn] that is a lot - check the raw replies above; the model may be "
                      "ignoring the output format.")

    # ---- aggregate ---------------------------------------------------------
    wall = time.time() - t0
    out = {"judge": a.judge, "judge_revision": judge.revision if todo else a.judge_revision,
           "n_requested": a.n, "dataset": a.dataset,
           "cost": {"judgements_scored": len(todo), "wall_seconds": round(wall, 1),
                    "judgements_per_second": round(len(todo) / wall, 2) if wall > 0 and todo else None,
                    "gpu_minutes": round(wall / 60.0, 1), "usd": 0.0,
                    "note": "one GPU, open-weights judge: the monetary cost is zero"},
           "missing_systems": missing, "systems": {}}
    for s in preds:
        rows = [cache[(i, s)] for i in ids if (i, s) in cache]
        if not rows:
            continue
        e = {"n": len(rows)}
        for d in DIMS:
            v = np.array([r[d] for r in rows], dtype=float)
            e[d] = float(v.mean())
            e[d + "_se"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        overall = np.array([np.mean([r[d] for d in DIMS]) for r in rows], dtype=float)
        e["overall"] = float(overall.mean())
        e["overall_se"] = float(overall.std(ddof=1) / np.sqrt(len(overall))) if len(overall) > 1 else 0.0
        out["systems"][s] = e

    # ---- did the judge actually discriminate? ------------------------------
    # A judge that answers the same number everywhere produces a table that looks
    # finished but measures nothing, so check before the numbers reach the paper.
    diag = {}
    allrows = [cache[(i, s)] for s in preds for i in ids if (i, s) in cache]
    for d in DIMS:
        vals = [r[d] for r in allrows]
        hist = {str(k): vals.count(k) for k in range(1, 6)}
        modal = max(hist.values()) / len(vals) if vals else 0.0
        diag[d] = {"histogram": hist, "modal_fraction": modal}
    means = [out["systems"][s]["overall"] for s in out["systems"]]
    spread = max(means) - min(means) if means else 0.0
    diag["overall_spread_between_systems"] = spread
    worst_modal = max(diag[d]["modal_fraction"] for d in DIMS)
    diag["degenerate"] = bool(worst_modal > 0.6 or spread < 0.15)
    out["diagnostics"] = diag

    out["variant"] = variant or "default"
    suffix = f"__{variant}" if variant else ""
    path = os.path.join(a.out_root, f"{a.dataset}__ljudge{suffix}.json")
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nwrote {path}")

    print("\n================ judge sanity check ================")
    for d in DIMS:
        h = diag[d]["histogram"]
        print(f"  {d:12s} " + "  ".join(f"{k}:{h[k]:5d}" for k in sorted(h)) +
              f"   modal {diag[d]['modal_fraction']*100:.0f}%")
    print(f"  spread between best and worst system (overall): {spread:.3f} of 5")
    if diag["degenerate"]:
        print("\n  *** THESE SCORES ARE NOT USABLE ***")
        print("  The judge is putting almost everything on one value, or it separates the")
        print("  systems by less than 0.15 of a point. Either way it is not measuring")
        print("  quality. Do not put these numbers in the paper. Check the raw replies")
        print("  printed above, then change the prompt or use a stronger judge model.")
    else:
        print("\n  Looks usable: the scale is being used and the systems are separated.")
    print("====================================================")


if __name__ == "__main__":
    main()

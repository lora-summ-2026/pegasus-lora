"""LLM-as-judge Likert evaluation (after Gao et al., 2023; SummEval dimensions).

Scores every system on the SAME random sample of test documents (default 200) on
consistency, relevance, fluency and coherence (1-5), temperature 0.
Needs:  pip install anthropic   and   export ANTHROPIC_API_KEY=...

Usage
  python llm_judge.py --dataset samsum
  python llm_judge.py --dataset cnn_dm --n 200 --model claude-sonnet-5
Output
  runs/<ds>__judge_cache.jsonl   one line per (doc, system) - makes the script resumable
  runs/<ds>__judge.json          mean score per system and dimension
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
SYSTEMS = {"frozen": dict(method="frozen"), "lora4": dict(method="lora", rank=4),
           "lora8": dict(method="lora", rank=8), "lora16": dict(method="lora", rank=16),
           "full": dict(method="full")}

PROMPT = """You will be given a source document and one summary written for it.
Rate the summary on four dimensions, each on an integer scale from 1 (worst) to 5 (best):

- consistency: the factual alignment between the summary and the source. A consistent summary
  contains only statements that are entailed by the source; penalize hallucinated facts.
- relevance: selection of important content from the source; the summary should include only
  important information and no redundancies or excess.
- fluency: the quality of individual sentences - grammar, spelling, punctuation, word choice.
- coherence: the collective quality of all sentences; the summary should be well-structured and
  well-organized, building a coherent body of information about the topic.

SOURCE DOCUMENT:
{source}

SUMMARY:
{summary}

Answer with a single JSON object and nothing else, e.g.
{{"consistency": 4, "relevance": 3, "fluency": 5, "coherence": 4}}"""


def ask(client, model, source, summary, retries=5):
    for k in range(retries):
        try:
            r = client.messages.create(model=model, max_tokens=100, temperature=0,
                                       messages=[{"role": "user",
                                                  "content": PROMPT.format(source=source, summary=summary)}])
            txt = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            obj = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
            return {d: int(obj[d]) for d in DIMS}
        except Exception as e:
            print(f"  retry {k + 1}: {e}")
            time.sleep(2 ** k)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--out_root", default="runs")
    a = ap.parse_args()
    import anthropic
    client = anthropic.Anthropic()

    preds = {}
    for sys_name, kw in SYSTEMS.items():
        p = os.path.join(a.out_root, run_name(a.dataset, **kw), "predictions.jsonl")
        rows = read_jsonl(p)
        if rows:
            preds[sys_name] = {r["id"]: r["prediction"] for r in rows}
        else:
            print(f"[missing] {sys_name}: {p}")
    common_ids = sorted(set.intersection(*[set(v) for v in preds.values()]))
    random.Random(EVAL_SUBSET_SEED).shuffle(common_ids)
    ids = sorted(common_ids[:a.n])
    sources = {ex["id"]: ex["source"] for ex in load_dataset_splits(a.dataset)["test"]
               if ex["id"] in set(ids)}

    cache_path = os.path.join(a.out_root, f"{a.dataset}__judge_cache.jsonl")
    cache = {(r["id"], r["system"]): r for r in read_jsonl(cache_path)}
    for n, i in enumerate(ids):
        for s in preds:
            if (i, s) in cache:
                continue
            sc = ask(client, a.model, sources[i], preds[s][i])
            if sc:
                row = {"id": i, "system": s, **sc}
                append_jsonl(cache_path, [row])
                cache[(i, s)] = row
        print(f"  judged {n + 1}/{len(ids)} documents", flush=True)

    out = {"model": a.model, "n_docs": len(ids), "systems": {}}
    for s in preds:
        rows = [cache[(i, s)] for i in ids if (i, s) in cache]
        m = {d: float(np.mean([r[d] for r in rows])) for d in DIMS}
        m["overall"] = float(np.mean([m[d] for d in DIMS]))
        out["systems"][s] = m
    json.dump(out, open(os.path.join(a.out_root, f"{a.dataset}__judge.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

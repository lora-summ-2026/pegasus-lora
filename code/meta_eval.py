"""OPTIONAL human meta-evaluation: how well do ROUGE and the LLM judge agree with people?

Step 1 - export a blind annotation sheet (systems shuffled and hidden):
  python meta_eval.py export --dataset cnn_dm --n 50
  -> runs/cnn_dm__annotation_sheet.csv   (give to annotators; they fill the 4 score columns, 1-5)
  -> runs/cnn_dm__annotation_key.csv     (item -> system mapping; do NOT give to annotators)

Step 2 - after annotation (one or more filled copies; scores are averaged across annotators):
  python meta_eval.py score --dataset cnn_dm --sheets ann_A.csv ann_B.csv
  -> runs/cnn_dm__meta.json   Spearman rho per metric/dimension at three levels:
     sample  : per document, correlate across the system summaries, then average over documents
     system  : correlate the per-system means (only 5 points - interpret with care)
     dataset : correlate over all (document, system) pairs pooled
"""
import argparse
import csv
import json
import os
import random

import numpy as np
from scipy.stats import spearmanr

from common import EVAL_SUBSET_SEED, load_dataset_splits, read_jsonl, run_name
from llm_judge import DIMS, SYSTEMS


def export(a):
    runs = {s: {r["id"]: r["prediction"] for r in
                read_jsonl(os.path.join(a.out_root, run_name(a.dataset, **kw), "predictions.jsonl"))}
            for s, kw in SYSTEMS.items()}
    runs = {s: v for s, v in runs.items() if v}
    judged = {r["id"] for r in read_jsonl(os.path.join(a.out_root, f"{a.dataset}__judge_cache.jsonl"))}
    ids = sorted(set.intersection(*[set(v) for v in runs.values()]) & (judged or set.union(*[set(v) for v in runs.values()])))
    random.Random(EVAL_SUBSET_SEED + 1).shuffle(ids)
    ids = ids[:a.n]
    src = {ex["id"]: ex["source"] for ex in load_dataset_splits(a.dataset)["test"] if ex["id"] in set(ids)}
    items, rng = [], random.Random(7)
    for i in ids:
        systems = list(runs)
        rng.shuffle(systems)
        for s in systems:
            items.append((i, s))
    sheet = os.path.join(a.out_root, f"{a.dataset}__annotation_sheet.csv")
    key = os.path.join(a.out_root, f"{a.dataset}__annotation_key.csv")
    with open(sheet, "w", newline="", encoding="utf-8") as f, open(key, "w", newline="") as g:
        w, k = csv.writer(f), csv.writer(g)
        w.writerow(["item", "doc_id", "source", "summary"] + DIMS)
        k.writerow(["item", "doc_id", "system"])
        for n, (i, s) in enumerate(items):
            w.writerow([n, i, src[i], runs[s][i]] + [""] * len(DIMS))
            k.writerow([n, i, s])
    print(f"wrote {sheet} ({len(items)} items) and {key}")


def score(a):
    key = {int(r["item"]): (r["doc_id"], r["system"]) for r in
           csv.DictReader(open(os.path.join(a.out_root, f"{a.dataset}__annotation_key.csv")))}
    human = {}
    for path in a.sheets:
        for r in csv.DictReader(open(path, encoding="utf-8")):
            if all(r[d].strip() for d in DIMS):
                human.setdefault(key[int(r["item"])], []).append([float(r[d]) for d in DIMS])
    human = {k: np.mean(v, 0) for k, v in human.items()}

    auto = {}
    for s, kw in SYSTEMS.items():
        for r in read_jsonl(os.path.join(a.out_root, run_name(a.dataset, **kw), "per_doc.jsonl")):
            auto.setdefault((r["id"], s), {}).update(
                {"ROUGE-1": r["rouge1"], "ROUGE-2": r["rouge2"], "ROUGE-L": r["rougeLsum"]})
    for r in read_jsonl(os.path.join(a.out_root, f"{a.dataset}__judge_cache.jsonl")):
        auto.setdefault((r["id"], r["system"]), {})["LLM judge"] = {d: r[d] for d in DIMS}

    keys = [k for k in human if k in auto]
    metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "LLM judge"]
    val = lambda k, m, j: (auto[k][m][DIMS[j]] if m == "LLM judge" else auto[k][m])
    out = {"n_items": len(keys), "rho": {}}
    for m in metrics:
        if not all(m in auto[k] for k in keys):
            continue
        out["rho"][m] = {}
        for j, d in enumerate(DIMS):
            H = {k: human[k][j] for k in keys}
            M = {k: val(k, m, j) for k in keys}
            docs = sorted({k[0] for k in keys})
            sa = []
            for doc in docs:
                ks = [k for k in keys if k[0] == doc]
                if len(ks) > 2:
                    r = spearmanr([H[k] for k in ks], [M[k] for k in ks]).correlation
                    if not np.isnan(r):
                        sa.append(r)
            systems = sorted({k[1] for k in keys})
            sy = spearmanr([np.mean([H[k] for k in keys if k[1] == s]) for s in systems],
                           [np.mean([M[k] for k in keys if k[1] == s]) for s in systems]).correlation
            da = spearmanr([H[k] for k in keys], [M[k] for k in keys]).correlation
            out["rho"][m][d] = {"sa": float(np.mean(sa)) if sa else None,
                                "sy": float(sy), "da": float(da)}
    json.dump(out, open(os.path.join(a.out_root, f"{a.dataset}__meta.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["export", "score"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--sheets", nargs="*", default=[])
    ap.add_argument("--out_root", default="runs")
    a = ap.parse_args()
    export(a) if a.mode == "export" else score(a)

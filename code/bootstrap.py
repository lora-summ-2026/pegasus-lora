"""Paired bootstrap significance test (Koehn, 2004) + TOST-style equivalence check.

Usage
  python bootstrap.py --dataset cnn_dm            # all standard comparisons, writes runs/<ds>__bootstrap.json
  python bootstrap.py --a runs/X/per_doc.jsonl --b runs/Y/per_doc.jsonl   # any single pair

Difference is always B - A, computed on the SAME resampled document indices (paired).
p-value: two-sided, 2 * min(P(diff<=0), P(diff>=0)) over the bootstrap distribution.
Equivalence: the (1-2*alpha) = 90% CI of the difference lies inside [-delta, +delta]
(delta is pre-registered; default 1.0 ROUGE-1 point).
"""
import argparse
import json
import os

import numpy as np

from common import read_jsonl, run_name


def load(path, metric):
    return {r["id"]: r[metric] for r in read_jsonl(path)}


def paired(a_scores, b_scores, n=1000, seed=0, delta=1.0):
    ids = sorted(set(a_scores) & set(b_scores))
    A = np.array([a_scores[i] for i in ids])
    B = np.array([b_scores[i] for i in ids])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(n, len(ids)))
    ma, mb = A[idx].mean(1), B[idx].mean(1)
    d = mb - ma
    ci = lambda x, lo, hi: [float(np.percentile(x, lo)), float(np.percentile(x, hi))]
    p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    ci90 = ci(d, 5, 95)
    return dict(n_docs=len(ids), n_resamples=n,
                mean_a=float(A.mean()), ci_a=ci(ma, 2.5, 97.5),
                mean_b=float(B.mean()), ci_b=ci(mb, 2.5, 97.5),
                diff=float(B.mean() - A.mean()), ci_diff=ci(d, 2.5, 97.5), p=p,
                delta=delta, ci90_diff=ci90,
                equivalent=bool(ci90[0] > -delta and ci90[1] < delta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--metric", default="rouge1")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--out_root", default="runs")
    a = ap.parse_args()

    if a.a and a.b:
        print(json.dumps(paired(load(a.a, a.metric), load(a.b, a.metric), a.n, 0, a.delta), indent=2))
        return

    ds = a.dataset
    path = lambda n: os.path.join(a.out_root, n, "per_doc.jsonl")
    ref = path(run_name(ds, "lora", 8))
    res = {"metric": a.metric}
    for key, other in [("frozen_vs_lora8", run_name(ds, "frozen")),
                       ("full_vs_lora8", run_name(ds, "full"))]:
        if os.path.exists(ref) and os.path.exists(path(other)):
            # A = LoRA-8, B = other system  ->  diff = other - LoRA-8 (as in the paper's table)
            res[key] = paired(load(ref, a.metric), load(path(other), a.metric), a.n, 0, a.delta)
        else:
            print(f"[missing] {key}: need {ref} and {path(other)}")
    out = os.path.join(a.out_root, f"{ds}__bootstrap.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

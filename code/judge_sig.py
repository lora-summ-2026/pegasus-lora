"""Paired bootstrap significance on the LLM-judge scores, and gap closure.

The judge scores every system on the SAME documents, so the comparison is paired and a
paired bootstrap (Koehn, 2004) is far more powerful than comparing the two standard
errors - the document-to-document variation cancels.

It also reports, for each dimension, how much of the zero-shot -> full fine-tuning gap
LoRA recovers, which is the quantity the paper reports for ROUGE. The two numbers are
directly comparable, and where they disagree that disagreement is the finding.

Usage
  python judge_sig.py --dataset samsum
Output
  runs/<ds>__ljudge_sig.json
"""
import argparse
import json
import os

import numpy as np

from common import read_jsonl

DIMS = ["consistency", "relevance", "fluency", "coherence"]
REF = "lora8"


def paired(A, B, n=1000, seed=0):
    """B - A on the same resampled document indices."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(A), size=(n, len(A)))
    d = B[idx].mean(1) - A[idx].mean(1)
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return dict(mean_a=float(A.mean()), mean_b=float(B.mean()),
                diff=float(B.mean() - A.mean()), ci95=[lo, hi], p=p,
                significant=bool(lo > 0 or hi < 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out_root", default="runs")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--variant", default="",
                    help="training-budget tag to analyse, e.g. ep10; empty = the default models")
    a = ap.parse_args()

    cache_path = os.path.join(a.out_root, f"{a.dataset}__ljudge_cache.jsonl")
    suffix = f"__{a.variant}" if a.variant else ""
    meta_path = os.path.join(a.out_root, f"{a.dataset}__ljudge{suffix}.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    pv = None
    rows = read_jsonl(cache_path)
    # use only the newest prompt version present, never a mix of two instruments
    versions = {r.get("pv") for r in rows}
    if versions:
        pv = max(v for v in versions if v is not None) if any(v is not None for v in versions) else None
    # Rows for a different training budget describe different models entirely, so mixing
    # them would silently overwrite one budget's scores with the other's.
    rows = [r for r in rows if r.get("pv") == pv and (r.get("variant") or "") == a.variant]
    if not rows:
        raise SystemExit(f"no judge rows in {cache_path}")

    by_sys = {}
    for r in rows:
        by_sys.setdefault(r["system"], {})[r["id"]] = r
    systems = sorted(by_sys)
    if REF not in systems:
        raise SystemExit(f"{REF} not among {systems}")

    # documents every system scored, so all comparisons use one identical set
    ids = sorted(set.intersection(*[set(v) for v in by_sys.values()]))
    print(f"[judge_sig] prompt version {pv}, variant {a.variant or 'default'}, "
          f"{len(ids)} documents scored by all of {systems}")

    def vec(s, dim):
        if dim == "overall":
            return np.array([np.mean([by_sys[s][i][d] for d in DIMS]) for i in ids], dtype=float)
        return np.array([by_sys[s][i][dim] for i in ids], dtype=float)

    out = {"dataset": a.dataset, "variant": a.variant or "default", "judge": meta.get("judge"),
           "judge_revision": meta.get("judge_revision"),
           "prompt_version": pv, "n_docs": len(ids), "comparisons": {}, "gap_closure": {}}

    for dim in DIMS + ["overall"]:
        ref = vec(REF, dim)
        out["comparisons"][dim] = {}
        for s in systems:
            if s == REF:
                continue
            out["comparisons"][dim][f"{s}_vs_{REF}"] = paired(ref, vec(s, dim), a.n)
        if "frozen" in systems and "full" in systems:
            fr, fu = vec("frozen", dim).mean(), vec("full", dim).mean()
            span = fu - fr
            out["gap_closure"][dim] = {
                "frozen": float(fr), "full": float(fu), "span": float(span),
                **{s: (float((vec(s, dim).mean() - fr) / span) if span else None)
                   for s in systems if s.startswith("lora")}}

    path = os.path.join(a.out_root, f"{a.dataset}__ljudge_sig{suffix}.json")
    json.dump(out, open(path, "w"), indent=2)

    # ---- readable summary --------------------------------------------------
    print(f"\npaired bootstrap against {REF}, {len(ids)} documents, {a.n} resamples")
    print(f"{'dimension':12s} {'system':8s} {'diff':>7s} {'95% CI':>18s} {'p':>7s}  verdict")
    print("-" * 68)
    for dim in ["overall"] + DIMS:
        for k, v in out["comparisons"][dim].items():
            s = k.split("_vs_")[0]
            ci = f"[{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]"
            verdict = "differs" if v["significant"] else "tie"
            print(f"{dim:12s} {s:8s} {v['diff']:+7.3f} {ci:>18s} {v['p']:7.3f}  {verdict}")
        print()

    print("gap closure - fraction of the zero-shot -> full fine-tuning gap that LoRA recovers")
    for dim in ["overall"] + DIMS:
        g = out["gap_closure"].get(dim)
        if not g:
            continue
        parts = "  ".join(f"{s}:{g[s]*100:5.1f}%" for s in sorted(g) if s.startswith("lora"))
        print(f"  {dim:12s} span {g['span']:.2f}   {parts}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

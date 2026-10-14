"""How much does the LLM judge actually disagree with ROUGE?

The paper argues that ROUGE overstates how close LoRA gets to full fine-tuning. That
argument is only worth making if the judge is measuring something ROUGE does not already
capture - so this quantifies the overlap directly, document by document, on data that is
already on disk. No GPU, no annotation, a few seconds.

Reports Spearman's rho between the judge's score and ROUGE at two levels:
  document : over all (document, system) pairs - does the judge rank individual summaries
             the way ROUGE does?
  system   : over the per-system means (only a handful of points; read with care) - do the
             two metrics at least order the systems the same way?

A low document-level correlation with a high system-level one is the interesting case: the
two metrics agree on which model is better while disagreeing on which summaries are good,
which is exactly why the size of a gap can differ between them.

Usage
  python judge_vs_rouge.py --dataset samsum
Output
  runs/<ds>__judge_vs_rouge.json
"""
import argparse
import json
import os

import numpy as np

from common import read_jsonl, run_name
from seed_stats import t_two_sided_p

DIMS = ["consistency", "relevance", "fluency", "coherence"]
ROUGES = ["rouge1", "rouge2", "rougeLsum"]
SYSTEMS = {
    "frozen": dict(method="frozen"),
    "lora4":  dict(method="lora", rank=4),
    "lora8":  dict(method="lora", rank=8),
    "lora16": dict(method="lora", rank=16),
    "full":   dict(method="full"),
}


def rankdata(x):
    """Ranks with ties averaged - the same convention scipy uses."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # average the ranks of tied values
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def spearman(a, b):
    """Spearman's rho and its two-sided p-value, without scipy."""
    n = len(a)
    if n < 3:
        return None, None
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    if denom == 0:
        return None, None
    rho = float((ra * rb).sum() / denom)
    if abs(rho) >= 1.0:
        return rho, 0.0
    t = rho * np.sqrt((n - 2) / (1 - rho ** 2))
    return rho, t_two_sided_p(float(t), n - 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out_root", default="runs")
    ap.add_argument("--epochs", type=float, default=None,
                    help="analyse the models trained with this budget (default: the default budget)")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    # The cache holds one row per (document, system, TRAINING BUDGET). Without filtering
    # on the budget the dict below silently keeps whichever row was written last, which
    # would correlate one budget's judge scores against another budget's ROUGE.
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

    rows = read_jsonl(os.path.join(a.out_root, f"{a.dataset}__ljudge_cache.jsonl"))
    pvs = {r.get("pv") for r in rows if r.get("pv") is not None}
    pv = max(pvs) if pvs else None
    rows = [r for r in rows if r.get("pv") == pv and (r.get("variant") or "") == variant]
    if not rows:
        raise SystemExit(f"no judge rows for {a.dataset} at variant "
                         f"{variant or 'default'} - run local_judge.py first")
    judge = {(r["id"], r["system"]): r for r in rows}

    # ---- ROUGE per document -------------------------------------------------
    rouge = {}
    for s, kw in SYSTEMS.items():
        p = os.path.join(a.out_root, run_name(a.dataset, **kw, **extra), "per_doc.jsonl")
        for r in read_jsonl(p):
            rouge[(r["id"], s)] = r

    keys = sorted(set(judge) & set(rouge))
    if len(keys) < 3:
        raise SystemExit("not enough overlapping (document, system) pairs")
    systems = sorted({k[1] for k in keys})
    print(f"[judge_vs_rouge] {a.dataset}: {len(keys)} (document, system) pairs, "
          f"prompt version {pv}, budget {variant or 'default'}, systems {systems}")

    def jscore(k, dim):
        return np.mean([judge[k][d] for d in DIMS]) if dim == "overall" else judge[k][dim]

    out = {"dataset": a.dataset, "prompt_version": pv, "variant": variant or "default",
           "n_pairs": len(keys),
           "systems": systems, "document_level": {}, "system_level": {}}

    print(f"\ndocument level - all {len(keys)} (document, system) pairs pooled")
    print(f"  {'judge':12s} " + "".join(f"{r:>14s}" for r in ROUGES))
    for dim in ["overall"] + DIMS:
        jv = [jscore(k, dim) for k in keys]
        cells, rec = [], {}
        for rg in ROUGES:
            rho, p = spearman(jv, [rouge[k][rg] for k in keys])
            rec[rg] = {"rho": rho, "p": p}
            cells.append(f"{rho:>8.3f}{'*' if p is not None and p < 0.05 else ' '}     ")
        out["document_level"][dim] = rec
        print(f"  {dim:12s} " + "".join(cells))

    print(f"\nsystem level - {len(systems)} points, one per system")
    print(f"  {'judge':12s} " + "".join(f"{r:>14s}" for r in ROUGES))
    for dim in ["overall"] + DIMS:
        jm = [np.mean([jscore(k, dim) for k in keys if k[1] == s]) for s in systems]
        cells, rec = [], {}
        for rg in ROUGES:
            rm = [np.mean([rouge[k][rg] for k in keys if k[1] == s]) for s in systems]
            rho, p = spearman(jm, rm)
            rec[rg] = {"rho": rho, "p": p}
            cells.append(f"{rho:>8.3f}      " if rho is not None else f"{'-':>8s}      ")
        out["system_level"][dim] = rec
        print(f"  {dim:12s} " + "".join(cells))

    ov = out["document_level"]["overall"]["rouge1"]["rho"]
    print(f"\n  * marks p < 0.05, at the document level only.")
    print(f"  Document-level rho for overall vs ROUGE-1: {ov:.3f}"
          f"  ->  {ov*ov*100:.0f}% shared variance,"
          f" so {100-ov*ov*100:.0f}% of what the judge measures is not ROUGE.")
    print("  Read the two levels together: agreement on the ORDER of systems alongside modest\n"
          "  agreement on individual summaries is what lets the two metrics report the same\n"
          "  ranking but a different size of gap.")
    print("  The system level has only a handful of points; an exact permutation test needs\n"
          "  rho = 1.0 for p < 0.05 with 5 systems, so treat those numbers as descriptive.")

    suffix = f"__{variant}" if variant else ""
    path = os.path.join(a.out_root, f"{a.dataset}__judge_vs_rouge{suffix}.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

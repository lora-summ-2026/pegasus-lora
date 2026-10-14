"""Aggregate repeated runs of the same configuration across random seeds.

Answers the question a reviewer will ask first: are the differences this paper reports
larger than the run-to-run noise of the training procedure itself?

For every configuration that was trained with more than one seed it reports the mean,
the sample standard deviation and the range of each ROUGE metric, and it flags the
paper's ordering claims whose margin is smaller than the seed spread.

Usage
  python seed_stats.py --runs runs --dataset samsum
  python seed_stats.py --runs runs                 # every dataset found
Output
  runs/seed_stats.json
"""
import argparse
import json
import re
import math
import os
from collections import defaultdict

import numpy as np

METRICS = ["rouge1", "rouge2", "rougeLsum"]


def load_runs(root):
    """Every metrics.json under root, keyed by (dataset, config-without-seed)."""
    out = defaultdict(list)
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name, "metrics.json")
        if not os.path.isfile(p):
            continue
        try:
            m = json.load(open(p))
        except json.JSONDecodeError:
            print(f"[skip] unreadable {p}")
            continue
        cfg = m.get("config", {})
        ds = cfg.get("dataset")
        seed = cfg.get("seed")
        if ds is None or seed is None:
            print(f"[skip] {name}: metrics.json predates seed tracking")
            continue
        # Remove ONLY the seed token. The training budget (__ep.., __tr..) is appended
        # after the seed, so cutting the name at "__seed" would drop the budget too and
        # pool models trained for different numbers of epochs into one configuration.
        key = (ds, re.sub(r"__seed\d+", "", name))
        out[key].append({"seed": seed, "run": name,
                         "epochs": cfg.get("epochs"),
                         "max_train_samples": cfg.get("max_train_samples"),
                         **m["rouge"]})
    return out


DEFAULT_CFG = "lora_r8_a16_qv_both"     # the configuration the paper recommends


def _betacf(a, b, x, itmax=200, eps=3e-16):
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    front = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t, df):
    """Two-sided p-value of Student's t. No scipy: the paper's numbers should not
    depend on a package that is not otherwise used anywhere in this pipeline."""
    if df <= 0:
        return None
    return _betai(0.5 * df, 0.5, df / (df + t * t))


def welch(m_a, s_a, n_a, m_b, s_b, n_b):
    """Welch's t-test from summary statistics. Returns (diff, se, t, p, df)."""
    d = m_b - m_a
    if s_a is None or s_b is None or n_a < 2 or n_b < 2:
        return d, None, None, None, None
    va, vb = s_a ** 2 / n_a, s_b ** 2 / n_b
    se = (va + vb) ** 0.5
    if se == 0:
        return d, se, None, None, None
    t = d / se
    df = (va + vb) ** 2 / (va ** 2 / (n_a - 1) + vb ** 2 / (n_b - 1))
    return d, se, t, t_two_sided_p(t, df), df


def pairwise(report, metric="rouge1"):
    results = {}
    """Compare every multi-seed configuration against the default, per dataset."""
    by_ds = {}
    for key, e in report.items():
        ds, cfg = key.split("::", 1)
        by_ds.setdefault(ds, {})[cfg] = e

    for ds, cfgs in sorted(by_ds.items()):
      # Compare within a training budget, never across one: a 10-epoch model and a
      # 5-epoch model are different experiments, and the budget is its own axis.
      budgets = sorted({"".join(re.findall(r"__(?:ep[0-9.]+|tr\d+)", k)) for k in cfgs})
      for budget in budgets:
        ref_key = f"{ds}__{DEFAULT_CFG}{budget}"
        ref = cfgs.get(ref_key)
        if not ref or ref["n"] < 2:
            continue
        others = {k: v for k, v in cfgs.items()
                  if k != ref_key and v["n"] >= 2
                  and "".join(re.findall(r"__(?:ep[0-9.]+|tr\d+)", k)) == budget}
        if not others:
            continue
        label = f"{DEFAULT_CFG}{budget}" if budget else DEFAULT_CFG
        print(f"\n{ds}{' [' + budget.lstrip('_') + ']' if budget else ''}: "
              f"each configuration vs the default ({label}), Welch's t on {metric}")
        print(f"  {'configuration':44s} {'diff':>6s} {'se':>6s} {'p':>7s}  verdict")
        print("  " + "-" * 72)
        rm, rs, rn = ref[metric]["mean"], ref[metric]["std"], ref["n"]
        for k, e in sorted(others.items()):
            d, se, t, p, df = welch(rm, rs, rn, e[metric]["mean"], e[metric]["std"], e["n"])
            if p is None:
                print(f"  {k[:44]:44s} {d:+6.2f} {'-':>6s} {'-':>7s}  need 2+ seeds on both")
                continue
            v = "REAL (beyond seed noise)" if p < 0.05 else "tie - report as no difference"
            results[f"{ds}::{k}"] = {"vs": ref_key, "diff": d, "se": se, "t": t, "df": df,
                                     "p": p, "significant": bool(p < 0.05)}
            print(f"  {k[:44]:44s} {d:+6.2f} {se:6.3f} {p:7.3f}  {v}")
        print("\n  Note: with 3 seeds the spread of each configuration is itself only roughly\n"
              "  estimated, so treat a p just under 0.05 as suggestive rather than settled.")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--dataset", default=None)
    a = ap.parse_args()

    groups = load_runs(a.runs)
    report = {}
    for (ds, cfg), rows in sorted(groups.items()):
        if a.dataset and ds != a.dataset:
            continue
        seeds = sorted(r["seed"] for r in rows)
        if len(set(seeds)) != len(seeds):
            print(f"[warn] {cfg}: duplicate seeds {seeds} - two runs share a seed")
        for field in ("epochs", "max_train_samples"):
            vals = {r.get(field) for r in rows if r.get(field) is not None}
            if len(vals) > 1:
                print(f"[ERROR] {cfg}: runs with different {field} {sorted(vals)} were grouped "
                      f"together. Refusing to average them.")
                continue
        entry = {"seeds": seeds, "n": len(rows)}
        for k in METRICS:
            v = np.array([r[k] for r in rows], dtype=float)
            entry[k] = {
                "mean": float(v.mean()),
                "std": float(v.std(ddof=1)) if len(v) > 1 else None,
                "min": float(v.min()), "max": float(v.max()),
                "range": float(v.max() - v.min()),
            }
        report[f"{ds}::{cfg}"] = entry

    # ---- readable summary --------------------------------------------------
    print(f"{'configuration':52s} {'n':>2s} {'R-1 mean':>9s} {'sd':>6s} {'range':>7s}")
    print("-" * 80)
    multi = []
    for k, e in report.items():
        r1 = e["rouge1"]
        sd = f"{r1['std']:.2f}" if r1["std"] is not None else "  -  "
        print(f"{k[:52]:52s} {e['n']:2d} {r1['mean']:9.2f} {sd:>6s} {r1['range']:7.2f}")
        if e["n"] > 1:
            multi.append((k, r1["std"], r1["range"]))

    if multi:
        worst = max(m[2] for m in multi)
        print(f"\nLargest ROUGE-1 spread across seeds: {worst:.2f} points.")
        print("That is only an alarm bell, NOT the test: it is the spread of the single "
              "noisiest\nconfiguration, so using it against every comparison punishes pairs "
              "that are both\nstable. The pairwise tests below are what decides each claim.")
    else:
        print("\nNo configuration has more than one seed yet - nothing to compare.")

    # ---- pairwise tests against the default configuration ------------------
    # Welch's t-test: it does not assume the two configurations are equally noisy,
    # and here they are very much not (the default is far more stable than alpha=32).
    comparisons = pairwise(report)

    path = os.path.join(a.runs, "seed_stats.json")
    json.dump({"configurations": report, "comparisons": comparisons},
              open(path, "w"), indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

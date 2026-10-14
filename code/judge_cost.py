"""Measure what the LLM-based evaluation actually cost, from SLURM's own accounting.

Why not read it from the judge's own output: local_judge.py records the wall time of the
invocation that produced the scores, but a re-run finds every summary already cached and
finishes in seconds. That number is the cost of replaying a cache, not the cost of the
evaluation, and putting it in the paper would understate the figure by two orders of
magnitude. SLURM's accounting is the honest record: it timed the jobs that did the work.

Usage
  python judge_cost.py                       # find judge jobs by name and sum them
  python judge_cost.py --jobs 123,124,125    # exactly these jobs
  python judge_cost.py --minutes 25.5        # set it by hand, if sacct is unavailable
Output
  runs/judge_cost.json, which make_latex.py turns into the numbers in Section 4.5.
"""
import argparse
import json
import os
import re
import subprocess
import sys


def elapsed_to_minutes(s):
    """SLURM Elapsed is [DD-]HH:MM:SS."""
    s = s.strip()
    if not s or s in ("", "INVALID"):
        return None
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [float(p) for p in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, sec = parts
    return days * 1440 + h * 60 + m + sec / 60.0


def sacct(jobs=None, pattern="judge"):
    """Every finished job whose name matches, as (jobid, name, elapsed_minutes, state)."""
    cmd = ["sacct", "-X", "-P", "-n", "-o", "JobID,JobName,Elapsed,State"]
    if jobs:
        cmd += ["-j", ",".join(jobs)]
    else:
        cmd += ["-u", os.environ.get("USER", ""), "-S", "2020-01-01"]
    try:
        raw = subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        sys.exit(f"could not run sacct ({e}). Re-run with --minutes <n> to set the figure "
                 "by hand, reading it from squeue/sacct yourself.")
    out = []
    for line in raw.splitlines():
        f = line.split("|")
        if len(f) < 4:
            continue
        jid, name, el, state = f[0], f[1], f[2], f[3]
        if jobs is None and pattern.lower() not in name.lower():
            continue
        mins = elapsed_to_minutes(el)
        if mins is None:
            continue
        out.append((jid, name, mins, state))
    return out


def coverage(runs):
    """How much judging actually happened, read off the judge's own output files.

    The point of this is to catch the failure that a summed job list cannot see: a judge
    run whose SLURM job was named something the pattern misses is silently left out, and
    the total then understates the cost of the evaluation. Counting the judgements that
    exist on disk gives an independent expectation to compare the job list against.
    """
    total = 0
    found = []
    if not os.path.isdir(runs):
        return 0, found
    for fn in sorted(os.listdir(runs)):
        m = re.match(r"(.+?)__ljudge(?:__(.+))?\.json$", fn)
        if not m:
            continue
        try:
            j = json.load(open(os.path.join(runs, fn)))
        except (json.JSONDecodeError, OSError):
            continue
        n = sum(int(e.get("n", 0)) for e in (j.get("systems") or {}).values())
        total += n
        found.append((fn, len(j.get("systems") or {}), n))
    return total, found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.expanduser("~/pegasus_lora/runs"))
    ap.add_argument("--jobs", default=None, help="comma-separated job ids")
    ap.add_argument("--pattern", default="judge", help="substring of the job name")
    ap.add_argument("--minutes", type=float, default=None,
                    help="set the total by hand and skip sacct entirely")
    a = ap.parse_args()

    if a.minutes is not None:
        rows, total = [], a.minutes
        print(f"total set by hand: {total:.1f} GPU-minutes")
    else:
        jobs = [j.strip() for j in a.jobs.split(",")] if a.jobs else None
        rows = sacct(jobs, a.pattern)
        ok = [r for r in rows if r[3].startswith("COMPLETED")]
        bad = [r for r in rows if not r[3].startswith("COMPLETED")]
        if not ok:
            sys.exit(f"no COMPLETED job matched {a.pattern!r}. Found: "
                     + (", ".join(f"{r[1]}({r[3]})" for r in rows) or "nothing")
                     + ". Re-run with --jobs or --minutes.")
        print(f"{'job':>10s}  {'name':28s} {'minutes':>8s}  state")
        for jid, name, mins, state in ok:
            print(f"{jid:>10s}  {name[:28]:28s} {mins:8.1f}  {state}")
        for jid, name, mins, state in bad:
            print(f"{jid:>10s}  {name[:28]:28s} {mins:8.1f}  {state}  (excluded: not COMPLETED)")
        total = sum(r[2] for r in ok)
        print(f"\n{len(ok)} judge jobs, {total:.1f} GPU-minutes in total")

    # ---- does the job list actually account for all the judging that happened? ----
    njudge, files = coverage(a.runs)
    complete = True
    if njudge:
        print(f"\njudgements on disk: {njudge} across {len(files)} judge output files")
        for fn, nsys, n in files:
            print(f"  {fn:44s} {nsys} systems, {n:5d} judgements")
        rate = njudge / total if total > 0 else 0
        print(f"\n{total:.1f} GPU-minutes for {njudge} judgements "
              f"= {rate:.0f} judgements per minute")
        # Two independent signals, because either alone can be fooled. A local 7B judge on
        # one consumer GPU scores on the order of 50-150 summaries a minute, so a much
        # higher apparent rate means the work was done by jobs this list does not contain.
        # And every output file had to be produced by at least one job, so more files than
        # jobs means jobs are missing (a re-run gives more jobs than files, never fewer).
        njobs = len(ok) if a.minutes is None else 0
        too_fast = rate > 160
        too_few_jobs = njobs and len(files) > njobs
        if too_fast or too_few_jobs:
            complete = False
            if too_few_jobs:
                print(f"\n{len(files)} judge output files but only {njobs} judge jobs found.")
            if too_fast:
                print(f"\n{rate:.0f} judgements per minute is not physically plausible for a "
                      "local 7B judge on one GPU.")
            print(f"""
WARNING: the job list is INCOMPLETE - judge runs were submitted under a job name that
{a.pattern!r} does not match, and the total above understates the cost of the evaluation.

List every job and find the missing ones:

    sacct -X -o JobID,JobName%30,Elapsed,State,AllocTRES%40 | less -S

then re-run with the full set, for example:

    python judge_cost.py --jobs 21729525,21732755,21732756,21732880,<the missing ids>

Section 4.5 of the paper reports this number, so it has to be the whole of it.""")

    os.makedirs(a.runs, exist_ok=True)
    path = os.path.join(a.runs, "judge_cost.json")
    json.dump({"total_gpu_minutes": round(total, 1), "usd": 0.0,
               "source": "sacct" if a.minutes is None else "manual",
               "judgements_on_disk": njudge,
               "judgements_per_minute": round(njudge / total, 1) if total else None,
               "coverage_plausible": complete,
               "jobs": [{"jobid": r[0], "name": r[1], "minutes": round(r[2], 1),
                         "state": r[3]} for r in rows],
               "note": "open-weights judge on one local GPU, so the monetary cost is zero"},
              open(path, "w"), indent=2)
    print(f"wrote {path}")
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())

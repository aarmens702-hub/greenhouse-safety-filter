"""Main comparison and model mismatch (paper Table 1, Figure 3, mismatch results): the AGC-tuned rules raw and
filtered, and the six PPO-Lagrangian agents filtered, on the 6 original seasons, default parameters and mismatch.
The agents' unfiltered rows come from controllers/run_ppolag.py.

Usage: python experiments/main_comparison.py WORKERS [--out PATH]   (default outputs/stage_real.jsonl)
96 seasons: 24 unfiltered (seconds each), 72 filtered (1-20 min each on one core). Resumes.
Needs the trained agents in models/ (see models/README.md). Frozen output: results/filter/stage_real.jsonl.
Was greenleaf-filter/stage_real.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jobqueue import ROOT, read_rows, run_queue

S6 = [("Amsterdam", 2010, 45), ("Amsterdam", 2010, 59), ("Amsterdam", 2010, 75), ("Wag", 2020, 5), ("Wag", 2020, 45), ("Wag", 2020, 85)]
ARMS = [("rb_tuned_agc2020", 0, 0), ("rb_tuned_agc2020", 85, 0), ("ppolag:d0.01_s0_nl16", 85, 0), ("ppolag:d0.01_s0_agc25", 85, 0)]
ARMS += [(f"ppolag:d0.01_s{s}_{p}", 85, 0) for s in (1, 2) for p in ("agc25", "nl16")]
ARMS += [("rb_tuned_agc2020", 0, 1), ("rb_tuned_agc2020", 85, 1), ("ppolag:d0.01_s0_agc25", 85, 1), ("ppolag:d0.01_s0_nl16", 85, 1)]
ARMS += [(f"ppolag:d0.01_s{s}_{p}", 85, 1) for s in (1, 2) for p in ("agc25", "nl16")]

if __name__ == "__main__":
    OUT = sys.argv[sys.argv.index("--out")+1] if "--out" in sys.argv else str(ROOT/"outputs"/"stage_real.jsonl")
    done = {(r["ctrl"], r["limit"], r["loc"], r["start"], int(r["mismatch"])) for r in read_rows(OUT)}
    jobs = []
    for c, lim, mm in ARMS:
        for loc, yr, st in S6:
            if (c, float(lim), loc, st, mm) in done: continue
            x = (["--limit", str(lim), "--h", "4", "--obj", "mindev"] if lim else []) + (["--mismatch"] if mm else [])
            jobs.append(((c, lim, loc, st, mm), ["--ctrl", c, "--loc", loc, "--year", str(yr), "--start", str(st), "--out", OUT]+x))
    run_queue(jobs, int(sys.argv[1]))
    print("REAL_DONE", flush=True)

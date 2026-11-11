"""Plants the filter does not know (exploratory stage 2b), rules only: the sim-tuned rules on Bie et al.'s
calibrated plant, on a plant with lamp cooling off (cool0) and on a low initial leaf area (lai08), unfiltered and
filtered with observer and sensor noise; plus the default plant with observer and noise, with and without
mismatch. Amsterdam 2010 day 59 and Wag 2020 day 45. The unfiltered Bie rows are Figure 4's "no filter" lines.

Usage: python experiments/calibration_baselines.py WORKERS [--out PATH]   (default outputs/stage2b.jsonl)
Frozen output: results/filter/stage2b.jsonl (which also holds rows for an earlier residual-RL agent, not in this
repository or the paper). Was greenleaf-filter/stage2b.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jobqueue import ROOT, read_rows, run_queue

S = [("Amsterdam", 2010, 59), ("Wag", 2020, 45)]; C = ["rb_tuned"]
J = []
for s in S:
    for c in C:
        for pl in ("bie", "cool0", "lai08"): J.append((c, s, pl, False, False, False, 0))      # unfiltered on each unknown plant
for s in S:
    for c in C:
        for pl in ("bie", "cool0", "lai08"): J.append((c, s, pl, False, True, True, 85))
        J.append((c, s, "none", True, True, True, 85)); J.append((c, s, "none", False, True, True, 85)); J.append((c, s, "none", False, True, False, 85))

if __name__ == "__main__":
    OUT = sys.argv[sys.argv.index("--out")+1] if "--out" in sys.argv else str(ROOT/"outputs"/"stage2b.jsonl")
    done = {(r["ctrl"], r["loc"], r["start"], r["plant"], r["mismatch"], r["observer"], r["noise"], r["limit"]) for r in read_rows(OUT)}
    jobs = []
    for c, s, pl, mm, ob, no, lim in J:
        if (c, s[0], s[2], pl, mm, ob, no, lim) in done: continue
        a = ["--ctrl", c, "--loc", s[0], "--year", str(s[1]), "--start", str(s[2]), "--plant", pl, "--out", OUT]
        if mm: a.append("--mismatch")
        if ob: a.append("--observer")
        if no: a.append("--noise")
        if lim: a += ["--limit", str(lim), "--obj", "mindev", "--h", "4"]
        jobs.append(((c, s, pl, mm, ob, no, lim), a))
    run_queue(jobs, int(sys.argv[1]))
    print("STAGE2B_DONE", flush=True)

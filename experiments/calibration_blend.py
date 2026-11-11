"""Calibration study, part (a) (paper Figure 4a, T2): the sim-tuned rules on Bie et al.'s calibrated plant, filtered
by a model moved 0, 25, 50, 75 or 100% of the way from the default parameters to Bie's. Arms:
a1 exact state, default model; a2 a1 + lower RH limit; a3 exact state, Bie model; a4 a3 + lower RH limit;
a5 observer + sensor noise, Bie model, lower RH limit; m25/m50/m75 as a5 with the model 25/50/75% toward Bie.
Amsterdam 2010 day 59 and Wag 2020 day 45.

Usage: python experiments/calibration_blend.py WORKERS [ARM ...] [--out PATH]   (default: all arms; outputs/bie.jsonl)
16 seasons, 7-30 min each. Resumes. Frozen output: results/filter/bie.jsonl (which also holds rows for an
earlier residual-RL agent, not in this repository or the paper). Was greenleaf-filter/stage_bie.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jobqueue import ROOT, read_rows, run_queue

ARMS = {"a1": [], "a2": ["--lowrh"], "a3": ["--model-frac", "1"], "a4": ["--model-frac", "1", "--lowrh"], "a5": ["--model-frac", "1", "--lowrh", "--observer", "--noise"]}
for m in ("m25", "m50", "m75"):
    ARMS[m] = ["--model-frac", str(int(m[1:])/100), "--lowrh", "--observer", "--noise"]

if __name__ == "__main__":
    av = sys.argv[1:]
    OUT = av[av.index("--out")+1] if "--out" in av else str(ROOT/"outputs"/"bie.jsonl")
    if "--out" in av: i = av.index("--out"); del av[i:i+2]
    workers, pick = int(av[0]), av[1:]
    base = ["--plant", "bie", "--limit", "85", "--h", "4", "--obj", "mindev", "--out", OUT]
    jobs = []
    for arm, extra in ARMS.items():
        if pick and arm not in pick: continue
        for loc, yr, st in (("Amsterdam", 2010, 59), ("Wag", 2020, 45)):
            jobs.append((("rb_tuned", loc, st, arm), ["--ctrl", "rb_tuned", "--loc", loc, "--year", str(yr), "--start", str(st)]+base+extra))
    def key(r):
        return (r["ctrl"], r["loc"], r["start"], r["model_frac"], r["lowrh"], r.get("observer"))
    def jk(j):
        k, a = j; f = float(a[a.index("--model-frac")+1]) if "--model-frac" in a else 0.
        return (k[0], k[1], k[2], f, "--lowrh" in a, "--observer" in a)
    done = {key(r) for r in read_rows(OUT)}
    jobs = [j for j in jobs if jk(j) not in done]
    run_queue(jobs, workers, runner="safety_filter.run_season_calib")
    print("BIE_DONE", flush=True)

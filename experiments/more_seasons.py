"""Six new seasons, pre-registered 2026-11-06 (docs/preregistration_filter.md, "More seasons"; paper T6):
Amsterdam 2010 starts 15, 30, 90, 105 and Wag 2020 starts 25, 65; AGC-tuned rules raw and filtered, and the
six PPO-Lagrangian agents filtered, nominal plant.

Usage: python experiments/more_seasons.py WORKERS [--out PATH]   (default outputs/stage_more.jsonl)
48 seasons, 42 of them filtered (2-15 min each). Resumes; a run over 45 min of awake time is killed and
logged to <out>.timeouts.jsonl. Frozen output: results/filter/stage_more.jsonl. Was greenleaf-filter/stage_more.py."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jobqueue import ROOT, read_rows, run_queue

S6 = [("Amsterdam", 2010, 15), ("Amsterdam", 2010, 30), ("Amsterdam", 2010, 90), ("Amsterdam", 2010, 105), ("Wag", 2020, 25), ("Wag", 2020, 65)]
ARMS = [("rb_tuned_agc2020", 0, 0), ("rb_tuned_agc2020", 85, 0)] + [(f"ppolag:d0.01_s{s}_{p}", 85, 0) for p in ("agc25", "nl16") for s in (0, 1, 2)]
LIMIT_S = 45*60

if __name__ == "__main__":
    OUT = sys.argv[sys.argv.index("--out")+1] if "--out" in sys.argv else str(ROOT/"outputs"/"stage_more.jsonl")
    TO = OUT.removesuffix(".jsonl") + ".timeouts.jsonl"
    done = {(r["ctrl"], r["limit"], r["loc"], r["start"], int(r["mismatch"])) for r in read_rows(OUT)}
    done |= {tuple(r["key"]) for r in read_rows(TO)}
    jobs = []
    for c, lim, mm in ARMS:
        for loc, yr, st in S6:
            if (c, float(lim), loc, st, mm) in done: continue
            x = (["--limit", str(lim), "--h", "4", "--obj", "mindev"] if lim else []) + (["--mismatch"] if mm else [])
            jobs.append(((c, lim, loc, st, mm), ["--ctrl", c, "--loc", loc, "--year", str(yr), "--start", str(st), "--out", OUT]+x))
    def timeout(k):
        with open(TO, "a") as fh: fh.write(json.dumps(dict(key=list(k), limit_s=LIMIT_S, at=time.strftime("%F %X")))+"\n")
    run_queue(jobs, int(sys.argv[1]), timeout_s=LIMIT_S, on_timeout=timeout)
    print("MORE_DONE", flush=True)

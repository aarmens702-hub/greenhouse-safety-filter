"""Calibration study, part (b) (paper Figure 4b, T3): which group of parameters the filter's model must have right.
Rules, Bie plant, observer + noise + lower RH limit (as arm a5). One group wrong (V,A / G,A / G,V right) or one
group right (G / V / A); G geometry, V cover heat exchange and vents, A actuators per m2. Amsterdam 2010 day 59 and
Wag 2020 day 45. With --check, instead runs the all-groups-right check (G,V,A) on Amsterdam, which should match
the fully calibrated model.

Usage: python experiments/calibration_groups.py WORKERS [--check] [--out PATH]
(default outputs/bie_groups.jsonl, or outputs/bie_groups_check.jsonl with --check). 12 seasons, 7-30 min each.
Frozen outputs: results/filter/bie_groups.jsonl, bie_groups_check.jsonl. Was greenleaf-filter/stage_groups.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jobqueue import ROOT, read_rows, run_queue

ARMS = ["V,A", "G,A", "G,V", "G", "V", "A"]   # first three: one group wrong; last three: one group right
SEASONS = (("Amsterdam", 2010, 59), ("Wag", 2020, 45))

if __name__ == "__main__":
    check = "--check" in sys.argv
    OUT = sys.argv[sys.argv.index("--out")+1] if "--out" in sys.argv else str(ROOT/"outputs"/("bie_groups_check.jsonl" if check else "bie_groups.jsonl"))
    arms, seasons = (["G,V,A"], SEASONS[:1]) if check else (ARMS, SEASONS)
    base = ["--ctrl", "rb_tuned", "--plant", "bie", "--limit", "85", "--h", "4", "--obj", "mindev", "--lowrh", "--observer", "--noise", "--out", OUT]
    done = {(r["loc"], r["right"]) for r in read_rows(OUT)}
    jobs = [((loc, g), ["--loc", loc, "--year", str(yr), "--start", str(st), "--right", g]+base) for g in arms for loc, yr, st in seasons if (loc, g) not in done]
    run_queue(jobs, int(sys.argv[1]), runner="safety_filter.run_season_calib")
    print("GROUPS_DONE", flush=True)

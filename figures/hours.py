"""Violation hours for Figure 3a, split into humidity and temperature + CO2, at AGC f2.5 with AGC-trained agents,
original 6 seasons, default parameters; checks the totals equal the frozen hours in numbers.json. Builds H:
{group: (humidity hours per seed, temperature + CO2 hours per seed, total per seed)}. Imported by fig3_performance.py.
(The first half of the paper's helpers/fig3_hours.py.)"""
import statistics as S
import numpy as np
from common import *


real = rows(FR/"stage_real.jsonl"); lag = rows(RES/"ppolag/scores.jsonl")
T4 = NUM["T4 real prices"]

def season_mean(rs, f):
    d = {(r["loc"], r["start"]): r for r in rs}
    assert set(d) >= set(SE), "missing seasons"
    return S.mean(f(d[k]) for k in SE)

groups = {
    "Rules": ([r for r in real if r["ctrl"] == "rb_tuned_agc2020" and not r["limit"] and not r["mismatch"]],),
    "Rules\n+ filter": ([r for r in real if r["ctrl"] == "rb_tuned_agc2020" and r["limit"] and not r["mismatch"]],),
    "RL": tuple([r for r in lag if r["tag"] == f"d0.01_s{s}_agc25" and not r["mismatch"] and r["plant"] == "none"] for s in range(3)),
    "RL\n+ filter": tuple([r for r in real if r["ctrl"] == f"ppolag:d0.01_s{s}_agc25" and r["limit"] and not r["mismatch"]] for s in range(3)),
}
check = {"Rules": T4["agc-rules raw mm0"]["hours"], "Rules\n+ filter": T4["agc-rules +filter mm0"]["hours"]}
for s in range(3):
    check[f"RL s{s}"] = T4[f"d0.01_s{s}_agc25 raw mm0"]["hours"]; check[f"RL+f s{s}"] = T4[f"d0.01_s{s}_agc25 +filter mm0"]["hours"]

H = {}
for name, seeds in groups.items():
    rh = [season_mean(rs, lambda r: r["grade_rh"]) for rs in seeds]
    tc = [season_mean(rs, lambda r: r["temp_h"] + r["co2_h"]) for rs in seeds]
    tot = [a + b for a, b in zip(rh, tc)]
    H[name] = (rh, tc, tot)
# totals must equal the frozen hours
assert abs(H["Rules"][2][0] - check["Rules"]) < .01 and abs(H["Rules\n+ filter"][2][0] - check["Rules\n+ filter"]) < .01
for s in range(3):
    assert abs(H["RL"][2][s] - check[f"RL s{s}"]) < .01 and abs(H["RL\n+ filter"][2][s] - check[f"RL+f s{s}"]) < .01

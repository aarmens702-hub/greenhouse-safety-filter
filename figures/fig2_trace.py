"""Illustrative trace for Figure 2 (not a scored result; does not touch numbers.json).
Runs safety_filter.run_season once without and once with the filter, for the AGC-tuned rules on Amsterdam 2010
day 59, and saves per-step RH, temperature, CO2, applied controls and, with the filter, every proposed and applied
action plus the predicted RH over the next hour for each searched step. Both rows are appended to
figures/data/fig2_rows.jsonl; fig2_in_action.py checks they equal the frozen stage_real rows.
Usage: python figures/fig2_trace.py raw|filter [CTRL] [--days N] [--outdir DIR]   (default figures/data, overwrites)
(CTRL default rb_tuned_agc2020; e.g. ppolag:d0.01_s0_agc25. The filtered run takes about 10 minutes.)
Was the paper's helpers/fig2_trace.py, which patched logging lines into the runner; the runner now has a hook."""
import sys
from pathlib import Path

import casadi as ca
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from safety_filter.filter import climate
from safety_filter.run_season import build_parser, run, write

av = sys.argv[1:]; days = "60"; DATA = ROOT/"figures"/"data"
if "--days" in av: i = av.index("--days"); days = av[i+1]; del av[i:i+2]
if "--outdir" in av: i = av.index("--outdir"); DATA = Path(av[i+1]); del av[i:i+2]
mode = av[0]; CTRL = av[1] if len(av) > 1 else "rb_tuned_agc2020"; TAG = "" if CTRL == "rb_tuned_agc2020" else "_rl"
LOG, PRED = [], []


def _rh_path(s, x, u):
    """Predicted RH over the next hour for a held action, same model and weather as the filter's own check."""
    out = [climate(x)[2]]; w = s.raw.weather_data; t = s.raw.timestep
    for i in range(s.h):
        d = np.array(w[min(t+i, len(w)-1)], dtype=np.float64); d[1] += s.bias
        x = s.raw.F(x0=ca.DM(x), u=ca.DM(u), p=ca.vertcat(ca.DM(d), s.p))["xf"].full().ravel(); out.append(climate(x)[2])
    return out


def log(s, x, u_prop, u_appl, searched):
    LOG.append((s.raw.timestep, u_prop, u_appl))
    if searched: PRED.append((s.raw.timestep, _rh_path(s, x, u_prop), _rh_path(s, x, u_appl)))


a = build_parser().parse_args(["--ctrl", CTRL, "--loc", "Amsterdam", "--year", "2010", "--start", "59", "--days", days,
                               "--out", str(DATA/f"fig2_rows{TAG}.jsonl")] + (["--limit", "85", "--h", "4", "--obj", "mindev"] if mode == "filter" else []))
row, infos, f = run(a, on_decision=log)
write(row, a.out)
x = np.stack([i["x"] for i in infos]); u = np.stack([i["controls"] for i in infos])
clim = np.array([climate(r) for r in x])   # ppm, T, RH
out = dict(rh=clim[:, 2], T=clim[:, 1], co2=clim[:, 0], u=u)
if LOG:
    out["proposed"] = np.stack([p for _, p, _ in LOG]); out["applied"] = np.stack([q for _, _, q in LOG])
if PRED:
    out["pred_t"] = np.array([t for t, _, _ in PRED]); out["pred_prop"] = np.array([p for _, p, _ in PRED]); out["pred_appl"] = np.array([q for _, _, q in PRED])
np.savez(DATA/f"fig2_{mode}{TAG}.npz", **out)
print(mode, "steps", len(x), "rh>85 h", float((clim[:, 2] > 85).sum()*.25), "filter steps", f.hits if f else 0)

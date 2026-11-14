"""Figure 2: the filter in action (illustrative; agc-rules, Amsterdam 2010 day 59, default parameters).
Two separate closed-loop runs on the same weather: rules alone and rules + filter.
Top: RH in each run, with the 85% limit. Bottom: the vent opening actually applied in each run (rules alone vent late
and wide, after RH is already high; with the filter the vents open early and a little). Window = the 2 days where rules
alone spend the most time above 85%. Also checks both runs reproduce the frozen stage_real rows.
The two runs are in figures/data/, made by figures/fig2_trace.py."""
import json
import numpy as np
from common import *

raw, fil = np.load(DATA/"fig2_raw.npz"), np.load(DATA/"fig2_filter.npz")
# the two runs must match the frozen rows for this season
fz = {(r["ctrl"], bool(r["limit"])): r for r in rows(FR/"stage_real.jsonl") if r["loc"] == "Amsterdam" and r["start"] == 59 and not r["mismatch"] and r["ctrl"] == "rb_tuned_agc2020"}
mine = [json.loads(l) for l in open(DATA/"fig2_rows.jsonl")]
for m in mine:
    f = fz[(m["ctrl"], bool(m["limit"]))]
    assert abs(hrs(m) - hrs(f)) < 1e-9 and abs(m["yield_kg"] - f["yield_kg"]) < 1e-9, "trace run differs from frozen row"

N, D = len(raw["rh"]), 96
prop, appl = fil["proposed"][-N:], fil["applied"][-N:]           # one filter decision per step
changed = np.abs(appl - prop).max(axis=1) > 1e-6
over = (raw["rh"] > 85).astype(float)
w = 2*D; start = int(np.argmax(np.convolve(over, np.ones(w), "valid")))
sl = slice(start, start + w); t = np.arange(N)[sl]/D

fig, (a, b) = plt.subplots(2, 1, figsize=(COL1, 3.1), sharex=True, gridspec_kw={"height_ratios": [1.7, 1]})
h_raw, h_fil = over[sl].sum()*.25, float((fil["rh"][sl] > 85).sum()*.25)
a.plot(t, raw["rh"][sl], color=C["rules"], lw=.9, label="rules alone")
a.plot(t, fil["rh"][sl], color=C["rules+filter"], lw=1.1, label="rules + filter")
a.axhline(85, color=C["accent"], ls="--", lw=.8); a.text(t[-1] + .04, 85, "85%\nlimit", color=C["accent"], fontsize=9, va="center", ha="left", clip_on=False, linespacing=.9)
lo = min(raw["rh"][sl].min(), fil["rh"][sl].min()); a.set_ylim(min(60, lo - 12), 100)  # empty band under the curves holds the legend
a.text(t[0] + .02, 98.6, f"Above 85% in these 2 days:\nrules alone {h_raw:.0f} h, with filter {h_fil:.0f} h", fontsize=9, va="top", ha="left", fontweight="bold")
a.legend(frameon=False, fontsize=9, loc="lower center", ncol=2, handlelength=1.6, columnspacing=1.2, borderaxespad=.2)
a.set_ylabel("RH (%)", fontsize=9)

VENT = 3
b.plot(t, raw["u"][sl, VENT], color=C["rules"], lw=.9)               # vents actually applied, rules alone
b.plot(t, appl[sl, VENT], color=C["rules+filter"], lw=1.1)             # vents actually applied, rules + filter (same colours as top)
b.set_ylabel("Vent opening", fontsize=9); b.set_ylim(-.05, 1.05); b.set_xlabel("Day of season")
fig.tight_layout(h_pad=.4); save(fig, "fig2_in_action")
rv, fv = raw["u"][sl, VENT], appl[sl, VENT]
print("vents in window: rules alone mean", round(float(rv.mean()), 2), "(>0.5 in", round(100*float((rv > .5).mean())), "% of steps); with filter mean", round(float(fv.mean()), 2),
      "| season hours above 85%: rules", (raw["rh"] > 85).sum()*.25, "filter", (fil["rh"] > 85).sum()*.25)
vchg = (np.abs(appl[:, VENT] - prop[:, VENT]) > 1e-6)
print("fig2 done; window days", round(t[0], 1), "-", round(t[-1], 1), "| above 85% in window: rules", h_raw, "h, filter", h_fil, "h",
      "| filter changed", int(changed.sum()), "of", N, "steps (", round(100*changed.mean()), "% ); vents changed in", int((vchg & changed).sum()), "of those (", round(100*(vchg & changed).sum()/changed.sum()), "% )")

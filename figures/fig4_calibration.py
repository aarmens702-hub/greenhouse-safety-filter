"""Figure 4: calibration. Rules on Bie's plant. (a) violation hours vs how far the filter's model is moved from the default
toward Bie's parameters; (b) one group of parameters right or wrong. Sources: numbers.json T2, T3; no-filter baseline from stage2b.jsonl."""
import numpy as np
from common import *

T2, T3 = NUM["T2 Bie"], NUM["T3 groups"]
base = {r["loc"]: hrs(r) for r in rows(FR/"stage2b.jsonl") if r["ctrl"] == "rb_tuned" and r["plant"] == "bie" and not r["limit"]}
assert base == {"Amsterdam": 120.5, "Wag": 128.0}, base
SITES = (("Amsterdam", "Amsterdam 2010", C["ams"]), ("Wag", "Wageningen 2020", C["wag"]))

fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.3), gridspec_kw={"width_ratios": [1, 1.15]})

fr = [0.25, 0.5, 0.75, 1.0]
for loc, lab, col in SITES:
    y0 = T2[f"rb_tuned {loc} frac0.0 low1 ob0"]["hours"]        # default model; exact sensors (only run at 0%)
    ys = [T2[f"rb_tuned {loc} frac{f} low1 ob1"]["hours"] for f in fr]   # noisy sensors + observer
    a.plot([0] + [100*f for f in fr], [y0] + ys, color=col, lw=1.2, zorder=2)
    a.scatter([100*f for f in fr], ys, color=col, s=14, zorder=3, label=lab)
    a.scatter([0], [y0], facecolor="white", edgecolor=col, s=14, zorder=3)
    a.axhline(base[loc], color=col, ls="--", lw=.8)
a.text(101, base["Wag"] + 6, "no filter", ha="right", va="bottom", fontsize=7, color="#444444")
a.set_xticks([0, 25, 50, 75, 100], ["0%\n(default)", "25%", "50%", "75%", "100%\n(right)"])
a.set_xlabel("Filter's model: share of the way from default to real")
a.set_ylabel("Violation hours per season"); a.set_ylim(0, 330)
a.set_title("(a) How right the filter's model is", loc="left")
a.legend(frameon=False, loc="upper right")

cats = [("Geometry\nonly", "G"), ("All but\ngeometry", "V,A"), ("Vents\nonly", "V"), ("Equipment\nonly", "A")]
x = np.arange(len(cats)); w = .36
for j, (loc, lab, col) in enumerate(SITES):
    vals = [T3[f"right={k} {loc}"]["hours"] for _, k in cats]
    bars = b.bar(x + (j - .5)*w, vals, width=w, color=col, label=lab)
    for xi, v in zip(x + (j - .5)*w, vals): b.text(xi, v + 3, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5)
    b.hlines(base[loc], -.5, len(cats) - .5, color=col, ls="--", lw=.8)
b.set_xticks(x, [c[0] for c in cats]); b.set_xlim(-.5, len(cats) - .5); b.set_ylim(0, 235)
b.set_ylabel("Violation hours per season")
b.set_title("(b) Which settings the filter has right", loc="left")
b.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.0))
fig.tight_layout(w_pad=2.5); save(fig, "fig4_calibration")
print("fig4 done", base)

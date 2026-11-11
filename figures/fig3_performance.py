"""Figure 3: (a) violation hours; (b) break-even chart at challenge energy prices: profit of rules + filter and
RL + filter vs fruit price, crossover and the measured gap at €2.50 marked. The Dutch-energy break-even
(€1.68/kg; -0.3 at €1.60) is in the caption/text. Sources: numbers.json T4 and hash-checked rows (hours.py)."""
import statistics as S
import numpy as np
from common import *
from matplotlib.patches import Patch
from hours import H   # builds H, checks totals

T4 = NUM["T4 real prices"]
def line(rows_, price, f0):
    y = S.mean(r["yield_kg"] for r in rows_); p0 = S.mean(r[price] for r in rows_)
    return p0 - y*f0, y

fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.6), gridspec_kw={"width_ratios": [1, 1.1]})
names = list(H); x = np.arange(len(names)); cols = [C["rules"], C["rules+filter"], C["rl"], C["rl+filter"]]
rh = [S.mean(H[n][0]) for n in names]; tc = [S.mean(H[n][1]) for n in names]
a.bar(x, rh, color=cols, width=.6); a.bar(x, tc, bottom=rh, color=cols, width=.6, hatch="////", edgecolor="white", linewidth=0)
for i, n in enumerate(names):
    a.text(i, S.mean(H[n][2]) + 8, f"{S.mean(H[n][2]):.0f} h", ha="center", va="bottom", fontsize=7.5)
a.set_xticks(x, names); a.set_ylabel("Violation hours per season"); a.set_ylim(0, 420)
a.set_title("(a) Violation hours", loc="left", fontsize=8.5)
a.legend(handles=[Patch(facecolor="#BBBBBB", label="Humidity"), Patch(facecolor="#BBBBBB", hatch="////", edgecolor="white", label="Temperature + CO$_2$")], frameon=False, loc="upper right")

fr = np.linspace(.8, 3.0, 50)
r0, r1 = line([T4["agc-rules +filter mm0"]], "AGC f2.5", 2.5)
l0, l1 = line([T4[f"d0.01_s{s}_agc25 +filter mm0"] for s in range(3)], "AGC f2.5", 2.5)
be = (r0 - l0)/(l1 - r1); yb = r0 + r1*be
b.axvspan(be, fr[-1], color=C["rl+filter"], alpha=.07, lw=0)
b.plot(fr, r0 + r1*fr, color=C["rules+filter"], lw=1.5, label="Rules + filter")
b.plot(fr, l0 + l1*fr, color=C["rl+filter"], lw=1.5, label="RL + filter")
b.scatter([be], [yb], color="black", s=14, zorder=4)
b.annotate(f"break-even €{be:.2f}/kg", (be, yb), xytext=(be + .15, yb - 4.5), fontsize=7, arrowprops=dict(arrowstyle="-", lw=.6))
ym0, ym1 = r0 + r1*2.5, l0 + l1*2.5
b.plot([2.5, 2.5], [ym0, ym1], color=C["accent"], lw=1.6)
b.text(2.44, (ym0 + ym1)/2, f"{ym1 - ym0:+.1f} €/m²\nat €2.50", color=C["accent"], fontsize=7, fontweight="bold", va="center", ha="right")
b.text(2.95, 11, "RL + filter earns more →", ha="right", va="center", fontsize=6.8, color=C["rl+filter"])
b.set_xlim(fr[0], fr[-1]); b.set_xlabel("Fruit price (€/kg)"); b.set_ylabel("Profit (€/m² per season)")
b.set_title("(b) Profit vs fruit price (challenge energy prices)", loc="left", fontsize=8.5)
b.legend(frameon=False, loc="upper left")
fig.tight_layout(w_pad=3); save(fig, "fig3_performance")
print("fig3 done; break-even", round(be, 3), "gap", round(ym1 - ym0, 2))

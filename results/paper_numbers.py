#!/usr/bin/env python3
"""Rebuilds every number the paper uses from raw result rows. Read only; prints tables and writes numbers.json.
Each table names its source files. Exploratory results on 6 reused seasons; no sealed weather.
Usage: python results/paper_numbers.py [OUT]   (default: results/numbers.json, which it must reproduce byte for byte)
The paper uses T2 and T3 (rules rows), T4, T4b, T6 and T7. T1 and T5 are earlier exploratory comparisons kept so
that numbers.json is complete; their residual-RL and PPO rows come from agents of the earlier repository."""
import json, statistics as S, collections, sys
from pathlib import Path
R = Path(__file__).resolve().parent
FR = R/"filter"
L = lambda p: [json.loads(l) for l in open(p)]
P = {"sim f1.6": (1.6, .09, .30, .30), "NL f1.6": (1.6, .0429, .0889, .08), "AGC f2.5": (2.5, .03, .06, .08), "AGC f1.6": (1.6, .03, .06, .08)}
pf = lambda r, p: r["yield_kg"]*p[0]-r["heat_cost"]*p[1]/.09-r["elec_cost"]*p[2]/.30-r["co2_cost"]*p[3]/.30
hrs = lambda r: r["grade_rh"]+r["temp_h"]+r["co2_h"]
tot = hrs   # the grader counts too-dry hours inside grade_rh; older rows lack rh_lt50, so never sum it separately
en = lambda r: r["heat_cost"]+r["elec_cost"]+r["co2_cost"]
SE = [("Amsterdam", 45), ("Amsterdam", 59), ("Amsterdam", 75), ("Wag", 5), ("Wag", 45), ("Wag", 85)]
by = lambda rows: {(r["loc"], r["start"]): r for r in rows}
OUT = {}
def summ(d):
    assert set(d) >= set(SE), f"missing seasons {set(SE)-set(d)}"
    m = lambda f: S.mean(f(d[s]) for s in SE)
    return dict(hours=round(m(hrs), 2), yield_kg=round(m(lambda r: r["yield_kg"]), 3), energy_sim=round(m(en), 2), **{k: round(m(lambda r: pf(r, p)), 2) for k, p in P.items()})
def table(name, src, rows):
    print(f"\n## {name}\nsource: {src}"); OUT[name] = {}
    for k, d in rows:
        try: s = summ(d)
        except AssertionError as e: print(f"  {k:40s} INCOMPLETE {e}"); continue
        OUT[name][k] = s; print(f"  {k:40s} " + " ".join(f"{a} {b}" for a, b in s.items()))

# T1 default plant, sim-tuned controllers, exact and mismatch (filter h4 mindev limit 85)
s2 = L(FR/"stage2.jsonl")
for r in s2:   # old learned mismatch rows graded heat at nominal boiler (docs/preregistration_filter.md bug note)
    if r["ctrl"] != "rb_tuned" and r["mismatch"] and not r.get("grade_fixed"): r["heat_cost"] *= .85
sel = lambda c, lim, mm, h=4: by(r for r in s2 if r["ctrl"] == c and r["limit"] == lim and r["mismatch"] == mm and (lim == 0 or r.get("h") == h))
rows = []
for mm in (False, True):
    for c in ("rb_tuned", "residual:seed100", "ppo:seed0"):
        rows += [(f"{c} raw mm{int(mm)}", sel(c, 0, mm)), (f"{c} +filter h4 mm{int(mm)}", sel(c, 85.0, mm))]
k4 = by(r | {"grade_rh": r["rh"], "temp_h": r["temp"], "co2_h": r["co2"]} for r in L(R/"k4/score.jsonl") if r["arm"] == "C")
rows.append(("retuned rules C (K4)", k4))
table("T1 default plant", "results/filter/stage2.jsonl; results/k4/score.jsonl", rows)

# T2 Bie plant calibration curve (rules and residual), observer + noise + lower RH limit; baselines from stage2b_fix
bie = L(FR/"bie.jsonl"); fix = L(FR/"stage2b_fix.jsonl")
print("\n## T2 Bie calibration\nsource: results/filter/bie.jsonl, stage2b_fix.jsonl, stage2b.jsonl (rules baseline)")
base = {("residual:seed100", r["loc"]): r for r in fix if r["plant"] == "bie"}
s2b = L(FR/"stage2b.jsonl")
for r in s2b:
    if r["ctrl"] == "rb_tuned" and r["plant"] == "bie" and not r["limit"]: base[("rb_tuned", r["loc"])] = r
OUT["T2 Bie"] = {}
for c in ("rb_tuned", "residual:seed100"):
    for loc in ("Amsterdam", "Wag"):
        b = base.get((c, loc)); bt = tot(b) if b else None
        line = f"  {c:17s} {loc:9s} base {bt} |"
        for r in sorted((r for r in bie if r["ctrl"] == c and r["loc"] == loc), key=lambda r: (r["model_frac"], r["lowrh"], r["observer"])):
            key = f"{c} {loc} frac{r['model_frac']} low{int(r['lowrh'])} ob{int(r['observer'])}"
            OUT["T2 Bie"][key] = dict(hours=tot(r), change_pct=round(100*(tot(r)/bt-1), 1) if bt else None, yield_kg=round(r["yield_kg"], 3), energy_sim=round(en(r), 2))
            line += f" f{r['model_frac']}/l{int(r['lowrh'])}/o{int(r['observer'])} {tot(r)}" + (f" ({100*(tot(r)/bt-1):+.0f}%)" if bt else "")
        print(line)

# T3 one-group calibration (rules)
g = L(FR/"bie_groups.jsonl"); chk = L(FR/"bie_groups_check.jsonl")
print("\n## T3 groups\nsource: results/filter/bie_groups.jsonl, bie_groups_check.jsonl"); OUT["T3 groups"] = {}
for r in chk + g:
    OUT["T3 groups"][f"right={r['right']} {r['loc']}"] = dict(hours=tot(r), yield_kg=round(r["yield_kg"], 3), energy_sim=round(en(r), 2))
    print(f"  right={r['right']:6s} {r['loc']:9s} hours {tot(r):7.2f} yield {r['yield_kg']:.2f} energy {en(r):.1f}")

# T4 real prices: agc2020 rules and fair-price PPO-Lag, with and without filter
real = L(FR/"stage_real.jsonl"); lag = L(R/"ppolag/scores.jsonl")
rows = []
for mm in (False, True):
    rows += [(f"agc-rules raw mm{int(mm)}", by(r for r in real if r["ctrl"] == "rb_tuned_agc2020" and not r["limit"] and r["mismatch"] == mm)),
             (f"agc-rules +filter mm{int(mm)}", by(r for r in real if r["ctrl"] == "rb_tuned_agc2020" and r["limit"] and r["mismatch"] == mm))]
    for t in sorted({r["tag"] for r in lag if r["tag"].endswith(("_nl16", "_agc25", "dual2.0"))}):
        rows.append((f"{t} raw mm{int(mm)}", by(r for r in lag if r["tag"] == t and r["mismatch"] == mm and r["plant"] == "none")))
        rows.append((f"{t} +filter mm{int(mm)}", by(r for r in real if r["ctrl"] == "ppolag:"+t and r["limit"] and r["mismatch"] == mm)))
table("T4 real prices", "results/filter/stage_real.jsonl; results/ppolag/scores.jsonl", rows)

# T4b paired: each PPO-Lag + filter seed minus agc-rules + filter, at its own training price, nominal and mismatch
OUT["T4b paired"] = {}
def paired(rows, SEs, mm):
    ref = by(r for r in rows if r["ctrl"] == "rb_tuned_agc2020" and r["limit"] and r["mismatch"] == mm)
    res = {}
    for arm, price in (("agc25", "AGC f2.5"), ("nl16", "NL f1.6")):
        ds, hs, wins = [], [], 0
        for s in range(3):
            d = by(r for r in rows if r["ctrl"] == f"ppolag:d0.01_s{s}_{arm}" and r["limit"] and r["mismatch"] == mm)
            if not (set(d) >= set(SEs) and set(ref) >= set(SEs)): return None
            per = [pf(d[k], P[price])-pf(ref[k], P[price]) for k in SEs]
            ds.append(S.mean(per)); hs.append(S.mean(hrs(d[k]) for k in SEs)); wins += sum(x > 0 for x in per)
        res[arm] = dict(price=price, per_seed=[round(x, 2) for x in ds], mean=round(S.mean(ds), 2), sd=round(S.stdev(ds), 2), hours_per_seed=[round(h, 1) for h in hs],
                        ref_hours=round(S.mean(hrs(ref[k]) for k in SEs), 1), season_seed_wins=f"{wins}/{3*len(SEs)}")
    return res
for mm in (False, True):
    print(f"\n## T4b paired, own training price, mm{int(mm)}")
    res = paired(real, SE, mm); OUT["T4b paired"][f"mm{int(mm)}"] = res
    for k, v in (res or {}).items(): print(f"  {k}: {v}")

# T6 more seasons (docs/preregistration_filter.md 'More seasons'): 6 new nominal seasons, and pooled with the original 6
more = L(FR/"stage_more.jsonl"); NEW = [("Amsterdam", 15), ("Amsterdam", 30), ("Amsterdam", 90), ("Amsterdam", 105), ("Wag", 25), ("Wag", 65)]
print("\n## T6 more seasons\nsource: results/filter/stage_more.jsonl (+ stage_real.jsonl nominal for pooled)"); OUT["T6 more seasons"] = {}
pool = more + [r for r in real if not r["mismatch"]]
for name, rows_, SEs in (("new 6", more, NEW), ("pooled 12", pool, NEW+SE)):
    raw = by(r for r in rows_ if r["ctrl"] == "rb_tuned_agc2020" and not r["limit"]); rf = by(r for r in rows_ if r["ctrl"] == "rb_tuned_agc2020" and r["limit"])
    res = paired(rows_, SEs, False)
    if res is None or not set(raw) >= set(SEs): print(f"  {name} INCOMPLETE"); continue
    agc = res["agc25"]; rh = agc["ref_hours"]; holds = agc["mean"] >= 1 and S.mean(agc["hours_per_seed"]) <= max(rh*1.1, rh+5)
    OUT["T6 more seasons"][name] = dict(rules_raw_hours=round(S.mean(hrs(raw[k]) for k in SEs), 1), rules_filter_hours=rh, pre_registered_rule_holds=holds, **res)
    print(f"  {name}: {OUT['T6 more seasons'][name]}")

# T7 what the hours are made of (rules + filter vs AGC-trained PPO-Lag + filter, original 6, nominal), and the final Lagrange multipliers
print("\n## T7 hours breakdown and multipliers"); OUT["T7"] = {}
for c in ("rb_tuned_agc2020", "ppolag:d0.01_s0_agc25", "ppolag:d0.01_s1_agc25", "ppolag:d0.01_s2_agc25"):
    d = by(r for r in real if r["ctrl"] == c and r["limit"] and not r["mismatch"])
    OUT["T7"][c] = {f: round(S.mean(d[k][f] for k in SE), 2) for f in ("grade_rh", "temp_h", "co2_h")}
    print(f"  {c:26s} {OUT['T7'][c]}")
OUT["T7"]["final_lambda"] = {t.name: round(json.loads((t/"done.json").read_text())["final_lam"], 3) for t in sorted((R/"ppolag/runs").iterdir()) if (t/"done.json").exists()}
print("  final multipliers:", OUT["T7"]["final_lambda"])

# T5 first PPO-Lag comparison (sim-price training)")
simrf = sel("rb_tuned", 85.0, False)
for s in range(5):
    d = by(r for r in lag if r["tag"] == f"d0.01_s{s}" and not r["mismatch"] and r["plant"] == "none")
    print(f"  s{s} hours {S.mean(hrs(d[k]) for k in SE):.1f} seasonal cost {S.mean(d[k]['sgr_cost'] for k in SE):.1f} (budget 57.6) | " + " ".join(f"{n} {S.mean(pf(d[k], p)-pf(simrf[k], p) for k in SE):+.2f}" for n, p in P.items()))
out = Path(sys.argv[1]) if len(sys.argv) > 1 else R/"numbers.json"
json.dump(OUT, open(out, "w"), indent=1)
print(f"\nwrote {out}")

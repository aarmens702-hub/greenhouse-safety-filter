"""Score a trained PPO-Lagrangian agent on one season with the filter's grading. No filter here (the filtered
runs use safety_filter.run_season). Reads models/<tag>/ (or $MODELS_DIR/<tag>/); appends one JSON line to --out.
Usage: python controllers/eval_ppolag.py --tag d0.01_s0_agc25 --loc Amsterdam --year 2010 --start 59 [--plant none|bie] [--mismatch]"""
import argparse, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from greenleaf_core import metrics, scenarios, ppo as train
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
BIE = {46:14.982,47:72.632,48:3.0,49:5.211,50:4.675,56:1.101,59:0.110,60:2.37e-4,61:0.0185,108:4403.9,109:20.0,172:300.0}
MISMATCH = {60: 2.0, 50: .8, 51: .8, 52: .8, 53: .8, 55: .8, 108: .85, 21: 1.3}
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--ck", default="final"); ap.add_argument("--loc", required=True); ap.add_argument("--year", type=int, required=True); ap.add_argument("--start", type=int, required=True)
ap.add_argument("--plant", default="none", choices=["none", "bie"]); ap.add_argument("--mismatch", action="store_true"); ap.add_argument("--days", type=int, default=60); ap.add_argument("--out", default=str(ROOT/"outputs"/"ppolag_scores.jsonl"))
a = ap.parse_args()
if not ((a.loc == "Amsterdam" and a.year == 2010) or (a.loc == "Wag" and a.year == 2020)): sys.exit("refused: only Amsterdam 2010 and Wag 2020 are evaluation weather")
if a.loc == "Wag": scenarios.WEATHER_DIR = scenarios.AGC_WEATHER_DIR
def climate(x):
    T = x[2]; ppm = 1e6*8.3144598*(T+273.15)*(x[0]*1e-6)/(101325*44.01e-3)
    return ppm, T, min(100., 100*x[15]/(610.78*np.exp(17.2694*T/(T+238.3))))
rd = Path(os.environ.get("MODELS_DIR", ROOT/"models"))/a.tag; seed = int(a.tag.split("_s")[1].split("_")[0])
cfg = dict(method="ppo", seed=seed, season_length=60, penalty_weight=0.0)
st = train.build_vec_env(cfg, 1, scenario={"location": a.loc, "growth_year": a.year, "start_day": a.start}, season_length=a.days)
v = VecNormalize.load(rd/f"vecnorm_{a.ck}.pkl", st.venv); v.training = False; v.norm_reward = False
m = PPO.load(rd/f"ckpt_{a.ck}.zip", device="cpu", custom_objects={"learning_rate": 2e-5, "lr_schedule": lambda _: 2e-5, "clip_range": lambda _: .2}); obs = v.reset(); raw = v.unwrapped.envs[0].unwrapped
if a.mismatch:
    for i, f in MISMATCH.items(): raw.p[i] = raw.p[i]*f
if a.plant == "bie":
    for i, val in BIE.items(): raw.p[i] = val
P = np.array(raw.p, dtype=np.float64).copy(); infos = []; done = False; t0 = time.time()
while not done:
    act, _ = m.predict(obs, deterministic=True); obs, _, d, si = v.step(act); infos.append(si[0]); done = bool(d[0])
g = metrics.grade(infos, SimpleNamespace(p=P, N=raw.N))
x = np.stack([i["x"] for i in infos]); c = np.array([climate(r) for r in x]); ppm, Ta, rh = c[:, 0], c[:, 1], c[:, 2]
cost = float(sum(i["cost_co2"]+i["cost_temp"]+i["cost_rh"] for i in infos))
sgr = float((np.maximum(0, Ta-34)+np.maximum(0, 15-Ta)).sum()/15 + (np.maximum(0, ppm-1600)+np.maximum(0, 300-ppm)).sum()/2500 + (np.maximum(0, rh-85)+np.maximum(0, 50-rh)).sum()/15)
out = dict(tag=a.tag, ck=a.ck, loc=a.loc, year=a.year, start=a.start, plant=a.plant, mismatch=a.mismatch, days=a.days, cost=cost, sgr_cost=sgr, grade_rh=g["viol_hours_rh"], temp_h=g["viol_hours_temp"], co2_h=g["viol_hours_co2"],
    rh_gt85=float((rh > 85).sum()*.25), rh_lt50=float((rh < 50).sum()*.25), t_lt15=float((Ta < 15).sum()*.25), t_gt34=float((Ta > 34).sum()*.25), yield_kg=g["yield_kg_m2"], heat_cost=g["heat_cost"], elec_cost=g["elec_cost"], co2_cost=g["co2_cost"], profit_sim=g["profit_eur_m2"], valid=bool(g["valid"]), solver_failures=int(g["solver_failures"]), steps=len(infos), wall_s=round(time.time()-t0, 1))
Path(a.out).parent.mkdir(parents=True, exist_ok=True); open(a.out, "a").write(json.dumps(out)+"\n"); print(json.dumps(out))

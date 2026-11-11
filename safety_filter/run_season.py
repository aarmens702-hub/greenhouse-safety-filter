"""Run one 60-day season of a controller, with or without the safety filter, and append one JSON row.

    python -m safety_filter.run_season --ctrl rb_tuned_agc2020 --loc Amsterdam --year 2010 --start 59 \
        --limit 85 --h 4 --obj mindev [--mismatch] --out outputs/rows.jsonl

--ctrl: rb_tuned_agc2020 or rb_tuned (rules), or ppolag:<tag> (a PPO-Lagrangian agent in models/<tag>/,
or $MODELS_DIR/<tag>/). --limit 0 (default) runs without the filter; the paper's filter is
--limit 85 --h 4 --obj mindev. --mismatch scales plant parameters the filter does not know about
(see safety_filter.filter.MISMATCH) and biases the filter's outdoor temperature by +1.5 C.
--plant bie/cool0/lai08 swaps in a plant the filter's model does not match. --observer/--noise give
the filter measured temperature, humidity and CO2 instead of the full state.

Only Amsterdam 2010 and Wag (Wageningen) 2020 are accepted. An unfiltered season takes seconds; a
filtered one 1-20 minutes (median about 10) on one core.

This is the paper's filter_test_real.py with paths made repo-relative and the code split into functions;
the computation is unchanged.
"""
import argparse, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from controllers import rules as rule_based
from greenleaf_core import envs, metrics, scenarios, ppo as train
from safety_filter.filter import BIE, MISMATCH, Filter, climate

MODELS = Path(os.environ.get("MODELS_DIR", ROOT/"models"))


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ctrl", required=True)      # rb_tuned_agc2020 | rb_tuned | ppolag:<tag>
    ap.add_argument("--loc", required=True); ap.add_argument("--year", type=int, required=True); ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--limit", type=float, default=0)   # 0 = no filter
    ap.add_argument("--obj", default="mindev", choices=["mindev", "econ"]); ap.add_argument("--h", type=int, default=1)
    ap.add_argument("--mismatch", action="store_true"); ap.add_argument("--observer", action="store_true"); ap.add_argument("--noise", action="store_true")
    ap.add_argument("--plant", default="none", choices=["none", "bie", "cool0", "lai08"]); ap.add_argument("--days", type=int, default=60); ap.add_argument("--out", required=True)
    return ap


def check_weather(a):
    """Refuse anything but the evaluation weather; point the env at the AGC weather for Wag."""
    if not ((a.loc == "Amsterdam" and a.year == 2010) or (a.loc == "Wag" and a.year == 2020)):
        sys.exit(f"refused: {a.loc} {a.year}; only Amsterdam 2010 and Wag 2020 are evaluation weather")
    if a.loc == "Wag":
        scenarios.WEATHER_DIR = scenarios.AGC_WEATHER_DIR


def change_plant(raw, a):
    """Apply mismatch or an unknown plant to the simulated greenhouse (after the filter copied the nominal model)."""
    if a.mismatch:
        for i, m in MISMATCH.items(): raw.p[i] = raw.p[i]*m
    if a.plant == "bie":
        for i, v in BIE.items(): raw.p[i] = v
    if a.plant == "cool0":
        raw.p[186] = 0.; raw.p[172] = 142.5
    if a.plant == "lai08":
        raw.x[23] *= .8/2.55; raw.x_prev[23] *= .8/2.55


def install(raw, f):
    """Filter the final control whichever way the controller reached it."""
    orig = raw.action_scheme.to_full_control_input
    raw.action_scheme.to_full_control_input = lambda action: f(orig(action)).astype(np.float32)


def simulate(a, arm):
    """Run the season. arm(raw) sets up the filter and plant and returns the filter (or None).
    Returns (infos, raw, plant parameters captured before stepping, filter, start time)."""
    sc = {"location": a.loc, "growth_year": a.year, "start_day": a.start}
    PLANT = []
    def arm_and_capture(raw):
        f = arm(raw); PLANT.append(np.array(raw.p, dtype=np.float64).copy()); return f
    t0 = time.perf_counter(); infos = []
    if a.ctrl.startswith("rb_"):
        env = envs.SolverFailureFlag(envs.make_raw_env(scenario=sc, season_length=a.days, normalize_actions=False))
        c = rule_based.make_controller(rule_based.load_rb_params(a.ctrl[3:])); env.reset(); raw = env.unwrapped; f = arm_and_capture(raw)
        done = False
        while not done:
            _, _, te, tr, info = env.step(rule_based.rb_action(env, c)); infos.append(info); done = te or tr
    else:
        kind, seed = a.ctrl.split(":")
        if kind != "ppolag": sys.exit(f"unknown controller {a.ctrl}")
        rd = MODELS/seed; ck = "final"   # seed is the run tag, e.g. d0.01_s0_agc25
        cfg = dict(method="ppo", seed=int(seed.split("_s")[1].split("_")[0]), season_length=60, penalty_weight=0.0)
        st = train.build_vec_env(cfg, 1, scenario=sc, season_length=a.days)
        v = VecNormalize.load(rd/f"vecnorm_{ck}.pkl", st.venv); v.training = False; v.norm_reward = False
        m = PPO.load(rd/f"ckpt_{ck}.zip", device="cpu", custom_objects={"learning_rate": 2e-5, "lr_schedule": lambda _: 2e-5, "clip_range": lambda _: .2}); obs = v.reset(); raw = v.unwrapped.envs[0].unwrapped; f = arm_and_capture(raw)
        done = False
        while not done:
            act, _ = m.predict(obs, deterministic=True); obs, _, d, si = v.step(act); infos.append(si[0]); done = bool(d[0])
    return infos, raw, PLANT[0], f, t0


def season_row(a, infos, raw, plant, f, t0):
    """The graded season as one result row."""
    # Vec envs auto-reset at season end, which restores raw.p, so grade with the plant captured before stepping.
    g = metrics.grade(infos, SimpleNamespace(p=plant, N=raw.N))
    x = np.stack([i["x"] for i in infos]); u = np.stack([i["controls"] for i in infos])
    rh = np.array([climate(r)[2] for r in x])
    return dict(grade_fixed=True, ctrl=a.ctrl, loc=a.loc, year=a.year, start=a.start, days=a.days, observer=a.observer, noise=a.noise, plant=a.plant, est_err_med=float(np.median(f.est_err)) if f and f.est_err else 0., limit=a.limit, obj=a.obj if a.limit else None, h=a.h if a.limit else None, mismatch=a.mismatch,
        rh_gt80=float((rh > 80).sum()*.25), rh_gt85=float((rh > 85).sum()*.25), rh_gt90=float((rh > 90).sum()*.25), rh_gt95=float((rh > 95).sum()*.25),
        grade_rh=g["viol_hours_rh"], temp_h=g["viol_hours_temp"], co2_h=g["viol_hours_co2"], yield_kg=g["yield_kg_m2"],
        heat_cost=g["heat_cost"], elec_cost=g["elec_cost"], co2_cost=g["co2_cost"], profit_sim=g["profit_eur_m2"], valid=bool(g["valid"]), solver_failures=int(g["solver_failures"]), steps=len(infos),
        max_du=float(np.abs(np.diff(u, axis=0)).max()), filter_steps=f.hits if f else 0, evals=f.evals if f else 0, eval_fails=f.fails if f else 0,
        max_step_ms=round(f.tmax*1e3, 1) if f else 0, wall_s=round(time.perf_counter()-t0, 1))


def run(a, on_decision=None):
    """One season; returns (row, infos, filter). on_decision is passed to the Filter (logging only)."""
    check_weather(a)
    def arm(raw):
        f = Filter(raw, a.limit, a.obj, a.h, a.mismatch, observer=a.observer, noise=a.noise, on_decision=on_decision) if a.limit else None
        change_plant(raw, a)
        if f: install(raw, f)
        return f
    infos, raw, plant, f, t0 = simulate(a, arm)
    return season_row(a, infos, raw, plant, f, t0), infos, f


def write(row, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as fh: fh.write(json.dumps(row)+"\n")
    print(json.dumps(row))


def main(argv=None):
    a = build_parser().parse_args(argv)
    row, _, _ = run(a)
    write(row, a.out)


if __name__ == "__main__":
    main()

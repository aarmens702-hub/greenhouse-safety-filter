"""Smoke test (a few seconds): build the env on Amsterdam 2010 day 59, get one action from the AGC-tuned rules,
pass it through the safety filter once, then take one filtered step. Usage: python tests/smoke_test.py"""
import sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from controllers import rules
from greenleaf_core import envs
from safety_filter import Filter, climate
from safety_filter.run_season import install

t0 = time.perf_counter()
env = envs.SolverFailureFlag(envs.make_raw_env(scenario={"location": "Amsterdam", "growth_year": 2010, "start_day": 59},
                                               season_length=1, normalize_actions=False))
env.reset(); raw = env.unwrapped
c = rules.make_controller(rules.load_rb_params("tuned_agc2020"))
u_prop = raw.action_scheme.to_full_control_input(rules.rb_action(env, c))
f = Filter(raw, limit=85, obj="mindev", h=4)
u = f(u_prop)
assert u.shape == (6,) and np.all((u >= 0) & (u <= 1))
print("proposed", np.round(u_prop, 3), "\napplied ", np.round(u, 3), "| searched:", bool(f.hits), "| model evaluations:", f.evals)
install(raw, Filter(raw, limit=85, obj="mindev", h=4))
_, _, _, _, info = env.step(rules.rb_action(env, c))
ppm, T, rh = climate(info["x"])
print(f"after one filtered step: CO2 {ppm:.0f} ppm, T {T:.1f} C, RH {rh:.1f}% | {time.perf_counter()-t0:.1f} s")
print("SMOKE_OK")

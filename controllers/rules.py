"""The grower's rules: GreenLight-Gym's RuleBasedController with tuned parameters, one action per step.

Copied from greenleaf_control/rule_based.py in the author's earlier repository greenleaf-dashboard
(control/, commit 701b43e). The rule logic itself is upstream's (gl_gym.components.rule_based); only
the parameter sets are ours, in configs/:
- rule_based_tuned_agc2020.yml: parameters tuned at the AGC 2020 energy tariffs (fruit EUR 1.6/kg);
  the "rules" arm of the main comparison (run name rb_tuned_agc2020).
- rule_based_tuned.yml: parameters tuned at the simulator's default prices; the rules arm of the
  calibration study (run name rb_tuned).
Both were tuned in greenleaf-dashboard (tune_rb.py: 12 setpoints, Optuna TPE, on the training
weather Amsterdam 2008, 2009, 2011); the tuning code is not part of this repository.

rb_action returns ABSOLUTE controls in [0, 1], so the rule-based controller needs an env built with
normalize_actions=False.
"""

from pathlib import Path

import numpy as np
import yaml
from gl_gym.components.rule_based import RuleBasedController
from gl_gym.core.types import StepContext

CONFIGS = Path(__file__).resolve().parent / "configs"
NAMES = ("tuned", "tuned_agc2020")


def load_rb_params(name: str) -> dict:
    """The controller's parameters from configs/rule_based_<name>.yml; name is one of NAMES."""
    if name not in NAMES:
        raise ValueError(f"unknown rule-based params {name!r}; expected one of {NAMES}")
    return yaml.safe_load((CONFIGS / f"rule_based_{name}.yml").read_text())["GreenLightEnv"]


def make_controller(params: dict) -> RuleBasedController:
    """Upstream's RuleBasedController with these parameters."""
    return RuleBasedController(**params)


def rb_action(raw_env, controller) -> np.ndarray:
    """The controller's action for the env's current step: float32, shape (6,), clipped to [0, 1]."""
    e = raw_env.unwrapped
    ctx = StepContext(
        t=e.timestep,
        dt=e.dt,
        Np=e.Np,
        x_prev=e.x_prev,
        x=e.x,
        u=e.u,
        p=e.p,
        d=e.weather_data,
        hour_of_day=e.hour_of_day,
        day_of_year=e.day_of_year,
    )
    return np.clip(controller.predict(ctx), 0.0, 1.0).astype(np.float32)

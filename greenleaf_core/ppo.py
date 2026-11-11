"""PPO settings and the vectorised env stack used to train and evaluate the RL agents.

Subset of greenleaf_control/train.py (greenleaf-dashboard, commit 701b43e): the env stack is
DummyVecEnv -> VecMonitor -> VecNormalize, with configs/ppo.yml (GreenLight-Gym2's PPO settings)
as written.
"""

import functools
from pathlib import Path

import torch
import yaml
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

from greenleaf_core import envs

CONFIGS = Path(__file__).resolve().parent / "configs"
# SB3's own defaults, stated explicitly; gamma comes from configs/ppo.yml.
VECNORMALIZE = {"norm_obs": True, "norm_reward": True, "clip_obs": 10.0, "clip_reward": 10.0}
PRICE_REGIME = "sim"
# configs/ppo.yml's policy_kwargs strings, mapped explicitly; any other string raises.
FROM_YML = {
    "optimizer_class": {"adam": torch.optim.Adam},
    "activation_fn": {"silu": torch.nn.SiLU},
    "log_std_init": {"np.log(1)": 0.0},
}


def ppo_yml() -> dict:
    """configs/ppo.yml's GreenLightEnv block, as written."""
    return yaml.safe_load((CONFIGS / "ppo.yml").read_text())["GreenLightEnv"]


def from_yml(key: str, value):
    """A policy_kwargs value from ppo.yml, with its string (if any) mapped by FROM_YML."""
    if not isinstance(value, str):
        return value
    table = FROM_YML.get(key, {})
    if value not in table:
        raise ValueError(f"configs/ppo.yml policy_kwargs {key}: no mapping for {value!r}")
    return table[value]


def build_vec_env(
    config: dict,
    n_envs: int,
    *,
    scenario: dict | None = None,
    season_length: int | None = None,
) -> VecNormalize:
    """The run's env stack, env i = make_env(method, seed=seed + i), VecNormalize training.

    scenario and season_length override the config's weather and season (for evaluation). The envs
    are priced at the config's price_regime, sim for a config without one.
    """
    fns = [
        functools.partial(
            envs.make_env,
            config["method"],
            seed=config["seed"] + i,
            scenario=scenario,
            season_length=config["season_length"] if season_length is None else season_length,
            penalty_weight=config["penalty_weight"],
            price_regime=config.get("price_regime", PRICE_REGIME),
        )
        for i in range(n_envs)
    ]
    venv = VecMonitor(SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns))
    return VecNormalize(venv, gamma=ppo_yml()["gamma"], **VECNORMALIZE)

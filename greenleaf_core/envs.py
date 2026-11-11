"""The env, built one way. Subset of greenleaf_control/envs.py (greenleaf-dashboard, commit 701b43e).

make_raw_env is upstream's tomato env (gl_gym 0.3.2 @2d3febb) with ConstraintCostReward and
scenarios.WEATHER_DIR, a flat Box observation, and season progress when asked. make_env builds the
RL agents' env ("ppo") and always puts SolverFailureFlag on top.

- A season is exactly N steps (96 a day, 5,760 for 60 days). Upstream ends `terminated` on step
  N + 1, so _SeasonEnd ends it on step N. Nothing truncates for time, so a truncation is always one
  of upstream's swallowed solver failures.
- The rule-based controller works on the raw env (absolute controls, raw state), below anything
  that normalizes observations or actions.
- A season that would read sealed weather (scenarios.SEALED_YEARS) is refused: its own year, or the
  next year's file, which upstream appends when the season plus its 0.5-day lookahead runs past the
  end of the year.
- Every step's info carries x_prev and x, the raw state before and after the step (for metrics).
- The reward is priced at a price regime (REWARD_PRICES), sim unless another is named.

metrics.py grades runs from raw states and controls and must never import this module (it
imports rewards).
"""

from itertools import product

import gl_gym  # noqa: F401  (importing gl_gym registers ENV_ID with gymnasium)
import gymnasium as gym
import numpy as np
from gymnasium.utils import RecordConstructorArgs

from greenleaf_core import scenarios
from greenleaf_core.rewards import ConstraintCostReward

ENV_ID = "gl_gym/GreenLightTomato-v0"
METHODS = ("ppo",)
# Price regimes as upstream's reward kwargs, laid over its registered ones: sim keeps them all
# (fruit 1.6 EUR/kg, heat 0.09 and electricity 0.3 EUR/kWh, CO2 0.3 EUR/kg). agc2020's fruit 1.6 is
# the simulator's price; only heat, electricity and CO2 are AGC2's 2020 tariffs (Hemming et al.
# 2020, Sensors 20(22):6430, section 2). Its electricity 0.06 is the unweighted mean of AGC2's 0.08
# on-peak (07:00-23:00) and 0.04 off-peak tariffs, flat (not time-of-use). Written apart from
# metrics.PRICE_REGIMES, so the grader stays independent of the reward. controllers/train_ppolag.py
# adds the nl16 and agc25 regimes the paper's agents train at.
REWARD_PRICES = {
    "sim": {},
    "agc2020": {"fruit_price": 1.6, "heating_price": 0.03, "elec_price": 0.06, "co2_price": 0.08},
}
_REFUSED = "sealed test weather (scenarios.SEALED_YEARS)"


def _refuse_test(s: dict, season_length: int) -> None:
    """Raises if a season on s would read a sealed year's weather file.

    Upstream int()-casts the year (so "2010" reads 2010.csv), and appends <year + 1>.csv when start
    day + season + the 0.5-day lookahead runs past the year's file. The + 1 day covers the lookahead
    and rounding, so the check is conservative.
    """
    year = int(s["growth_year"])
    if scenarios.is_test({**s, "growth_year": year}):
        raise ValueError(f"scenario {s}: its own year, {year}, is {_REFUSED}")
    if int(s["start_day"]) + season_length + 1 >= 365 and scenarios.is_test(
        {**s, "growth_year": year + 1}
    ):
        raise ValueError(
            f"scenario {s} over {season_length} days runs past its year, so upstream would read the"
            f" next year's file, {year + 1}.csv, which is {_REFUSED}"
        )


class _RawStates(gym.Wrapper, RecordConstructorArgs):
    """Puts float64 copies of the raw state before and after each step in info["x_prev"], info["x"].

    Upstream's step sets x_prev = copy(x) before integrating x, so after it they are the two ends.
    """

    def __init__(self, env: gym.Env):
        RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw = self.env.unwrapped
        info["x_prev"] = np.array(raw.x_prev, dtype=np.float64, copy=True)
        info["x"] = np.array(raw.x, dtype=np.float64, copy=True)
        return obs, reward, terminated, truncated, info


class _SeasonEnd(gym.Wrapper, RecordConstructorArgs):
    """Ends the season `terminated` after exactly N steps; upstream runs N + 1."""

    def __init__(self, env: gym.Env):
        RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw = self.env.unwrapped
        return obs, reward, bool(terminated or raw.timestep >= raw.N), truncated, info


class _RefuseTestReset(gym.Wrapper, RecordConstructorArgs):
    """Refuses reset(options={"scenario": s}) onto a season that would read sealed weather, which
    bypasses upstream's sampler (and so the check at construction)."""

    def __init__(self, env: gym.Env):
        RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

    def reset(self, *, seed=None, options=None):
        if options is not None and "scenario" in options:
            _refuse_test(options["scenario"], self.env.unwrapped.season_length)
        return self.env.reset(seed=seed, options=options)


class SeasonProgress(gym.ObservationWrapper, RecordConstructorArgs):
    """Appends season progress, timestep / N in [0, 1], to a flat Box observation."""

    def __init__(self, env: gym.Env):
        RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)
        space = env.observation_space
        self.observation_space = gym.spaces.Box(
            np.append(space.low, 0.0).astype(np.float32),
            np.append(space.high, 1.0).astype(np.float32),
            dtype=np.float32,
        )

    def observation(self, observation):
        raw = self.env.unwrapped
        return np.append(observation, np.float32(raw.timestep / raw.N)).astype(np.float32)


class SolverFailureFlag(gym.Wrapper, RecordConstructorArgs):
    """Sets info["solver_failure"] = bool(truncated).

    Nothing truncates for time (a season is exactly N steps), so upstream truncates only when its
    ODE solver fails, an error it swallows and prints. That includes a failure on the season's last step.
    """

    def __init__(self, env: gym.Env):
        RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["solver_failure"] = bool(truncated)
        return obs, reward, terminated, truncated, info


def make_raw_env(
    *,
    scenario: dict | None = None,
    sampler_kwargs: dict | None = None,
    season_length: int = 60,
    normalize_actions: bool,
    penalty_weight: float = 1.0,
    season_progress: bool = False,
    price_regime: str = "sim",
) -> gym.Env:
    """Upstream's env on a fixed scenario or a random sampler, with ConstraintCostReward at
    price_regime's prices (REWARD_PRICES; an unknown name raises).

    Ends the season after exactly N steps and returns a flat Box observation
    (FlattenObservation, then SeasonProgress if asked). Refuses sealed weather.
    """
    if price_regime not in REWARD_PRICES:
        raise ValueError(
            f"unknown price regime {price_regime!r}; expected one of {tuple(REWARD_PRICES)}"
        )
    if (scenario is None) == (sampler_kwargs is None):
        raise ValueError("make_raw_env needs exactly one of scenario or sampler_kwargs")
    if scenario is not None:
        sampler, sampler_args, support = "fixed", scenario, [scenario]
    else:
        sampler, sampler_args = "random", sampler_kwargs
        support = [
            {"location": loc, "growth_year": year, "start_day": day}
            for loc, year, day in product(
                sampler_kwargs["locations"],
                sampler_kwargs["growth_years"],
                sampler_kwargs["start_days"],
            )
        ]
    for s in support:
        _refuse_test(s, season_length)
    env = gym.make(
        ENV_ID,
        season_length=season_length,
        normalize_actions=normalize_actions,
        weather_data_dir=str(scenarios.WEATHER_DIR),
        reward_fn=ConstraintCostReward,
        # GreenhouseReward has no price defaults, so keep the registered ones, with the regime's
        # prices over them (none for sim).
        reward_kwargs={
            **gym.spec(ENV_ID).kwargs["reward_kwargs"],
            **REWARD_PRICES[price_regime],
            "penalty_weight": penalty_weight,
        },
        weather_scenario_sampler=sampler,
        weather_scenario_sampler_kwargs=sampler_args,
    )
    env = _RefuseTestReset(_SeasonEnd(_RawStates(env)))
    env = gym.wrappers.FlattenObservation(env)
    if season_progress:
        env = SeasonProgress(env)
    return env


def make_env(
    method: str,
    *,
    seed: int,
    scenario: dict | None = None,
    season_length: int = 60,
    penalty_weight: float = 1.0,
    price_regime: str = "sim",
) -> gym.Env:
    """The RL agents' env, reset once with seed (and its action space seeded): normalized actions,
    season progress in the observation. A given scenario overrides the training sampler."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if scenario is not None:
        weather = {"scenario": scenario}
    else:
        weather = {"sampler_kwargs": scenarios.TRAIN_SAMPLER_KWARGS}
    env = make_raw_env(
        **weather,
        season_length=season_length,
        normalize_actions=True,
        penalty_weight=penalty_weight,
        season_progress=True,
        price_regime=price_regime,
    )
    env = SolverFailureFlag(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env

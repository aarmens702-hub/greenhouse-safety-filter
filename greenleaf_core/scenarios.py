"""Weather locations and the sealed years.

A scenario is {"location": str, "growth_year": int, "start_day": int}; the env reads
WEATHER_DIR/<location>/<growth_year>.csv and starts the season on day start_day.

- Amsterdam 2008-2011 are the files bundled with gl_gym 0.3.2 (GreenLight-Gym2, commit 2d3febb),
  byte-identical to the copies used for the paper. WEATHER_DIR points there by default.
- "Wag" 2020 is Wageningen weather from the Autonomous Greenhouse Challenge 2nd edition dataset
  (CC0), converted to gl_gym's format; it lives in this repository under data/weather
  (AGC_WEATHER_DIR, see data/README.md). Scripts switch WEATHER_DIR to it for location "Wag".

Years 2012-2023 are sealed test weather in the wider project: nothing here may simulate them.
Adapted from greenleaf_control/scenarios.py (greenleaf-dashboard, commit 701b43e).
"""

from pathlib import Path

import gl_gym

ROOT = Path(__file__).resolve().parents[1]
# Read at call time, so scripts can point it at AGC_WEATHER_DIR.
WEATHER_DIR = Path(gl_gym.__file__).resolve().parent / "data" / "weather"
AGC_WEATHER_DIR = ROOT / "data" / "weather"

# Training sampler: gl-gym's random sampler draws a year and a start day at every reset.
TRAIN_SAMPLER_KWARGS = {
    "locations": ["Amsterdam"],
    "growth_years": [2008, 2009, 2011],
    "start_days": list(range(45, 76)),
}

SEALED_YEARS = frozenset(range(2012, 2024))


def is_test(scenario: dict) -> bool:
    """True for any scenario in a sealed year, at any location."""
    return scenario["growth_year"] in SEALED_YEARS

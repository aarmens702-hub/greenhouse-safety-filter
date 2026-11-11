# Frozen results

Every number in the paper comes from `numbers.json`, which `paper_numbers.py` rebuilds byte for byte
from the raw rows here (sha256 values in `docs/FROZEN.md`). One JSON row per simulated season.

| File | Made by | Used for |
|---|---|---|
| `filter/stage_real.jsonl` | `experiments/main_comparison.py` | T4, T4b, T7: Table 1, Figure 3, mismatch |
| `filter/stage_more.jsonl` | `experiments/more_seasons.py` | T6: the 6 new seasons |
| `ppolag/scores.jsonl` | `controllers/run_ppolag.py` (`eval_ppolag.py`) | unfiltered RL rows of T4, T5 |
| `ppolag/runs/<tag>/done.json`, `lag.csv` | `controllers/train_ppolag.py` | T7 final multipliers; training logs |
| `filter/bie.jsonl` | `experiments/calibration_blend.py` | T2: Figure 4a |
| `filter/bie_groups.jsonl`, `bie_groups_check.jsonl` | `experiments/calibration_groups.py` | T3: Figure 4b |
| `filter/stage2b.jsonl`, `stage2b_fix.jsonl` | `experiments/calibration_baselines.py` | T2 and Figure 4 "no filter" baselines |
| `filter/bie_noise.jsonl` | `experiments/calibration_noise.sh` | pre-registered sensor-noise check (not in numbers.json) |
| `filter/stage2.jsonl`, `k4/score.jsonl` | earlier versions of the runner and a retuned-rules study | T1 only (not in the paper) |

Rows for `residual:seed100`, `ppo:seed0` and `lag:*` controllers come from RL agents of the author's
earlier repository (greenleaf-dashboard); they appear in T1 and T2 of numbers.json but not in the paper,
and those agents are not part of this repository. Rows with `grade_fixed` absent predate a grading fix
(docs/preregistration_filter.md, "Bug found"); paper_numbers.py applies the documented correction.

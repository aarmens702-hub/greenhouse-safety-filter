# Frozen numbers

Frozen 2026-11-06 23:49 at Aarmen's request. The paper may only use numbers from numbers.json. Any change after this needs a dated note below saying what changed and why.

Rebuild: `python3 paper_numbers.py` must reproduce numbers.json byte for byte (sha256 below).

## Headline (exploratory, simulated, no sealed weather)
- Filter on rules, default plant: 381.8 h -> 17.3 h (T1).
- AGC f2.5, AGC-trained PPO-Lag + filter minus agc-rules + filter, nominal, original 6 seasons: +4.93 EUR/m2 (SD 0.45, 3 seeds), 4.5-5.8 h vs 42.5 h; 18/18 season-seed wins (T4b).
- Same, mismatch: +4.49 (SD 0.32), 25-35 h vs 103.1 h (T4b).
- Same, 6 new seasons (pre-registered 2026-11-06): +4.03 (SD 1.31), 5.2-6.1 h vs 58.0 h; 17/18 wins; rule holds. Pooled 12: +4.48 (SD 0.88), 35/36 wins (T6).
- NL f1.6, NL-trained: -0.26 (SD 0.46) original 6; -0.48 (SD 0.78) new 6; -0.37 pooled; -0.71 mismatch. About even (T4b, T6).
- Rules + filter's hours are mostly temperature (28.5 of 42.5); final multipliers 0.96-1.00 at d=0.01, so describe as PPO with a fixed cost weight of about 1 (T7).

## Disclosures added by the code check (paper/code_check.md)
- Filter output bypasses the agent's 0.1/step rate limit (both arms filtered).
- Hours are summed per constraint; one step can count for RH and temperature.
- Rules tuned at fruit 1.6; AGC agents trained at fruit 2.5.
- 4 runs of the extra seasons were paused by Mac sleep and requeued; all 48 finished, none timed out.

## sha256
b8c633a651ddde1cd39674a72542a83eaa72535b9d992580b73f59841542405a  numbers.json
8a0b1df2b5fb56dc5e7021c78f577a6732612d0684289ebba6d0f55d53cde712  paper_numbers.py
6e31a69eb8a708a0faf6353fdc4b0f582ef5f584e394a8a0c58b787aea33693b  stage_real.jsonl
1eff4e9721411ab4241b0842a009d6524b85c627439db7f551b6f59b85f39065  stage_more.jsonl
598411e36472fefe346f087d46c7120ece2d60a254256528f5ff7859d70b5835  stage2.jsonl
aa4f126013e5f0f7710f9414d93da38f8bc7c37006deca187cc69d9ea2938380  bie.jsonl
7e3a516788ad81e30e3962d47263a840bbc7593439d70d8c9f75646b4afd4566  stage2b_fix.jsonl
ade5651c7f4afcc6eb2b74bab93f3c67465844f848ff5342e05c7a6363503a36  stage2b.jsonl
17c6e7233db617745fb15555234c090994e08f74ff849251cfaef53fe5e541ad  bie_groups.jsonl
776ed4aef0a346f4d4b2d8e2af0da97c80678f4d924620da87e3a8142f6594f1  bie_groups_check.jsonl
2758d4a1b4dd874ba301094defd70ab24487d7e2b23a8d122f17e7676881c422  cycles/05/ppolag/scores.jsonl
a585e9cc4232471c17ad1c756cb64d037eeaca2eac36fe6d57cb07b7d43d84b8  cycles/02/k4/score.jsonl

## Note 2026-11-09: figures and table (no frozen number changed)
Figure scripts in figures/ read numbers.json and raw rows whose sha256 above they check on every load (common.py). Fig 3a splits hours into humidity vs temperature from the frozen rows and asserts the totals equal numbers.json. Fig 2 comes from one new illustrative rerun of agc-rules, Amsterdam 2010 day 59, with and without the filter (figures/fig2_trace.py); both runs reproduce the frozen stage_real rows exactly (checked in fig2_in_action.py). New fact from that run, illustrative only: the filter changed the rules' action in 4,861 of 5,760 steps (84%).

## Note 2026-11-10: public repository (no frozen number changed)
This file was copied into the public repository unchanged above this note. Paths above are the
working-folder layout; in the repository the same files live at the paths below, with the same
sha256 (the figure scripts check these on every load). `results/paper_numbers.py` differs from the
frozen `paper_numbers.py` only in file paths, an optional output argument and its docstring; it
rebuilds `results/numbers.json` byte for byte (sha256 b8c633a6... as above).
Rebuild in the repository: `python results/paper_numbers.py`.

41c71f16f1a39c0c402c8758b05fcf5ff9ae4424b935457dbae6ce2fd7c50997  results/paper_numbers.py
b8c633a651ddde1cd39674a72542a83eaa72535b9d992580b73f59841542405a  results/numbers.json
6e31a69eb8a708a0faf6353fdc4b0f582ef5f584e394a8a0c58b787aea33693b  results/filter/stage_real.jsonl
1eff4e9721411ab4241b0842a009d6524b85c627439db7f551b6f59b85f39065  results/filter/stage_more.jsonl
598411e36472fefe346f087d46c7120ece2d60a254256528f5ff7859d70b5835  results/filter/stage2.jsonl
aa4f126013e5f0f7710f9414d93da38f8bc7c37006deca187cc69d9ea2938380  results/filter/bie.jsonl
7e3a516788ad81e30e3962d47263a840bbc7593439d70d8c9f75646b4afd4566  results/filter/stage2b_fix.jsonl
ade5651c7f4afcc6eb2b74bab93f3c67465844f848ff5342e05c7a6363503a36  results/filter/stage2b.jsonl
17c6e7233db617745fb15555234c090994e08f74ff849251cfaef53fe5e541ad  results/filter/bie_groups.jsonl
776ed4aef0a346f4d4b2d8e2af0da97c80678f4d924620da87e3a8142f6594f1  results/filter/bie_groups_check.jsonl
2758d4a1b4dd874ba301094defd70ab24487d7e2b23a8d122f17e7676881c422  results/ppolag/scores.jsonl
a585e9cc4232471c17ad1c756cb64d037eeaca2eac36fe6d57cb07b7d43d84b8  results/k4/score.jsonl

Also included, not used by numbers.json: `results/filter/bie_noise.jsonl` (the pre-registered
sensor-noise check, sha256 0df8c26491e549d309981cfa3d842cf4d14426cd0ed303ba3e96b6f375c125d7).

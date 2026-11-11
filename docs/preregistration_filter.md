<!--
Copied verbatim from the project's working notes (greenleaf-filter/PLAN.md, last edited 2026-11-06).
It is the dated pre-registration log for the filter experiments: each section was written before the runs
it describes, and later changes are recorded in place. File names refer to the original scripts:
filter_test.py / filter_test_real.py -> safety_filter/run_season.py; filter_test_bie.py ->
safety_filter/run_season_calib.py; stage_real.py -> experiments/main_comparison.py; stage_more.py ->
experiments/more_seasons.py; stage2b.py -> experiments/calibration_baselines.py; stage_bie.py ->
experiments/calibration_blend.py; stage_groups.py -> experiments/calibration_groups.py; bie_noise.sh ->
experiments/calibration_noise.sh; results/*.jsonl -> results/filter/. Stages 1, 2, 2c and K4 used an
earlier version of the runner and earlier RL agents; they are not part of the paper or this repository.
References to reports/, overnight/cycles/ and PROTOCOL are to notes outside this repository.
-->

# GreenLeaf physics safety filter: exploratory test plan

Frozen 2026-11-04, before any run of the variants below. Changing a rule after results exist must be written here with the reason.

**Status: exploratory.** Nothing here is registered or a held-out claim. Amsterdam 2010 already selected the agents' checkpoints, and the AGC2 weather has been used in earlier spikes. Any paper result needs a separate registered run on unspent weather (2014-18 under PROTOCOL v5), and that needs Aarmen's yes.

## Question
Can a filter built from the known greenhouse model cut humidity violations for the rules and your trained agents without giving up more than 5% of yield?

## What changed from the first prototype
The prototype lost 11-24% of yield on one 2012 season. Two possible causes get tested:
1. It forced humidity under 85%, an unsourced default limit that real growers exceed 42-63% of the time. Test 85% and 90%.
2. Once triggered, it optimized for economics instead of staying close to what the controller wanted, and it looked only 15 minutes ahead. Test a minimum-change objective and a 1-hour lookahead.

## Weather: only these
- Amsterdam 2010, start days 45, 59, 75 (the validation split)
- AGC2 logged weather ("Wag" 2020), start days 5, 45, 85
- Never 2012 or 2013 through the simulator. Never `allow_test=True`. Never 2014-2023 for Amsterdam, London, Abbotsford.

## Stage 1: pick a variant (rb_tuned only, Amsterdam 2010, start day 59)
Unfiltered baseline, then 8 variants: limit {85, 90} x objective {econ, mindev} x lookahead {1 step, 4 steps}.
Metrics, per variant against the unfiltered run: reduction in hours above that variant's limit, and yield loss.

Selection, in order:
1. Keep variants with at least 50% reduction and at most 5% yield loss.
2. Prefer limit 85, then the larger reduction, then the shorter lookahead.
3. If none qualify, take the smallest yield loss among those with at least 50% reduction. Continue only if that loss is at most 10%. Otherwise stop and report.

## Stage 2: test it (the selected variant only)
Controllers: rb_tuned; residual_w05 seeds 100, 101, 102; ppo seeds 0, 1, 2. All 6 seasons above.
Conditions: raw and filtered, each with the exact model, and each under plant mismatch.
Mismatch: leakage x2, cover heat exchange x0.8, vent area x0.8, boiler x0.85, cEvap4Day x1.3, plus a +1.5 C outdoor bias seen only by the filter's model.

## Kill rules (any one stops the filter angle)
Pooled over seasons, per controller. rb_tuned and residual_w05 gate; PPO is reported but does not gate, since it already grows about 30% less fruit.
- **K1, yield:** with the exact model, less than 50% reduction in hours above the limit, or more than 5% yield loss.
- **K2, side effects:** temperature plus CO2 hours rise by more than 10% and by more than 10 h.
- **K3, mismatch:** under mismatch, the fractional reduction falls below half of what it was with the exact model.
- **K4, simpler answer (stage 3):** rules retuned with a violation penalty (200 trials, train-split weather only) get within 10% of the filter's hours at equal or better yield.

## If it passes
Stage 3 is K4. Then, before any paper:
- Source the humidity limit from at least two papers read in full, not snippets. Otherwise report a sweep and defend no single limit.
- Add a state estimate, since a real greenhouse doesn't measure all 28 states.
- Register a confirmatory run on unspent weather.

## Known weaknesses
- The filter sees the full state, and exact weather for its lookahead except under mismatch.
- Mismatch only changes parameters, not the model's structure.
- Profit is reported only across a fruit price range, and it isn't a gating metric, because All off wins at realistic prices.

Results go in `results/`. Stage logs are append-only.

## Changes after freezing
- 2026-11-04, before any variant ran: removed the weakness saying rule controllers see shifted parameters. The judge showed `predict` never reads `ctx.p` (all-NaN p gives identical hours).
- 2026-11-04, after seeing only the unfiltered stage 1 baseline (no filter variant had reported): a limit can't qualify unless the unfiltered controller has at least 20 h above it on the same seasons. The baseline showed 244.25 h above 85% but only 2.25 h above 90%, so a percentage cut at 90% would be an empty pass. Variants at a limit that fails this are still recorded and reported, but can't be selected or pass K1. If this applies to 90%, also report the gap with real logs, where teams had 174-431 h above 90% per 60 days.
- 2026-11-04, same point: stage 2 adds lag_b100 seeds 200, 201, 202 unfiltered, as a reported comparison that doesn't gate, so the filter can be compared directly with the learned constraint. This needs `lag_b100` added to `RUNS` in `filter_test.py` before stage 2.
- Correction to the two notes above: when they were written, the first filter variant had just finished (results file had 2 lines), but its result had not been read by anyone. The rule change was based only on the baseline row.

## Stage 1 result and selection (2026-11-05)
7 of 8 variants finished before a stop requested by Aarmen. The unfinished one (90, econ, 4 steps) can't change the choice, because 90% fails the 20 h rule (baseline 2.25 h).
Selected by the frozen rule: limit 85, mindev, 4 steps (98.0% reduction, yield +6.1%). The profit cost at simulator prices was EUR 2.79/m2.
Exploratory addition, decided before stage 2 ran: also run limit 85, mindev, 1 step (92.5%, yield +1.1%, profit cost EUR 0.58/m2) on the same seasons. It doesn't gate, and it can't replace the selected variant under this plan.

## Stage 2b: realistic-information stress test (exploratory, written before any run of it)
Stage 2 gives the filter the true 28 states and a mild mismatch. reports/7-skeptic.md measured one-step errors on real AGC2 logs of 1.85-4.30 K and 3.8-10.25 RH points, with a crude observer. So:
- **Observer:** the filter only sees measured air temperature, humidity, and CO2. Its other states come from running its own nominal model forward with the controls actually applied.
- **Sensor noise:** AR(1) errors in what it measures, SD 2 K and 5 RH points, correlation 0.98 per 15 minutes, fixed seed. These are sized to the upper part of the real errors and treated as noise on measured values.
- **Plants the filter doesn't know:** Bie's printed calibrated parameters; lamp cooling off with lamp cap 142.5; initial LAI 0.8; and the stage 2 combined mismatch.
Controllers: rb_tuned, residual seed 100. Seasons: Amsterdam 2010 day 59; AGC2 day 45. Selected filter (85, mindev, 4 steps). Unfiltered baselines use the same plant.

**K5 (kill if):** with observer and sensor noise on the worst plant, reduction in hours above 85% falls below half of the exact-model reduction, or yield loss exceeds 5%, for either controller. If K5 fires only on the plant where the baseline is already low (under 20 h), report it rather than kill.
Hardest conditions run first.

## Stage 2c: what protection does the energy buy? (exploratory, written before any run of it)
Prior-work check (overnight/cycles/02/filter-prior-work.md): horticultural concern is free water on the plant, not RH>85 itself, and growers control on humidity deficit. So:
- New metrics on every run from here: **wet canopy hours** (air vapour pressure above saturation at canopy temperature x[4], a condensation proxy), hours with VPD below 0.2, 0.3, 0.5 kPa, and saved hourly traces.
- New arm, **target wet**: same min-change filter, 4 steps, but the humidity term is condensation on the canopy (margin 50 Pa), not RH>85. Temperature and CO2 terms are unchanged.
- Compare unfiltered, RH filter, and wet filter on rb_tuned and residual seed 100, Amsterdam 2010 day 59 and AGC2 day 45.
**Decision metric:** energy cost per wet hour avoided, and extra energy compared with the RH filter.
**If:** the wet filter removes at least 80% of wet hours for at most half the RH filter's extra energy, the study should target wetness and report RH as secondary. If unfiltered wet hours are near zero on these seasons, this target has no substance in this simulator; report that as a limit (the model has no leaf-wetness state, so this is only a proxy).
No claim of disease protection can follow from this: the simulator has no disease model.

## K4, stronger retune (written before it ran)
The first retune (cycles/01/rules) used 60 trials and one penalty, so it's weak. Run a second: all 18 parameters, 150 trials, train weather only, heavier penalty 0.25 EUR/m2 per violation hour, new sampler seed, starting from the unfiltered rules and the first retune's best. Score on the same 6 seasons. K4 fires if either retune gets within 10% of the filter's hours at equal or better yield. Also report energy.

## Bug found 2026-11-05, 04:40, and fixed
`evaluate._grade` read `raw.p` after the vec env's season-end auto-reset, so learned controllers under mismatch or an unknown plant had their costs computed at nominal parameters. Rule-based rows were correct. Hours and yield were never affected, because they come from recorded states. Exact-model rows weren't affected. Fixed in all three scripts; new rows carry `grade_fixed`. Old learned mismatch rows are corrected in analysis (heat cost x 0.85, boiler is the only cost-related parameter in the stage 2 mismatch). Learned rows on Bie's or other plants from stage 2b get rerun if any exist. Earlier 'residual raw mm energy +1.45' was this artifact.

## Bie diagnosis (written 2026-11-05, before any run; exploratory; Amsterdam 2010 d59 and Wag 2020 d45 only, both already used)
Script: filter_test_bie.py (adds lower RH limit --lowrh, filter model --model-frac, logs t_lt15, t_gt34, rh_lt50, and SafeGreenRL-style scaled cost). Results: results/bie.jsonl. Plant always Bie, h=4, limit 85. Scored on total graded hours (RH outside 50-85, T outside 15-34, CO2), against the unfiltered baseline on the same season and controller.
Arms: a1 exact state, nominal model. a2 a1 + lowrh. a3 exact state, Bie model. a4 a3 + lowrh. a5 observer + noise, Bie model, lowrh. m25/m50/m75 observer + noise, lowrh, model 25/50/75% toward Bie, only if a4 passes.
Controllers: rb_tuned first, then residual:seed100. PPO not run.
Pass for an arm: total graded hours down at least 50% against baseline, in both seasons, for both controllers, with yield loss at most 5%.
Conclusions fixed in advance:
- a4 fails: wrong model is not the cause. The filter stops being the headline.
- a4 passes, a5 fails: sensors break it, not the model.
- a5 passes, m75 fails: works only with a nearly exact model; paper becomes a warning about transfer.
- a5 and m50 pass: "works if roughly calibrated"; report how far m25 gets.
- a1 passes while the old observer + noise row fails: the earlier result was caused by sensors.
Limits: Bie's parameters are one calibration, read in part. a3 to a5 give the filter the plant's own parameters, a best case. Linear mixing of parameters is a convenience, not a physical path. Two seasons, one RL seed.
Added 2026-11-05 after a4 and a5 passed for rules, before any run of these: sensor noise check. a5 on rules looked too good for 5-point RH noise, so rerun with noise seeds 11 and 23, and with noise x2 (seed 7), on Amsterdam d59, logging realised noise SD and hours with true RH>85 while measured RH<84. If the realised SD is far below 5 or 2, or the result is seed-specific, the a5 pass is withdrawn. results/bie_noise.jsonl. Partial-model arms started for rules.

## One-group calibration test (written 2026-11-05 12:40, before any run; exploratory)
Why: the linear blend toward Bie's settings mixes all 12 together, including floor area, which nobody is unsure of. This asks which group of settings the filter actually needs right.
Groups, in per-m2 terms so floor area doesn't silently change actuator size: G geometry (46 floor, 47 cover area, 48 and 49 heights); V cover heat exchange and vents (50, 56, 59, 60 leakage, 61); A actuators per m2 (boiler 108/46, CO2 109/46, lamps 172). Roof area p55 is not in Bie's set and stays default throughout.
Arms, rules, Bie plant, observer + noise + lower RH limit (as a5), Amsterdam d59 and Wag d45: one group wrong (V,A right; G,A right; G,V right) and one group right (G; V; A). Same seasons as before, no new weather.
Reading, fixed now: a group matters if leaving it wrong loses the pass (total hours down at least 50% from 120.5 and 128) in either season. A group is enough on its own if getting only it right passes in both. Report energy and yield next to hours. Mixed or no single group: say so.
Limit: 3 groups is my grouping, not a calibration procedure; 2 seasons; one plant.

## Base controller at real prices (written 2026-11-05 13:45, after seeing the fair-price PPO-Lag rows; exploratory, ordering disclosed)
Prompt: PPO-Lag trained at NL f1.6 or AGC f2.5 lights 70-76% of steps and holds CO2 near 1,180 ppm; sim-tuned rules light 6%. Yield 15.4-16.4 vs 9.36 kg; profit vs rules+filter about 0 at NL f1.6 and +8 to +9.6 at AGC f2.5; hours 87-128 vs 17. So the sim-tuned rules are the wrong baseline at real prices.
Runs, 6 seasons, filter h4 mindev limit 85, nominal (mismatch after if time): (1) rules tuned at AGC2 2020 tariffs (repo config rule_based_tuned_agc2020, fruit 1.6), raw and filtered; (2) PPO-Lag NL-trained s0 and AGC-trained s0, filtered (raw rows exist).
Reading, fixed now, at each arm's training price:
- RL keeps a role: PPO-Lag+filter hours within 10% or 5 h of the matched rules+filter, and profit at least 1 EUR/m2 higher. Headline: RL does the economics, the filter does the safety.
- Filter erases it: lead under 1 EUR/m2, as with the residual. Same story as before.
- Filter can't contain it: PPO-Lag+filter hours more than twice the matched rules+filter. Report as a limit.
Matched rules at AGC f2.5: there are no rules tuned at fruit 2.5; agc2020 rules (fruit 1.6) are used and that is stated.
Result, seed 0 (15:06): at AGC f2.5 the AGC-trained PPO-Lag + filter has 4.5 h vs 42.5 for agc-rules + filter and +4.41 EUR/m2: "RL keeps a role". At NL f1.6 the NL-trained arm + filter is -0.78: "filter erases it" there (the raw NL arm was already about even with agc-rules raw).
Addendum (written 15:10, before running): seeds 1 and 2 of both arms + filter, nominal; then mismatch for agc-rules raw and filtered and both seed-0 arms filtered. The verdict above is restated on the mean of 3 seeds, paired on season, with seed SD; it stands only if the 3-seed mean still meets the same rule.

## More seasons (written 2026-11-06, before any run; exploratory)
Why: the real-price verdict rests on 6 reused seasons. Add 6 new 60-day windows on the only allowed non-training weather: Amsterdam 2010 starts 15, 30, 90, 105 and Wag 2020 starts 25, 65 (Wag file is 150 days, so starts stop at 90). No training years: the PPO-Lag agents trained on Amsterdam 2008/09/11, so scoring there would flatter them.
Arms, all nominal, filter h4 mindev limit 85: agc-rules raw and filtered; PPO-Lag agc25 s0-s2 and nl16 s0-s2, filtered. 48 jobs. Script stage_more.py, results/stage_more.jsonl.
Reading, fixed now, same rule as the real-price section, on the 3-seed mean over the 6 new seasons, paired on season:
- Holds: AGC-trained + filter at AGC f2.5 at least 1 EUR/m2 ahead of agc-rules + filter, with hours within 10% or 5 h of it (or fewer).
- Report the new-season and pooled 12-season numbers side by side, whatever they show. If the new seasons fail the rule, the headline is restated as holding on the original 6 only, and the paper says so.
Limits: Amsterdam starts 15, 30, 90, 105 and Wag 25 fall outside the agents' training start days (45-75), so they test some shift in season timing too. Windows from the same year overlap, so these are not independent years.

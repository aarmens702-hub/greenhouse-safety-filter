<!--
Copied verbatim from the project's working notes (cycles/05/ppolag/PLAN.md, written 2026-11-05).
Pre-registration for the PPO-Lagrangian agents. train_ppolag.py and ppolag_eval.py are now
controllers/train_ppolag.py and controllers/eval_ppolag.py; the "fair-price rerun" arms are trained and
scored by controllers/run_ppolag.py. The first section's agents (trained at simulator prices, and on
Bie's plant) are reported in results/numbers.json (T5, T7) but not in the paper.
-->

# PPO-Lagrangian vs physics filter (written 2026-11-05 before any full training; exploratory)
Requested by Aarmen after reading SafeGreenRL (SSRN 6989405, read in full; preprint, code on request). Own implementation: train_ppolag.py, choices the paper leaves open are listed in its header.
Train: d=0.01 (their 57.6 seasonal budget), 2M steps, 4 envs, seeds 0-4, Amsterdam 2008, 2009, 2011. Then 2 seeds on Bie's plant (--bie). Final checkpoint, no selection. No sealed weather, no 2012.
Score (ppolag_eval.py, same grading as the filter): Amsterdam 2010 d45, 59, 75; Wag 2020 d5, 45, 85, nominal and mismatch; Bie plant Amsterdam d59, Wag d45.
Compare with rules+filter 4-step, residual+filter 4-step, residual+filter 1-step on the same seasons. Report hours, yield, energy, profit at sim f1.6, NL f1.6, AGC f2.5.
Reproduction check, stated first: mean seasonal cost at or under 57.6 and well under rules. Their profit about -1 at sim prices is on different seasons (start days 1 to 291), so level is only a loose guide. If cost misses the budget, this is not their controller and the comparison is reported as weak.
Outcomes fixed in advance (nominal, 5 seeds, against rules+filter 4-step):
- PPO-Lag wins: hours within 10% or 5 h of the filter and higher profit at all three price sets. The filter's case on the default plant is gone.
- Filter wins: hours no worse and higher profit at NL f1.6 and AGC f2.5. Sim prices reported, not hidden.
- Split: result depends on price; paper says so and gives break-even fruit price.
Bie: PPO-Lag trained on the default plant is compared with the wrong-model filter (a1, old observer rows), the fair uncalibrated pair. PPO-Lag trained on Bie's plant is compared with the right-model filter (a4, a5), the fair calibrated pair. Calibrating a filter needs parameters; retraining needs 2M simulated steps on the plant. Both are stated.
Limits: 5 seeds, 6 seasons reused, unverified match to their code, d=0.01 only.

# Fair-price rerun (written 2026-11-05 11:40, after seeing the first comparison; exploratory, and this ordering is disclosed)
Why: the first runs trained on simulator prices (heat .09, elec .30, CO2 .30) and were scored at NL f1.6 and AGC f2.5, prices they never saw. A reviewer would reject that. Also the multiplier stayed 0.96-1.07 and mean step cost 0.004-0.007 under d=0.01: the budget never bound, so those runs behave like fixed-penalty PPO at about lambda 1.
Arms, all 2M steps, seeds 0-2, Amsterdam 2008, 2009, 2011, final checkpoint, profit-only reward:
- NL: reward at NL f1.6 prices, paper's dual settings (lr .05, d .01).
- AGC: reward at AGC f2.5 prices (fruit 2.5 is an estimate), same dual settings.
- TIGHT: reward at NL f1.6, dual lr 2.0, d .001 (5.76 per season), trying to reach the filter's hours. These settings are mine, not the paper's.
Score: same 6 seasons, nominal and mismatch. Rules stay tuned at sim prices and the filter unchanged, which favours PPO-Lag. Each arm is compared with rules+filter 4-step at the price it trained on.
Outcomes, nominal, mean of 3 seeds (paired on season; report seed SD):
- PPO-Lag wins at a price: hours within 10% or 5 h of the filter, and higher profit at that price.
- PPO-Lag trades: more profit but more hours. Report EUR per hour avoided; the paper says the filter is a different point on the trade-off, not better.
- Filter wins at a price: hours no worse and higher profit at that price.
The earlier "Filter wins" verdict stands only if it also holds at NL f1.6 and AGC f2.5 with the arm trained at that price. Otherwise it is withdrawn.
Limits: 3 seeds; rule outcomes depend on the fruit estimate; 6 reused seasons; own implementation.

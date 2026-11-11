# Figures and table

Each script reads only `results/numbers.json` and raw rows whose sha256 is checked against
`docs/FROZEN.md` on every load (`common.py`), and writes PDF and PNG files to `out/`.

| Output | Script | Source |
|---|---|---|
| Figure 1, system diagram | `fig1_system.tex` (`tectonic -X compile fig1_system.tex`) | none |
| Figure 2, filter in action | `fig2_in_action.py` | `data/fig2_*.npz` from `fig2_trace.py`; checked against `results/filter/stage_real.jsonl` |
| Figure 3, hours and break-even | `fig3_performance.py` (+ `hours.py`) | numbers.json T4; `stage_real.jsonl`, `ppolag/scores.jsonl` |
| Figure 4, calibration | `fig4_calibration.py` | numbers.json T2, T3; `stage2b.jsonl` |
| Table 1 | `table1.py` (writes `out/table1.tex`, `out/table1_preview.tex`) | numbers.json T4 |

Each Python script runs in a few seconds. `fig2_trace.py` reruns the two Figure 2 seasons
(rules alone: seconds; rules + filter: about 10 minutes) and overwrites `data/`.

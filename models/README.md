# Trained agents

The six PPO-Lagrangian agents the paper reports (`d0.01_s{0,1,2}_agc25`, `d0.01_s{0,1,2}_nl16`) are
about 11 MB each, so they are not committed. Download them from the
[v1.0.0 release](https://github.com/aarmens702-hub/greenhouse-safety-filter/releases/tag/v1.0.0)
(`SHA256SUMS.txt` there lists the checksums) and unzip each one here, so that each agent has

    models/<tag>/ckpt_final.zip
    models/<tag>/vecnorm_final.pkl

Or set `MODELS_DIR` to another folder with the same layout. To train them again instead, see
`controllers/run_ppolag.py`. Each run's `done.json` (final multiplier, wall time) and `lag.csv`
(multiplier log) are kept in `results/ppolag/runs/`.

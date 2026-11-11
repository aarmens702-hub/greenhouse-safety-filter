"""Calibration study: rules on a greenhouse with Bie et al.'s calibrated parameters, filtered by a model that is
default, partly calibrated, or calibrated in some parameter groups only. Appends one JSON row.

    python -m safety_filter.run_season_calib --ctrl rb_tuned --plant bie --loc Amsterdam --year 2010 --start 59 \
        --limit 85 --h 4 --obj mindev --lowrh --observer --noise --model-frac 0.5 --out outputs/calib.jsonl

--model-frac F moves the filter's 12 parameters F of the way from default to Bie's values.
--right G,V,A sets whole groups to Bie's values instead (G geometry, V cover heat exchange and vents,
A actuators per m2), others default. --lowrh also keeps predicted RH above 51%. --noise-seed and
--noise-mult change the sensor noise. Other options as in safety_filter.run_season. Rules only.
A filtered season takes 7-30 minutes (median about 20) on one core.

This is the paper's filter_test_bie.py (computation unchanged); it shares the season loop with run_season.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np

from safety_filter.filter import BIE, Filter, climate
from safety_filter.run_season import build_parser, change_plant, check_weather, install, season_row, simulate, write


def run(a, on_decision=None):
    assert a.plant == "bie"
    if not a.ctrl.startswith("rb_"): sys.exit("the calibration runner takes rule-based controllers only")
    check_weather(a)
    def arm(raw):
        f = Filter(raw, a.limit, a.obj, a.h, a.mismatch, observer=a.observer, noise=a.noise, noise_seed=a.noise_seed,
                   noise_mult=a.noise_mult, lowrh=a.lowrh, on_decision=on_decision) if a.limit else None
        if f:
            for i, v in BIE.items(): f.p[i] = f.p[i] + a.model_frac*(v-f.p[i])
        if f and a.right is not None:
            assert a.model_frac == 0
            rg = set(filter(None, a.right.split(","))); assert rg <= {"G", "V", "A"}
            boil, co2 = f.p[108]/f.p[46], f.p[109]/f.p[46]      # actuators per m2, so floor area doesn't resize them
            for g, ix in (("G", (46, 47, 48, 49)), ("V", (50, 56, 59, 60, 61))):
                if g in rg:
                    for i in ix: f.p[i] = BIE[i]
            if "A" in rg: boil, co2, f.p[172] = BIE[108]/BIE[46], BIE[109]/BIE[46], BIE[172]
            f.p[108], f.p[109] = boil*f.p[46], co2*f.p[46]
        change_plant(raw, a)
        if f: install(raw, f)
        return f
    infos, raw, plant, f, t0 = simulate(a, arm)
    x = np.stack([i["x"] for i in infos])
    rh = np.array([climate(r)[2] for r in x]); Ta = x[:, 2]; ppm = np.array([climate(r)[0] for r in x])
    sgr = float((np.maximum(0, Ta-34)+np.maximum(0, 15-Ta)).sum()/15 + (np.maximum(0, ppm-1600)+np.maximum(0, 300-ppm)).sum()/2500 + (np.maximum(0, rh-85)+np.maximum(0, 50-rh)).sum()/15)
    sn = np.array(f.seen) if f and f.seen else np.zeros((1, 4))
    extra = dict(right=a.right, noise_seed=a.noise_seed, noise_mult=a.noise_mult, sd_drh=float((sn[:, 1]-sn[:, 0]).std()), sd_dt=float((sn[:, 3]-sn[:, 2]).std()),
                 true_gt85_meas_lt84_h=float(((sn[:, 0] > 85) & (sn[:, 1] < 84)).sum()*.25), true_gt84_h=float((sn[:, 0] > 84).sum()*.25), true_rh_p50=float(np.median(sn[:, 0])),
                 true_rh_p99=float(np.percentile(sn[:, 0], 99)), lowrh=a.lowrh, model_frac=a.model_frac, t_lt15=float((Ta < 15).sum()*.25), t_gt34=float((Ta > 34).sum()*.25),
                 rh_lt50=float((rh < 50).sum()*.25), sgr_cost=sgr)
    return {**extra, **season_row(a, infos, raw, plant, f, t0)}, infos, f



def main(argv=None):
    ap = build_parser()
    ap.add_argument("--noise-seed", type=int, default=7); ap.add_argument("--noise-mult", type=float, default=1.); ap.add_argument("--lowrh", action="store_true")
    ap.add_argument("--model-frac", type=float, default=0.)   # fraction of the way from default to Bie in the filter's model
    ap.add_argument("--right", default=None)                  # groups at Bie values in the filter's model, e.g. G,V; others default
    a = ap.parse_args(argv)
    row, _, _ = run(a)
    write(row, a.out)


if __name__ == "__main__":
    main()

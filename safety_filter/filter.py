"""The physics-model predictive safety filter.

Every 15-minute step the controller proposes six actuator settings u_prop (boiler, CO2 dosing,
thermal screen, vents, lamps, blackout screen; all in [0, 1]). The filter holds u_prop for h steps
(h = 4, one hour, in the paper) on its own copy of the GreenLight model and sums a predicted
violation V over those steps, using internal margins inside the graded limits (RH 85 -> 84%,
temperature 15/34 -> 15.5/33.5 C, CO2 300/1600 -> 320/1580 ppm). If V is zero the action passes
unchanged. Otherwise a coordinate search (vents, boiler, thermal screen, blackout screen, lamps,
CO2; steps of +-0.1, 0.25, 0.5 around u_prop; two passes) picks the action minimising

    1000 * V(u) + ||u - u_prop||^2        (objective "mindev", used in the paper)

There is no guarantee: the lookahead is one hour, the search is approximate, and the filter's
model may differ from the plant.

Extracted unchanged in computation from the paper's run scripts (filter_test_real.py and its
calibration variant filter_test_bie.py); the variant's options are constructor arguments here.
"""
import time

import casadi as ca
import numpy as np

from greenleaf_core import metrics

# Bie et al.'s calibrated parameters (gl_gym parameter index -> value), used as the "real" plant
# in the calibration study.
BIE = {46: 14.982, 47: 72.632, 48: 3.0, 49: 5.211, 50: 4.675, 56: 1.101, 59: 0.110, 60: 2.37e-4, 61: 0.0185, 108: 4403.9, 109: 20.0, 172: 300.0}
# Plant mismatch (parameter index -> factor): leakage x2, cover heat exchange x0.8, vent area x0.8,
# boiler x0.85, cEvap4Day x1.3; plus an outdoor temperature bias seen only by the filter's model.
MISMATCH = {60: 2.0, 50: .8, 51: .8, 52: .8, 53: .8, 55: .8, 108: .85, 21: 1.3}
BIAS_T = 1.5


def climate(x):
    """(CO2 ppm, air temperature C, RH %) from the model state x."""
    T = x[2]; ppm = 1e6*8.3144598*(T+273.15)*(x[0]*1e-6)/(101325*44.01e-3)
    return ppm, T, min(100., 100*x[15]/(610.78*np.exp(17.2694*T/(T+238.3))))


class Filter:
    """Wraps one raw gl_gym env. Call it with the proposed action; it returns the applied action.

    raw: the env's unwrapped GreenLight env; the filter reads its state, weather and parameters.
    limit: RH limit in % (85 in the paper); the internal margin is limit - 1.
    obj: "mindev" (paper) or "econ" (an early variant: lowest predicted cost minus fruit value).
    h: lookahead steps. biased: add BIAS_T to the outdoor temperature the filter's model sees.
    observer, noise, noise_seed, noise_mult: what the filter knows of the state (see estimate).
    lowrh: also keep predicted RH above 51% (calibration study).
    on_decision: optional callback(filter, x, u_prop, u_applied) after every decision (logging only).
    """

    def __init__(s, raw, limit, obj="mindev", h=4, biased=False, *, observer=False, noise=False,
                 noise_seed=7, noise_mult=1., lowrh=False, on_decision=None):
        s.raw, s.limit, s.obj, s.h, s.bias = raw, limit, obj, h, BIAS_T if biased else 0.
        s.observer, s.noise, s.noise_mult, s.lowrh, s.on_decision = observer, noise, noise_mult, lowrh, on_decision
        s.p = np.array(raw.p, dtype=np.float64).copy()        # the filter's model starts nominal
        pr = metrics.PRICES; s.pr = pr; fl = s.p[46]
        s.hf = s.p[108]/fl*900/3600*1e-3*pr["heat"]; s.ef = s.p[172]*900/3600*1e-3*pr["elec"]; s.cf = s.p[109]/fl*900*1e-6*pr["co2"]
        s.evals = s.fails = s.hits = 0; s.tmax = 0.
        s.xhat = None; s.u_prev = None; s.rng = np.random.default_rng(noise_seed); s.seen = []; s.dT = s.dRH = 0.; s.est_err = []

    def estimate(s):
        """What the filter believes. Exact state unless observer: then measured T, vapour pressure and CO2 replace its model's, with optional AR(1) sensor noise."""
        true = np.array(s.raw.x, dtype=np.float64)
        if not s.observer: return true
        if s.xhat is None: s.xhat = true.copy()
        elif s.u_prev is not None:
            w = s.raw.weather_data; d = np.array(w[max(s.raw.timestep-1, 0)], dtype=np.float64); d[1] += s.bias
            try:
                xn = s.raw.F(x0=ca.DM(s.xhat), u=ca.DM(s.u_prev), p=ca.vertcat(ca.DM(d), s.p))["xf"].full().ravel()
                if np.all(np.isfinite(xn)): s.xhat = xn
            except Exception: s.fails += 1
        if s.noise:
            s.dT = .98*s.dT + s.rng.normal(0, s.noise_mult*2.0*np.sqrt(1-.98**2)); s.dRH = .98*s.dRH + s.rng.normal(0, s.noise_mult*5.0*np.sqrt(1-.98**2))
        _, T, rh = climate(true); Tm = T+s.dT; rhm = np.clip(rh+s.dRH, 1, 100); s.seen.append((rh, rhm, T, Tm))
        x = s.xhat.copy(); x[0] = true[0]; x[2] = Tm; x[15] = rhm/100*610.78*np.exp(17.2694*Tm/(Tm+238.3))
        s.xhat = x.copy(); s.est_err.append(float(np.max(np.abs(x-true)/(np.abs(true)+1.))))
        return x

    def V(s, x):
        """Predicted violation of one state against the internal margins."""
        ppm, T, rh = climate(x)
        if s.lowrh:
            low = max(0, 51-rh)/10
            return low + max(0, rh-(s.limit-1))/10 + max(0, 15.5-T)/5 + max(0, T-33.5)/5 + max(0, 320-ppm)/500 + max(0, ppm-1580)/500
        return max(0, rh-(s.limit-1))/10 + max(0, 15.5-T)/5 + max(0, T-33.5)/5 + max(0, 320-ppm)/500 + max(0, ppm-1580)/500

    def roll(s, x, u):
        """Hold u for h steps on the filter's model. Returns (violation, cost minus fruit value), or None if the solver fails."""
        v = c = 0.; w = s.raw.weather_data; t = s.raw.timestep; x0 = x
        for i in range(s.h):
            d = np.array(w[min(t+i, len(w)-1)], dtype=np.float64); d[1] += s.bias
            s.evals += 1
            try: xn = s.raw.F(x0=ca.DM(x), u=ca.DM(u), p=ca.vertcat(ca.DM(d), s.p))["xf"].full().ravel()
            except Exception: s.fails += 1; return None
            if not np.all(np.isfinite(xn)): s.fails += 1; return None
            v += s.V(xn); x = xn
        c = s.h*(u[0]*s.hf+u[4]*s.ef+u[1]*s.cf) - s.pr["fruit"]*(x[25]-x0[25])*1e-6/s.pr["dmfm"]
        return v, c

    def __call__(s, u_nom):
        t0 = time.perf_counter(); x = s.estimate(); u_nom = np.array(u_nom, dtype=np.float64)
        r = s.roll(x, u_nom)
        if r is None or r[0] == 0:
            if s.on_decision: s.on_decision(s, x, u_nom.copy(), u_nom.copy())
            s.u_prev = u_nom; return u_nom
        s.hits += 1
        def J(u, r):
            return 1e3*r[0] + (r[1] if s.obj == "econ" else float(np.sum((u-u_nom)**2)))
        u, best = u_nom.copy(), J(u_nom, r)
        for _ in range(2):
            for k in (3, 0, 2, 5, 4, 1):      # vents, boiler, thermal screen, blackout screen, lamps, CO2
                grid = (0, .25, .5, .75, 1.) if s.obj == "econ" else np.clip(u_nom[k]+np.array([-.5, -.25, -.1, .1, .25, .5]), 0, 1)
                for g in grid:
                    uc = u.copy(); uc[k] = g; rc = s.roll(x, uc)
                    if rc is not None and J(uc, rc) < best-1e-9: best, u = J(uc, rc), uc
        s.tmax = max(s.tmax, time.perf_counter()-t0)
        if s.on_decision: s.on_decision(s, x, u_nom.copy(), u.copy())
        s.u_prev = u
        return u

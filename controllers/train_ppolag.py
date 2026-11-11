"""PPO-Lagrangian after SafeGreenRL (Sharma & Mourya, SSRN 6989405), own implementation; their code is on request only.
From the paper: profit-only reward; step cost = scaled temp + CO2 + RH violations; same PPO base settings as the repo's ppo.yml;
cost critic with the value net's layout; initial multiplier 1.0; dual lr 0.05; cost-value coefficient 0.5; budget as mean step cost; 2M steps, 4 envs.
Our choices where the paper is silent: advantage (A_r - lam*A_c)/(1+lam) as in Ray et al. 2019; lam += 0.05*(mean rollout step cost - d), floored at 0;
cost critic has its own Adam at the policy's lr; no bootstrap for cost at season end; final checkpoint, no selection.
Trains on Amsterdam 2008, 2009, 2011 (SafeGreenRL also used 2010; here 2010 is evaluation weather).
Writes checkpoints, the multiplier log (lag.csv) and done.json to models/<tag>/ (or $MODELS_DIR/<tag>/).
Usage: python controllers/train_ppolag.py SEED D [STEPS] [--bie] [--price nl16|agc25] [--dual X]
Paper agents: SEED 0-2, D 0.01, --price agc25 or --price nl16 (2M steps, about 70 min each on one laptop core)."""
import functools, json, os, sys, time
from pathlib import Path
import numpy as np, torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from greenleaf_core import envs, ppo as train
from stable_baselines3 import PPO
th.set_num_threads(1)
BIE = {46:14.982,47:72.632,48:3.0,49:5.211,50:4.675,56:1.101,59:0.110,60:2.37e-4,61:0.0185,108:4403.9,109:20.0,172:300.0}
envs.REWARD_PRICES.update(nl16={"fruit_price": 1.6, "heating_price": .0429, "elec_price": .0889, "co2_price": .08}, agc25={"fruit_price": 2.5, "heating_price": .03, "elec_price": .06, "co2_price": .08})
av = sys.argv[1:]; price, dual = "sim", .05
if "--price" in av: i = av.index("--price"); price = av[i+1]; del av[i:i+2]
if "--dual" in av: i = av.index("--dual"); dual = float(av[i+1]); del av[i:i+2]
args = [a for a in av if a != "--bie"]; on_bie = "--bie" in av
seed, d = int(args[0]), float(args[1]); steps = int(args[2]) if len(args) > 2 else 2_000_000
tag = f"d{d}_s{seed}" + ("_bie" if on_bie else "") + ("" if price == "sim" else f"_{price}") + ("" if dual == .05 else f"_dual{dual}") + ("" if steps == 2_000_000 else f"_n{steps}")
RD = Path(os.environ.get("MODELS_DIR", ROOT/"models"))/tag; RD.mkdir(parents=True, exist_ok=True)
def make(i):
    e = envs.make_env("ppo", seed=seed+i, season_length=60, penalty_weight=0.0, price_regime=price)
    if on_bie:
        raw = e.unwrapped; orig = raw.reset
        for k, v in BIE.items(): raw.p[k] = v
        def reset(*a, **kw):
            o = orig(*a, **kw)
            for k, v in BIE.items(): raw.p[k] = v
            return o
        raw.reset = reset
    return e
venv = VecNormalize(VecMonitor(DummyVecEnv([functools.partial(make, i) for i in range(4)])), gamma=train.ppo_yml()["gamma"], **train.VECNORMALIZE)
kw = train.ppo_yml()
for k in ("total_timesteps", "n_envs", "policy"): del kw[k]
pk = {k: train.from_yml(k, v) for k, v in kw.pop("policy_kwargs").items()}
class PPOLag(PPO):
    def setup_lag(s):
        obs = s.observation_space.shape[0]; L = []; n = obs
        for w in pk["net_arch"]["vf"]: L += [th.nn.Linear(n, w), pk["activation_fn"]()]; n = w
        s.cnet = th.nn.Sequential(*L, th.nn.Linear(n, 1)); s.copt = th.optim.Adam(s.cnet.parameters(), lr=2e-5, amsgrad=True)
        s.lam = 1.0; s.cb, s.db = [], []; s.log = open(RD/"lag.csv", "a"); s.t0 = time.time()
    def train(s):
        c = np.array(s.cb, dtype=np.float32); dn = np.array(s.db, dtype=np.float32); s.cb, s.db = [], []
        B = s.rollout_buffer; O = th.as_tensor(B.observations, dtype=th.float32); T, N = c.shape
        with th.no_grad():
            V = s.cnet(O.reshape(T*N, -1)).reshape(T, N).numpy(); Vl = s.cnet(th.as_tensor(s._last_obs, dtype=th.float32)).squeeze(-1).numpy()
        A = np.zeros((T, N), dtype=np.float32); g = np.zeros(N, dtype=np.float32)
        for t in reversed(range(T)):
            nv = Vl if t == T-1 else V[t+1]; nt = 1-dn[t]
            g = c[t]+s.gamma*nv*nt-V[t] + s.gamma*s.gae_lambda*nt*g; A[t] = g
        R = th.as_tensor(A+V).reshape(-1); X = O.reshape(T*N, -1)
        for _ in range(s.n_epochs):
            for ix in th.randperm(T*N).split(s.batch_size):
                loss = .5*((s.cnet(X[ix]).squeeze(-1)-R[ix])**2).mean()
                s.copt.zero_grad(); loss.backward(); th.nn.utils.clip_grad_norm_(s.cnet.parameters(), s.max_grad_norm); s.copt.step()
        J = float(c.mean()); s.lam = max(0., s.lam+dual*(J-d))
        B.advantages = (B.advantages - s.lam*A)/(1+s.lam)
        s.log.write(f"{s.num_timesteps},{J:.5f},{s.lam:.4f},{time.time()-s.t0:.0f}\n"); s.log.flush()
        super().train()
EX = ["log", "cnet", "copt", "cb", "db", "t0"]
class CostCB(BaseCallback):
    def __init__(s): super().__init__(); s.next = 250_000
    def _on_step(s):
        i = s.locals["infos"]; s.model.cb.append([x["cost_co2"]+x["cost_temp"]+x["cost_rh"] for x in i]); s.model.db.append(s.locals["dones"].copy())
        if s.num_timesteps >= s.next:
            s.model.save(RD/f"ckpt_{s.next}.zip", exclude=EX); venv.save(RD/f"vecnorm_{s.next}.pkl"); s.next += 250_000
        return True
m = PPOLag("MlpPolicy", venv, policy_kwargs=pk, seed=seed, device="cpu", **kw); m.setup_lag()
m.learn(steps, callback=CostCB())
m.save(RD/"ckpt_final.zip", exclude=EX); venv.save(RD/"vecnorm_final.pkl"); th.save(m.cnet.state_dict(), RD/"cnet.pt")
json.dump(dict(seed=seed, d=d, steps=steps, on_bie=on_bie, price=price, dual=dual, final_lam=m.lam, wall=time.time()-m.t0), open(RD/"done.json", "w")); print("TRAIN_DONE", tag, m.lam, flush=True)

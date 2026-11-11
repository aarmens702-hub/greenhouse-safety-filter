"""Train the paper's PPO-Lagrangian agents (prices nl16 and agc25, seeds 0-2, plus the exploratory
d0.001 dual-lr-2.0 arm), then score each unfiltered on the 6 original seasons, nominal and mismatch.
Skips training for agents whose models/<tag>/done.json exists. Usage: python controllers/run_ppolag.py WORKERS
Each training run is 2M steps (about 70 min on one laptop core); each scoring run takes seconds."""
import os, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; MODELS = Path(os.environ.get("MODELS_DIR", HERE.parent/"models"))
PY = sys.executable
env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
T = [(f"d0.01_s{s}_nl16", [str(s), "0.01", "--price", "nl16"]) for s in range(3)] + [(f"d0.01_s{s}_agc25", [str(s), "0.01", "--price", "agc25"]) for s in range(3)] + [(f"d0.001_s{s}_nl16_dual2.0", [str(s), "0.001", "--price", "nl16", "--dual", "2.0"]) for s in range(3)]
S6 = [("Amsterdam", 2010, 45), ("Amsterdam", 2010, 59), ("Amsterdam", 2010, 75), ("Wag", 2020, 5), ("Wag", 2020, 45), ("Wag", 2020, 85)]
jobs = []
for tag, ta in T:
    if not (MODELS/tag/"done.json").exists(): jobs.append(("train", tag, [str(HERE/"train_ppolag.py")]+ta))
    for loc, y, st in S6:
        for x in ([], ["--mismatch"]): jobs.append(("score", tag, [str(HERE/"eval_ppolag.py"), "--tag", tag, "--loc", loc, "--year", str(y), "--start", str(st)]+x))
run = []
def ready(j):
    return j[0] == "train" or (MODELS/j[1]/"done.json").exists()
while jobs or run:
    while len(run) < int(sys.argv[1]):
        j = next((j for j in jobs if ready(j)), None)
        if j is None: break
        jobs.remove(j); run.append((j, subprocess.Popen(["nice", PY]+j[2], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True))); print("start", j[0], j[1], " ".join(j[2][1:]), time.strftime("%X"), flush=True)
    for it in run[:]:
        if it[1].poll() is not None:
            run.remove(it); print("done", it[0][0], it[0][1], it[1].returncode, time.strftime("%X"), flush=True)
            if it[1].returncode:
                print(it[1].stderr.read()[-500:], flush=True)
                if it[0][0] == "train": jobs[:] = [j for j in jobs if j[1] != it[0][1]]; print("dropped scoring for", it[0][1], flush=True)
    time.sleep(15)
print("PPOLAG_DONE", flush=True)

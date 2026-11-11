"""A small process queue shared by the experiment scripts: runs one season per subprocess, `nice`, one thread each.
Every experiment script resumes: jobs whose row is already in its output file are skipped."""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")


def read_rows(path):
    path = Path(path)
    return [json.loads(l) for l in open(path)] if path.exists() else []


def run_queue(jobs, workers, runner="safety_filter.run_season", timeout_s=None, on_timeout=None):
    """jobs: list of (key, argv). Runs `python -m runner argv` for each, at most `workers` at once.
    A job running longer than timeout_s (monotonic clock, so a sleeping computer doesn't count) is killed
    and on_timeout(key) is called."""
    run = []; print("jobs", len(jobs), flush=True)
    while jobs or run:
        while jobs and len(run) < workers:
            k, a = jobs.pop(0)
            p = subprocess.Popen(["nice", sys.executable, "-m", runner] + a, cwd=ROOT, env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            run.append((k, time.monotonic(), p)); print("start", k, time.strftime("%X"), flush=True)
        for it in run[:]:
            k, t0, p = it
            if p.poll() is None and timeout_s and time.monotonic()-t0 > timeout_s:
                p.kill(); p.wait(); run.remove(it); print("timeout", k, time.strftime("%X"), flush=True)
                if on_timeout: on_timeout(k)
            elif p.poll() is not None:
                run.remove(it); print("done", k, p.returncode, time.strftime("%X"), flush=True)
                if p.returncode: print(p.stderr.read()[-500:], flush=True)
        time.sleep(10)

#!/usr/bin/env python3
"""
Imitation-regularised Double DQN experiment (reward fixed = delta).

Every run starts from the behaviour-cloned checkpoint of its seed
(models/exp_guided/B_bc/seed<S>/dqn_best.pt = the BC network at step 0) and is then
fine-tuned for --max-steps env steps with eps = 0.05 (constant) and the SAME
hyper-parameters in all arms. Only lambda differs:

  B_lam0   : lambda = 0    (standard Double DQN; imitation CE is only *logged*)
  C_lam0.1 : lambda = 0.1
  D_lam1   : lambda = 1.0

A (clone only) is the step-0 validation row that every run writes before any RL.

Refuses protected paths and non-empty output dirs; verifies BC / frozen / improved
checkpoint hashes before and after.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "models" / "frozen" / "dqn_best_val1320.pt"
IMPROVED = ROOT / "models" / "improved" / "dqn_best.pt"
FORBIDDEN = (ROOT / "models" / "frozen", ROOT / "models" / "improved", ROOT / "models" / "exp_guided")
DEMO_TRAIN = ROOT / "results" / "exp_demos" / "heuristic_demos_train.npz"
DEMO_VAL = ROOT / "results" / "exp_demos" / "heuristic_demos_val.npz"
BC_INIT = {s: ROOT / "models" / "exp_guided" / "B_bc" / f"seed{s}" / "dqn_best.pt" for s in (42, 43)}

# Fresh RL training-seed ranges (episode k uses base + k). Disjoint from test 0-99,
# validation 100000-100049, diag 200000/210000, demos 300000/310000 and from each other.
TRAIN_SEED_BASE = {42: 400_000, 43: 500_000}

ARMS = {"B_lam0": 0.0, "C_lam0.1": 0.1, "D_lam1": 1.0}
SEEDS = (42, 43)
DEFAULT_EVAL_STEPS = "1000,2500,5000,7500,10000"
EPSILON = 0.05


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_safe(path: Path) -> None:
    r = path.resolve()
    for b in FORBIDDEN:
        if r == b.resolve() or b.resolve() in r.parents:
            raise SystemExit(f"Refusing protected path: {path}")


def manifest(exclude: Path) -> dict:
    out = {}
    for p in (ROOT / "models").rglob("*.pt"):
        if exclude in p.parents:
            continue
        st = p.stat()
        out[str(p.relative_to(ROOT))] = (st.st_size, st.st_mtime_ns)
    return out


def build_cmd(arm: str, seed: int, ck: Path, lg: Path, max_steps: int, eval_steps: str) -> list[str]:
    return [
        sys.executable, "-u", "-m", "tetris_rl.training.train_dqn",
        "--preset", "improved", "--reward-mode", "delta", "--seed", str(seed),
        "--train-seed-base", str(TRAIN_SEED_BASE[seed]),
        "--init-from", str(BC_INIT[seed]),
        "--epsilon-start", str(EPSILON), "--epsilon-end", str(EPSILON),
        "--max-steps", str(max_steps), "--eval-at-steps", eval_steps, "--eval-at-start",
        "--eval-games", "50", "--max-pieces", "500", "--device", "cpu", "--no-tb",
        "--demo-path", str(DEMO_TRAIN), "--demo-val-path", str(DEMO_VAL), "--demo-prefill",
        "--imitation-weight", str(ARMS[arm]), "--imitation-batch-size", "64",
        "--grad-diag-every", "50",
        "--checkpoint-dir", str(ck), "--log-dir", str(lg),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="exp_imitreg")
    ap.add_argument("--max-steps", type=int, default=10_000)
    ap.add_argument("--eval-at-steps", default=DEFAULT_EVAL_STEPS)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default="42,43")
    args = ap.parse_args()

    for p in (DEMO_TRAIN, DEMO_VAL, *BC_INIT.values()):
        if not p.is_file():
            raise SystemExit(f"Missing input: {p}")
    arms = [a for a in args.arms.split(",") if a]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    models_root, results_root = ROOT / "models" / args.tag, ROOT / "results" / args.tag
    assert_safe(models_root)
    assert_safe(results_root)
    plan = []
    for seed in seeds:
        for arm in arms:
            ck, lg = models_root / arm / f"seed{seed}", results_root / arm / f"seed{seed}"
            for d in (ck, lg):
                assert_safe(d)
                if d.exists() and any(d.iterdir()):
                    raise SystemExit(f"Output dir exists and is non-empty: {d}")
            plan.append((arm, seed, ck, lg))

    protected = {"frozen": FROZEN, "improved_best": IMPROVED, **{f"bc_seed{s}": p for s, p in BC_INIT.items()}}
    before = {k: sha256(p) for k, p in protected.items()}
    before_manifest = manifest(models_root)
    results_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "tag": args.tag, "max_steps": args.max_steps, "eval_at_steps": args.eval_at_steps,
        "epsilon": EPSILON, "arms": ARMS, "seeds": seeds, "train_seed_base": TRAIN_SEED_BASE,
        "parallel": args.parallel, "protected_before": before,
        "n_preexisting_model_files": len(before_manifest), "runs": [],
    }
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}
    t_all = time.time()
    pending, running = list(plan), []
    failed = False
    while pending or running:
        while pending and len(running) < args.parallel and not failed:
            arm, seed, ck, lg = pending.pop(0)
            lg.mkdir(parents=True, exist_ok=True)
            cmd = build_cmd(arm, seed, ck, lg, args.max_steps, args.eval_at_steps)
            (lg / "command.txt").write_text(" ".join(cmd) + "\n")
            f = (lg / "stdout.log").open("w")
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
            running.append((arm, seed, proc, f, time.time()))
            print(f"launched {arm} seed{seed} pid={proc.pid}", flush=True)
        time.sleep(2)
        for item in list(running):
            arm, seed, proc, f, t0 = item
            rc = proc.poll()
            if rc is None:
                continue
            f.close()
            running.remove(item)
            summary["runs"].append({"arm": arm, "seed": seed, "returncode": rc, "wall_s": time.time() - t0})
            (results_root / "run_summary.json").write_text(json.dumps(summary, indent=2))
            print(f"  done {arm} seed{seed} rc={rc} in {time.time() - t0:.0f}s", flush=True)
            if rc != 0:
                failed = True
        if failed and not running:
            raise SystemExit("A run failed; see stdout.log files")

    summary["total_wall_s"] = time.time() - t_all
    summary["protected_after"] = {k: sha256(p) for k, p in protected.items()}
    summary["protected_unchanged"] = before == summary["protected_after"]
    summary["preexisting_model_files_unchanged"] = before_manifest == manifest(models_root)
    (results_root / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))
    if not (summary["protected_unchanged"] and summary["preexisting_model_files_unchanged"]):
        raise SystemExit("A pre-existing model file changed!")


if __name__ == "__main__":
    main()

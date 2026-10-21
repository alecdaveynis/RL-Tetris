#!/usr/bin/env python3
"""
Heuristic-guided DQN experiment (reward fixed = delta). Sequential runs:

  A_mix : heuristic mixed into action selection, p 0.5 -> 0 over 15k env steps
  B_bc  : heuristic demos pre-fill the replay buffer + 3000-step behaviour-cloning pretrain

Same preset / epsilon schedule / 20k-step budget / validation protocol as
exp_reward_extended (whose `delta` runs are the controls). Refuses protected paths
and non-empty output dirs; verifies protected hashes and a model-file manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "models" / "frozen" / "dqn_best_val1320.pt"
IMPROVED = ROOT / "models" / "improved" / "dqn_best.pt"
FORBIDDEN = (ROOT / "models" / "frozen", ROOT / "models" / "improved")
DEMO_TRAIN = ROOT / "results" / "exp_demos" / "heuristic_demos_train.npz"
DEMO_VAL = ROOT / "results" / "exp_demos" / "heuristic_demos_val.npz"

METHODS = {
    "A_mix": ["--heuristic-mix-start", "0.5", "--heuristic-mix-steps", "15000"],
    "B_bc": ["--demo-path", str(DEMO_TRAIN), "--demo-val-path", str(DEMO_VAL), "--demo-prefill",
             "--bc-pretrain-steps", "3000", "--eval-at-start"],
}
ORDER = [("B_bc", 42), ("A_mix", 42), ("B_bc", 43), ("A_mix", 43)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(exclude: Path) -> dict:
    out = {}
    for p in (ROOT / "models").rglob("*.pt"):
        if exclude in p.parents:
            continue
        st = p.stat()
        out[str(p.relative_to(ROOT))] = (st.st_size, st.st_mtime_ns)
    return out


def assert_safe(path: Path) -> None:
    r = path.resolve()
    for b in FORBIDDEN:
        if r == b.resolve() or b.resolve() in r.parents:
            raise SystemExit(f"Refusing protected path: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="exp_guided")
    ap.add_argument("--max-steps", type=int, default=20_000)
    ap.add_argument("--eval-every-steps", type=int, default=2_500)
    args = ap.parse_args()

    for p in (DEMO_TRAIN, DEMO_VAL):
        if not p.is_file():
            raise SystemExit(f"Missing demos: {p} (run scripts/collect_heuristic_demos.py)")
    models_root, results_root = ROOT / "models" / args.tag, ROOT / "results" / args.tag
    assert_safe(models_root)
    assert_safe(results_root)
    plan = []
    for name, seed in ORDER:
        ck, lg = models_root / name / f"seed{seed}", results_root / name / f"seed{seed}"
        for d in (ck, lg):
            assert_safe(d)
            if d.exists() and any(d.iterdir()):
                raise SystemExit(f"Output dir exists and is non-empty: {d}")
        plan.append((name, seed, ck, lg))

    before = {"frozen": sha256(FROZEN), "improved_best": sha256(IMPROVED)}
    before_manifest = manifest(models_root)
    results_root.mkdir(parents=True, exist_ok=True)
    summary = {"tag": args.tag, "max_steps": args.max_steps, "order": ORDER,
               "protected_before": before, "n_preexisting_model_files": len(before_manifest), "runs": []}
    t_all = time.time()
    for name, seed, ck, lg in plan:
        lg.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", "-m", "tetris_rl.training.train_dqn",
               "--preset", "improved", "--reward-mode", "delta", "--seed", str(seed),
               "--max-steps", str(args.max_steps), "--eval-every-steps", str(args.eval_every_steps),
               "--save-every-steps", str(args.eval_every_steps), "--eval-games", "50",
               "--max-pieces", "500", "--device", "cpu", "--no-tb",
               "--checkpoint-dir", str(ck), "--log-dir", str(lg), *METHODS[name]]
        print("+", " ".join(cmd), flush=True)
        t0 = time.time()
        with (lg / "stdout.log").open("w") as f:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        summary["runs"].append({"method": name, "seed": seed, "returncode": rc, "wall_s": time.time() - t0})
        (results_root / "run_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"  done {name} seed{seed} rc={rc} in {time.time() - t0:.0f}s", flush=True)
        if rc != 0:
            raise SystemExit(f"Run failed: {name} seed{seed}; see {lg / 'stdout.log'}")

    summary["total_wall_s"] = time.time() - t_all
    summary["protected_after"] = {"frozen": sha256(FROZEN), "improved_best": sha256(IMPROVED)}
    summary["protected_unchanged"] = before == summary["protected_after"]
    summary["preexisting_model_files_unchanged"] = before_manifest == manifest(models_root)
    (results_root / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))
    if not (summary["protected_unchanged"] and summary["preexisting_model_files_unchanged"]):
        raise SystemExit("A pre-existing model file changed!")


if __name__ == "__main__":
    main()

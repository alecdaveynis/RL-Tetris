#!/usr/bin/env python3
"""
Extended fixed-step-budget comparison: delta vs heuristic_aligned reward.

Four fresh-init runs executed sequentially (delta42, HA42, delta43, HA43).
Only --reward-mode and --seed vary. Refuses to touch protected checkpoints or
to write into non-empty output directories. Verifies protected-file hashes and
an (size, mtime) manifest of every pre-existing model file before/after.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "models" / "frozen" / "dqn_best_val1320.pt"
IMPROVED = ROOT / "models" / "improved" / "dqn_best.pt"
FORBIDDEN = (ROOT / "models" / "frozen", ROOT / "models" / "improved")

SEEDS = (42, 43)
MODES = ("delta", "heuristic_aligned")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(exclude: Path) -> dict[str, tuple[int, int]]:
    out = {}
    for p in (ROOT / "models").rglob("*.pt"):
        if exclude in p.parents:
            continue
        st = p.stat()
        out[str(p.relative_to(ROOT))] = (st.st_size, st.st_mtime_ns)
    return out


def assert_safe(path: Path) -> None:
    r = path.resolve()
    for banned in FORBIDDEN:
        b = banned.resolve()
        if r == b or b in r.parents:
            raise SystemExit(f"Refusing protected path: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="exp_reward_extended")
    ap.add_argument("--max-steps", type=int, default=20_000)
    ap.add_argument("--eval-every-steps", type=int, default=2_500)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--extra", default="", help="Quoted extra train_dqn args (smoke only)")
    args = ap.parse_args()

    models_root = ROOT / "models" / args.tag
    results_root = ROOT / "results" / args.tag
    assert_safe(models_root)
    assert_safe(results_root)

    plan = []
    for seed in SEEDS:
        for mode in MODES:
            ck, lg = models_root / mode / f"seed{seed}", results_root / mode / f"seed{seed}"
            for d in (ck, lg):
                assert_safe(d)
                if d.exists() and any(d.iterdir()):
                    raise SystemExit(f"Output dir already exists and is non-empty: {d}")
            plan.append((mode, seed, ck, lg))

    before_hash = {"frozen": sha256(FROZEN), "improved_best": sha256(IMPROVED)}
    before_manifest = manifest(models_root)
    results_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "tag": args.tag,
        "max_steps": args.max_steps,
        "eval_every_steps": args.eval_every_steps,
        "device": args.device,
        "order": [(m, s) for m, s, _, _ in plan],
        "protected_before": before_hash,
        "n_preexisting_model_files": len(before_manifest),
        "runs": [],
    }
    t_all = time.time()
    for mode, seed, ck, lg in plan:
        cmd = [
            sys.executable, "-m", "tetris_rl.training.train_dqn",
            "--preset", "improved", "--reward-mode", mode, "--seed", str(seed),
            "--max-steps", str(args.max_steps),
            "--eval-every-steps", str(args.eval_every_steps),
            "--save-every-steps", str(args.eval_every_steps),
            "--eval-games", "50", "--max-pieces", "500",
            "--device", args.device, "--no-tb",
            "--checkpoint-dir", str(ck), "--log-dir", str(lg),
            *shlex.split(args.extra),
        ]
        print("+", " ".join(cmd), flush=True)
        t0 = time.time()
        lg.mkdir(parents=True, exist_ok=True)
        with (lg / "stdout.log").open("w") as logf:
            proc = subprocess.run(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        summary["runs"].append(
            {"mode": mode, "seed": seed, "returncode": proc.returncode, "wall_s": dt,
             "checkpoint_dir": str(ck.relative_to(ROOT)), "log_dir": str(lg.relative_to(ROOT))}
        )
        (results_root / "run_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"  done {mode} seed{seed} rc={proc.returncode} in {dt:.0f}s", flush=True)
        if proc.returncode != 0:
            raise SystemExit(f"Run failed: {mode} seed{seed}; see {lg / 'stdout.log'}")

    summary["total_wall_s"] = time.time() - t_all
    after_hash = {"frozen": sha256(FROZEN), "improved_best": sha256(IMPROVED)}
    after_manifest = manifest(models_root)
    summary["protected_after"] = after_hash
    summary["protected_unchanged"] = before_hash == after_hash
    summary["preexisting_model_files_unchanged"] = before_manifest == after_manifest
    (results_root / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))
    if not (summary["protected_unchanged"] and summary["preexisting_model_files_unchanged"]):
        raise SystemExit("A pre-existing model file changed!")


if __name__ == "__main__":
    main()

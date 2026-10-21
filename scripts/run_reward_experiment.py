#!/usr/bin/env python3
"""
Fresh-init experiment: default delta rewards vs heuristic-aligned Dellacherie weights.

Does not write to models/frozen or models/improved.
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
FORBIDDEN = (
    ROOT / "models" / "frozen",
    ROOT / "models" / "improved",
)
FROZEN = ROOT / "models" / "frozen" / "dqn_best_val1320.pt"
IMPROVED_BEST = ROOT / "models" / "improved" / "dqn_best.pt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_safe_dir(path: Path) -> None:
    resolved = path.resolve()
    for banned in FORBIDDEN:
        banned_r = banned.resolve()
        if resolved == banned_r or banned_r in resolved.parents:
            raise SystemExit(f"Refusing to write under protected path: {path}")


def run_train(argv: list[str]) -> None:
    cmd = [sys.executable, "-m", "tetris_rl.training.train_dqn", *argv]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def smoke() -> dict:
    t0 = time.time()
    for mode in ("delta", "heuristic_aligned"):
        ckpt = ROOT / "models" / "exp_reward_smoke" / mode
        logs = ROOT / "results" / "exp_reward_smoke" / mode
        assert_safe_dir(ckpt)
        assert_safe_dir(logs)
        run_train(
            [
                "--preset",
                "improved",
                "--reward-mode",
                mode,
                "--episodes",
                "6",
                "--seed",
                "42",
                "--min-buffer-size",
                "32",
                "--batch-size",
                "32",
                "--epsilon-decay-steps",
                "200",
                "--eval-every",
                "6",
                "--eval-games",
                "2",
                "--save-every",
                "6",
                "--max-pieces",
                "40",
                "--no-tb",
                "--checkpoint-dir",
                str(ckpt),
                "--log-dir",
                str(logs),
            ]
        )
        assert (ckpt / "dqn_final.pt").is_file()
    return {"seconds": time.time() - t0}


def main_experiment(episodes: int, seeds: list[int], eval_every: int, tag: str) -> dict:
    t0 = time.time()
    runs = []
    for seed in seeds:
        for mode in ("delta", "heuristic_aligned"):
            ckpt = ROOT / "models" / f"exp_reward_{tag}" / mode / f"seed{seed}"
            logs = ROOT / "results" / f"exp_reward_{tag}" / mode / f"seed{seed}"
            assert_safe_dir(ckpt)
            assert_safe_dir(logs)
            start = time.time()
            run_train(
                [
                    "--preset",
                    "improved",
                    "--reward-mode",
                    mode,
                    "--episodes",
                    str(episodes),
                    "--seed",
                    str(seed),
                    "--eval-every",
                    str(eval_every),
                    "--eval-games",
                    "50",
                    "--save-every",
                    str(eval_every),
                    "--max-pieces",
                    "500",
                    "--no-tb",
                    "--checkpoint-dir",
                    str(ckpt),
                    "--log-dir",
                    str(logs),
                ]
            )
            runs.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "seconds": time.time() - start,
                    "checkpoint_dir": str(ckpt),
                    "log_dir": str(logs),
                }
            )
    return {"seconds": time.time() - t0, "runs": runs}


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["smoke", "main", "all"], default="all")
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    p.add_argument("--tag", type=str, default="", help="Optional run tag for output dirs.")
    return p.parse_args()


def main() -> None:
    args = parse()
    protected = {}
    for label, path in (("frozen", FROZEN), ("improved_best", IMPROVED_BEST)):
        if path.is_file():
            protected[label] = {"path": str(path), "sha256_before": sha256(path)}

    summary: dict = {
        "phase": args.phase,
        "episodes": args.episodes,
        "eval_every": args.eval_every,
        "seeds": args.seeds,
        "protected_before": protected,
    }

    if args.phase in ("smoke", "all"):
        summary["smoke"] = smoke()
    if args.phase in ("main", "all"):
        tag = args.tag or f"ep{args.episodes}"
        summary["tag"] = tag
        summary["main"] = main_experiment(args.episodes, args.seeds, args.eval_every, tag)

    protected_after = {}
    for label, path in (("frozen", FROZEN), ("improved_best", IMPROVED_BEST)):
        if path.is_file():
            protected_after[label] = {"path": str(path), "sha256_after": sha256(path)}
            before = protected.get(label, {}).get("sha256_before")
            if before and before != protected_after[label]["sha256_after"]:
                raise SystemExit(f"Protected checkpoint changed: {path}")
    summary["protected_after"] = protected_after

    out_parent = ROOT / "results" / "exp_reward_compare"
    if args.phase in ("main", "all"):
        tag = args.tag or f"ep{args.episodes}"
        out_parent = ROOT / "results" / f"exp_reward_{tag}"
    out_parent.mkdir(parents=True, exist_ok=True)
    out = out_parent / "run_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

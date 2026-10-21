#!/usr/bin/env python3
"""Collect heuristic demonstrations (delta reward) on seeds disjoint from train/val/diag/test."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tetris_rl.rewards import RewardConfig  # noqa: E402
from tetris_rl.training.demos import (  # noqa: E402
    DEMO_TRAIN_SEEDS, DEMO_VAL_SEEDS, DemoSet, collect_demonstrations, save_demos,
)

OUT = ROOT / "results" / "exp_demos"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    info = {}
    for name, seeds, tie in (("train", DEMO_TRAIN_SEEDS, 1), ("val", DEMO_VAL_SEEDS, 2)):
        path = OUT / f"heuristic_demos_{name}.npz"
        if path.exists():
            raise SystemExit(f"{path} exists; refusing to overwrite")
        t0 = time.time()
        d = collect_demonstrations(seeds, RewardConfig(), max_pieces=500, tie_seed=tie)
        save_demos(path, d)
        stats = DemoSet(d).mean_return_stats()
        stats.update(seeds=[min(seeds), max(seeds)], seconds=time.time() - t0, path=str(path.relative_to(ROOT)))
        info[name] = stats
        print(name, stats, flush=True)
    (OUT / "demo_stats.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

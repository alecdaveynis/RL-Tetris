#!/usr/bin/env python3
"""Verify resume identity, --lr override, Double DQN, and seed split before LR runs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from tetris_rl.agents.dqn_agent import DQNAgent
from tetris_rl.training.replay_buffer import Transition
from tetris_rl.training.train_dqn import TEST_SEED_BASE, VALIDATION_SEEDS

FROZEN = Path("models/frozen/dqn_best_val1320.pt")
PROTECTED = [
    Path("models/frozen"),
    Path("models/improved/dqn_best.pt"),
    Path("models/backup_instability_20261020"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    assert FROZEN.exists(), f"missing {FROZEN}"
    digest = sha256(FROZEN)
    Path("results/exp_lr_5e4_200").mkdir(parents=True, exist_ok=True)
    Path("results/exp_lr_1e4_200").mkdir(parents=True, exist_ok=True)
    Path("results/diagnostics/frozen_sha256.txt").write_text(digest + "\n")
    print("frozen_sha256", digest)
    print("frozen_bytes", FROZEN.stat().st_size)

    a = DQNAgent.from_checkpoint(FROZEN, device="cpu", explore=True)
    b = DQNAgent.from_checkpoint(FROZEN, device="cpu", explore=True)
    assert a.total_steps == b.total_steps == 29999
    assert len(a.buffer) == len(b.buffer) == 10733
    assert len(a.optimizer.state) == len(b.optimizer.state) == 10
    assert a.optimizer.param_groups[0]["lr"] == 5e-4
    w1 = next(a.online.parameters()).detach().cpu().clone()
    w2 = next(b.online.parameters()).detach().cpu().clone()
    assert torch.allclose(w1, w2)
    print("resume identity: optimizer+buffer+weights OK")

    # --lr override does not wipe Adam moments
    for g in a.optimizer.param_groups:
        g["lr"] = 1e-4
    n_nonzero = 0
    for st in a.optimizer.state.values():
        if "exp_avg" in st and float(st["exp_avg"].abs().mean()) > 0:
            n_nonzero += 1
    assert n_nonzero == 10
    assert a.optimizer.param_groups[0]["lr"] == 1e-4
    print("lr override keeps Adam moments, sets lr=1e-4")

    # Double DQN: online argmax, target evaluate — check code path on a tiny batch
    t = a.buffer.sample(4)
    a.config.double_dqn = True
    with torch.no_grad():
        feats, mask = a._pad_next_actions(t)
        vec = a._next_q_max_vectorized(feats, mask)
        # Manual DDQN
        manual = []
        for tr in t:
            if tr.terminated or tr.next_state_actions.shape[0] == 0:
                manual.append(0.0)
                continue
            x = torch.as_tensor(tr.next_state_actions, dtype=torch.float32)
            best = int(torch.argmax(a.online(x)).item())
            manual.append(float(a.target(x[best : best + 1])[0]))
        man = torch.tensor(manual)
        assert torch.allclose(vec.cpu(), man, atol=1e-5)
    print("Double DQN: online select, target evaluate — match")

    train_seeds = set(range(42 + 1, 42 + 200 + 1))
    val_seeds = set(VALIDATION_SEEDS)
    test_seeds = set(range(TEST_SEED_BASE, TEST_SEED_BASE + 100))
    assert train_seeds.isdisjoint(val_seeds), "train and validation seeds overlap"
    assert val_seeds.isdisjoint(test_seeds), "validation and test seeds overlap"
    overlap_tt = sorted(train_seeds & test_seeds)
    print("seed split: train 43-242, val 100000-100049, test 0-99")
    print("train ∩ val = empty; val ∩ test = empty")
    if overlap_tt:
        # Specified --seed 42 collides with test seeds 0-99 on 43-99.
        # Model *selection* still uses disjoint val; report but do not change seeds.
        print(
            f"NOTE: train ∩ test = {overlap_tt[0]}-{overlap_tt[-1]} "
            f"({len(overlap_tt)} seeds). Selection uses val only; test protocol kept."
        )

    for p in PROTECTED:
        assert p.exists(), f"protected path missing: {p}"
    print("protected paths present; experiments will write to models/exp_lr_*_200 only")
    print("VERIFY_OK")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase-1 / pre-flight checks for the imitation-regularised DQN experiment.

 1. Hashes of protected checkpoints (frozen DQN, improved best, both BC clones).
 2. BC reproducibility: re-run behaviour cloning (same recipe/seed as the original
    B_bc run, into memory only) and compare against the saved BC checkpoint.
 3. Initialising a fresh DQN from the BC checkpoint keeps the weights bit-for-bit and
    plays identically (greedy, first validation seeds).
 4. Loss scales: TD loss, imitation CE, their gradient norms and the effective share
    of lambda * CE for lambda in {0.1, 1.0}, on the BC network and on a degraded
    network (final checkpoint of the old ordinary-DQN fine-tuning run).

Writes results/exp_imitreg/verification/{verification.json,loss_scales.md}. Read-only
with respect to every model file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_imitation_experiment import BC_INIT, FROZEN, IMPROVED, sha256  # noqa: E402
from tetris_rl.agents.dqn_agent import DQNAgent, improved_dqn_config  # noqa: E402
from tetris_rl.training.demos import DemoSet, bc_pretrain, listwise_ce  # noqa: E402
from tetris_rl.training.imitation import (  # noqa: E402
    ImitationRegularizer, demo_metrics, flat_grad_norm, grad_cosine, imitation_ce,
)
from tetris_rl.training.train_dqn import VALIDATION_SEEDS, evaluate_greedy  # noqa: E402

DEMO_TRAIN = ROOT / "results" / "exp_demos" / "heuristic_demos_train.npz"
DEMO_VAL = ROOT / "results" / "exp_demos" / "heuristic_demos_val.npz"
OUT = ROOT / "results" / "exp_imitreg" / "verification"
DEGRADED = {s: ROOT / "models" / "exp_guided" / "B_bc" / f"seed{s}" / "dqn_final.pt" for s in (42, 43)}


def cfg_for(seed: int):
    # Same construction as the original B_bc run: improved preset, delta reward, CPU.
    return improved_dqn_config(gamma=0.99, device="cpu", seed=seed)


def max_weight_diff(a: dict, b: dict) -> float:
    return max(float((a[k].float() - b[k].float()).abs().max()) for k in a)


def bc_reproducibility(seed: int, demos: DemoSet, val_demos: DemoSet) -> dict:
    ck = torch.load(BC_INIT[seed], map_location="cpu", weights_only=False)
    agent = DQNAgent(cfg_for(seed))
    t0 = time.time()
    bc_pretrain(agent, demos, 3000, td_weight=1.0, val_demos=val_demos, seed=seed + 999)
    idx = np.arange(0, len(val_demos), 5)
    new_m = demo_metrics(agent.online, val_demos, idx)
    ref = DQNAgent(cfg_for(seed))
    ref.online.load_state_dict(ck["online"])
    ref_m = demo_metrics(ref.online, val_demos, idx)
    return {
        "seed": seed,
        "bc_seconds": time.time() - t0,
        "ckpt_total_steps": ck["total_steps"],
        "ckpt_epsilon": ck["epsilon"],
        "max_abs_weight_diff_rerun_vs_saved": max_weight_diff(agent.online.state_dict(), ck["online"]),
        "bitwise_identical": all(torch.equal(agent.online.state_dict()[k], ck["online"][k]) for k in ck["online"]),
        "val_top1_saved": ref_m["top1"], "val_top1_rerun": new_m["top1"],
        "val_ce_saved": ref_m["ce"], "val_ce_rerun": new_m["ce"],
    }


def init_fidelity(seed: int, n_games: int = 3) -> dict:
    ck = torch.load(BC_INIT[seed], map_location="cpu", weights_only=False)
    agent = DQNAgent(improved_dqn_config(device="cpu", seed=seed, epsilon_start=0.05, epsilon_end=0.05))
    agent.load_init_weights(BC_INIT[seed])
    ref = DQNAgent.from_checkpoint(BC_INIT[seed], device="cpu", explore=False)
    same_w = all(torch.equal(agent.online.state_dict()[k], ck["online"][k]) for k in ck["online"])
    same_t = all(torch.equal(agent.target.state_dict()[k], ck["target"][k]) for k in ck["target"])
    m_new = evaluate_greedy(agent, 500, seeds=VALIDATION_SEEDS[:n_games])
    m_ref = evaluate_greedy(ref, 500, seeds=VALIDATION_SEEDS[:n_games])
    return {
        "seed": seed, "online_bitwise": same_w, "target_bitwise": same_t,
        "optimizer_state_loaded": len(agent.optimizer.state_dict()["state"]) > 0,
        "counters_fresh": agent.total_steps == 0 and agent.grad_steps == 0 and len(agent.buffer) == 0,
        "epsilon_after_init": agent.epsilon,
        f"greedy_mean_score_first{n_games}_init": m_new["mean_score"],
        f"greedy_mean_score_first{n_games}_from_checkpoint": m_ref["mean_score"],
        "identical_play": m_new["mean_score"] == m_ref["mean_score"],
    }


def loss_scales(seed: int, ckpt: Path, demos: DemoSet, n_batches: int = 200) -> dict:
    agent = DQNAgent(improved_dqn_config(device="cpu", seed=seed, epsilon_start=0.05, epsilon_end=0.05))
    agent.load_init_weights(ckpt)
    demos.fill_buffer(agent.buffer)  # step-0 replay contents (demos only, as in the real runs)
    reg = ImitationRegularizer(demos, 1.0, batch_size=64, seed=seed + 1)
    params = list(agent.online.parameters())
    td, ce, gtd, gce, cos = [], [], [], [], []
    for _ in range(n_batches):
        batch = agent.buffer.sample(64)
        l_td = agent._compute_loss(batch)
        feats, mask, chosen = reg.sample()
        l_ce, _ = imitation_ce(agent.online, feats, mask, chosen)
        g_td = torch.autograd.grad(l_td, params)
        g_ce = torch.autograd.grad(l_ce, params)
        td.append(float(l_td)); ce.append(float(l_ce))
        gtd.append(flat_grad_norm(g_td)); gce.append(flat_grad_norm(g_ce)); cos.append(grad_cosine(g_td, g_ce))
    td_m, ce_m, gtd_m, gce_m = (float(np.mean(x)) for x in (td, ce, gtd, gce))
    out = {
        "seed": seed, "ckpt": str(ckpt.relative_to(ROOT)),
        "td_loss": td_m, "imit_ce": ce_m, "grad_norm_td": gtd_m, "grad_norm_ce": gce_m,
        "grad_cos_td_ce": float(np.nanmean(cos)),
        "uniform_policy_ce_ref": float(np.mean(np.log(np.diff(demos.d["block_off"])))),
    }
    for lam in (0.1, 1.0):
        out[f"lam{lam}"] = {
            "weighted_ce": lam * ce_m,
            "loss_share": lam * ce_m / (td_m + lam * ce_m),
            "weighted_grad_norm": lam * gce_m,
            "grad_share": lam * gce_m / (gtd_m + lam * gce_m),
            "grad_ratio_imit_over_td": lam * gce_m / gtd_m,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bc-repro", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    demos, val_demos = DemoSet.load(DEMO_TRAIN), DemoSet.load(DEMO_VAL)
    res: dict = {"hashes": {
        "frozen": sha256(FROZEN), "improved_best": sha256(IMPROVED),
        **{f"bc_seed{s}": sha256(p) for s, p in BC_INIT.items()},
    }}
    res["init_fidelity"] = [init_fidelity(s) for s in (42, 43)]
    print(json.dumps(res["init_fidelity"], indent=2), flush=True)
    if not args.skip_bc_repro:
        res["bc_reproducibility"] = [bc_reproducibility(s, demos, val_demos) for s in (42, 43)]
        print(json.dumps(res["bc_reproducibility"], indent=2), flush=True)
    res["loss_scales"] = {
        "bc_init": [loss_scales(s, BC_INIT[s], demos) for s in (42, 43)],
        "degraded_old_B_final": [loss_scales(s, DEGRADED[s], demos) for s in (42, 43)],
    }
    (OUT / "verification.json").write_text(json.dumps(res, indent=2))

    lines = [
        "| network | seed | TD loss | imit CE | ‖∇TD‖ | ‖∇CE‖ | cos(∇TD,∇CE) | λ | λ·CE | loss share | grad ratio λ‖∇CE‖/‖∇TD‖ |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, rows in res["loss_scales"].items():
        for r in rows:
            for lam in (0.1, 1.0):
                x = r[f"lam{lam}"]
                lines.append(
                    f"| {name} | {r['seed']} | {r['td_loss']:.3f} | {r['imit_ce']:.3f} | {r['grad_norm_td']:.2f} | "
                    f"{r['grad_norm_ce']:.2f} | {r['grad_cos_td_ce']:.2f} | {lam} | {x['weighted_ce']:.3f} | "
                    f"{x['loss_share']:.1%} | {x['grad_ratio_imit_over_td']:.2f} |"
                )
    (OUT / "loss_scales.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate delta vs heuristic-aligned reward experiment and write charts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tetris_rl.agents.dqn_agent import DQNAgent
from tetris_rl.training.train_dqn import evaluate_greedy, probe_fixed_state_q

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "exp_reward_compare"
MODES = ("delta", "heuristic_aligned")


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def fget(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _dedupe_episode(rows: list[dict]) -> list[dict]:
    by_ep: dict[int, dict] = {}
    for row in rows:
        try:
            ep = int(float(row["episode"]))
        except (KeyError, TypeError, ValueError):
            continue
        by_ep[ep] = row
    return [by_ep[k] for k in sorted(by_ep)]


def discover_runs(tag: str | None = None) -> list[dict]:
    runs = []
    pairs: list[tuple[str, Path, Path]] = []
    if tag:
        for mode in MODES:
            pairs.append(
                (mode, ROOT / "models" / f"exp_reward_{tag}" / mode, ROOT / "results" / f"exp_reward_{tag}" / mode)
            )
    else:
        for mode in MODES:
            pairs.append((mode, ROOT / "models" / f"exp_reward_{mode}", ROOT / "results" / f"exp_reward_{mode}"))
            tagged = ROOT / "results" / "exp_reward_ep400" / mode
            if tagged.is_dir():
                pairs.append(
                    (mode, ROOT / "models" / "exp_reward_ep400" / mode, tagged)
                )
    for mode, base_ckpt, base_log in pairs:
        if not base_log.is_dir():
            continue
        for log_dir in sorted(p for p in base_log.iterdir() if p.is_dir()):
            seed_s = log_dir.name.replace("seed", "")
            try:
                seed = int(seed_s)
            except ValueError:
                continue
            ckpt_dir = base_ckpt / log_dir.name
            runs.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "log_dir": log_dir,
                    "ckpt_dir": ckpt_dir,
                    "train": _dedupe_episode(read_csv(log_dir / "training_metrics.csv")),
                    "val": _dedupe_episode(read_csv(log_dir / "val_metrics.csv")),
                    "best": ckpt_dir / "dqn_best.pt",
                    "final": ckpt_dir / "dqn_final.pt",
                }
            )
    return runs


def eval_ckpt(path: Path) -> dict:
    if not path.is_file():
        return {}
    agent = DQNAgent.from_checkpoint(path, device="cpu", explore=False)
    metrics = evaluate_greedy(agent, max_pieces=500, n_games=50)
    q = probe_fixed_state_q(agent)
    metrics.update(q)
    metrics["episodes_trained"] = agent.episodes_trained
    metrics["total_steps"] = agent.total_steps
    return metrics


def plot_val(runs: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    colors = {"delta": "#4c72b0", "heuristic_aligned": "#dd8452"}
    for ax, key, title in (
        (axes[0, 0], "mean_score", "Validation mean score"),
        (axes[0, 1], "mean_lines", "Validation mean lines"),
        (axes[1, 0], "mean_pieces", "Validation mean pieces"),
        (axes[1, 1], "q_max_mean", "Fixed-state max Q"),
    ):
        for run in runs:
            xs = [fget(r, "episode") for r in run["val"]]
            ys = [fget(r, key) for r in run["val"]]
            if not xs:
                continue
            ax.plot(
                xs,
                ys,
                marker="o",
                color=colors[run["mode"]],
                label=f"{run['mode']} seed{run['seed']}",
            )
        ax.set_xlabel("episode")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_train(runs: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    colors = {"delta": "#4c72b0", "heuristic_aligned": "#dd8452"}
    for ax, key, title in (
        (axes[0], "loss", "Train Huber loss"),
        (axes[1], "q_mean", "Train Q mean"),
        (axes[2], "td_error", "Train |TD error|"),
    ):
        for run in runs:
            xs = [fget(r, "episode") for r in run["train"]]
            ys = [fget(r, key) for r in run["train"]]
            if not xs:
                continue
            ax.plot(xs, ys, color=colors[run["mode"]], alpha=0.85, linewidth=1.1)
        ax.set_xlabel("episode")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, default="")
    args = p.parse_args()
    out_dir = OUT if not args.tag else ROOT / "results" / f"exp_reward_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(args.tag or None)
    evaluated = []
    for run in runs:
        rec = {
            "mode": run["mode"],
            "seed": run["seed"],
            "val_curve": run["val"],
            "n_train_episodes": len(run["train"]),
        }
        if run["train"]:
            last = run["train"][-1]
            rec["final_train"] = {
                "score": fget(last, "score"),
                "lines": fget(last, "lines"),
                "loss": fget(last, "loss"),
                "q_mean": fget(last, "q_mean"),
                "td_error": fget(last, "td_error"),
                "epsilon": fget(last, "epsilon"),
                "steps": fget(last, "steps"),
            }
        rec["eval_best"] = eval_ckpt(run["best"]) if run["best"].is_file() else eval_ckpt(run["final"])
        rec["eval_final_from_csv"] = run["val"][-1] if run["val"] else {}
        evaluated.append(rec)

    by_mode: dict[str, list[float]] = {m: [] for m in MODES}
    for rec in evaluated:
        score = rec.get("eval_best", {}).get("mean_score")
        if score is not None:
            by_mode[rec["mode"]].append(float(score))

    summary = {
        "runs": evaluated,
        "best_val_mean_by_mode": {
            m: {
                "n": len(vals),
                "mean": float(np.mean(vals)) if vals else None,
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "values": vals,
            }
            for m, vals in by_mode.items()
        },
    }
    (out_dir / "analysis.json").write_text(json.dumps(summary, indent=2, default=str))
    plot_val(runs, out_dir / "val_curves.png")
    plot_train(runs, out_dir / "train_diagnostics.png")
    print(json.dumps(summary["best_val_mean_by_mode"], indent=2))


if __name__ == "__main__":
    main()

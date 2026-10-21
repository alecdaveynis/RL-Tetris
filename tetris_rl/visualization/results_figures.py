"""
README figures built from the project's own experiment outputs in ``results/``.

    python -m tetris_rl.visualization.results_figures --out-dir docs/images

Writes ``results_overview.png``, ``imitation_tradeoff.png`` and ``training_pipeline.png``
(and ``.svg`` of each). Numbers are read from:

  results/ab_comparison/comparison_summary.json   random / heuristic / from-scratch DQNs
  results/exp_imitreg/analysis/analysis.json      clone + imitation-regularised arms
  results/exp_imitreg/<arm>/seed<S>/eval_metrics.csv   step-0 (clone) top-out

``results/`` is gitignored; the generated images in ``docs/images`` are what the README embeds.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from tetris_rl.visualization.analysis import REPO_ROOT  # noqa: E402

INK, MUTED, FAINT, PAPER = "#1F2937", "#6B7280", "#E5E7EB", "#FFFFFF"
CORNELL = "#B31B1B"
C_RAND, C_SCRATCH, C_HEUR = "#9CA3AF", "#E4B02B", "#2F6FDB"
C_CLONE, C_L0, C_L01, C_L1 = "#7C8DA6", "#E45756", "#2A9D8F", "#9D6FB0"

plt.rcParams.update(
    {
        "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "path",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#9CA3AF",
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "text.color": INK,
    }
)

HEURISTIC_VAL = 21_422.0  # heuristic on the validation seeds (500-piece cap); see reports/


def _load(res: Path):
    ab = {r["agent"]: r for r in json.loads((res / "ab_comparison/comparison_summary.json").read_text())}
    sbf = json.loads((res / "exp_imitreg/analysis/analysis.json").read_text())["start_best_final"]
    arms: dict[str, dict[int, dict]] = {}
    for key, v in sbf.items():
        arm, seed = key.split("/")
        arms.setdefault(arm, {})[int(seed.replace("seed", ""))] = v
    clone_topout = {}
    for seed in (42, 43):
        with open(res / f"exp_imitreg/B_lam0/seed{seed}/eval_metrics.csv") as f:
            row = next(csv.DictReader(f))
        clone_topout[seed] = float(row["top_out_rate"])
    return ab, arms, clone_topout


def _save(fig, out: Path, name: str, dpi: int = 200):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=dpi, facecolor=PAPER, bbox_inches="tight", pad_inches=0.25)
    fig.savefig(out / f"{name}.svg", facecolor=PAPER, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def _hbars(ax, labels, values, colors, dots=None, xmax=28_000, ref=None):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.62, zorder=2)
    for k, (yi, v) in enumerate(zip(y, values)):
        right = max([v] + list((dots[k] if dots and dots[k] else [])))
        ax.text(right + xmax * 0.018, yi, f"{v:,.0f}", va="center", fontsize=11, fontweight="bold")
    if dots:
        for yi, d in zip(y, dots):
            if d:
                ax.scatter(d, [yi] * len(d), s=26, color="white", edgecolor=INK, lw=1.1, zorder=4)
    if ref is not None:
        ax.axvline(ref, color=C_HEUR, ls="--", lw=1.2, zorder=1)
        ax.text(ref, len(labels) - 0.32, "heuristic", color=C_HEUR, fontsize=9, ha="center")
    ax.set_yticks(y, labels, fontsize=11)
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k" if v else "0")
    ax.grid(axis="x", color=FAINT, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("mean game score (higher is better)", fontsize=10, color=MUTED)


def results_overview(res: Path, out: Path):
    ab, arms, _ = _load(res)
    fig, (a, b) = plt.subplots(1, 2, figsize=(14.5, 4.6), gridspec_kw={"width_ratios": [1, 1.05]})

    labels = ["Random", "DQN (original)", "DQN (improved)", "Heuristic"]
    vals = [ab[k]["mean_score"] for k in ("random", "dqn_original", "dqn_improved", "heuristic")]
    _hbars(a, labels, vals, [C_RAND, C_SCRATCH, C_SCRATCH, C_HEUR])
    a.set_title("(a) DQN trained from scratch", loc="left", fontsize=13, fontweight="bold")
    a.text(0, -1.75, f"{ab['random']['n_games']} seeded games (seeds 0-99), greedy, 500-piece cap",
           fontsize=9, color=MUTED)

    def stats(arm, key):
        d = [arms[arm][s][key] for s in (42, 43)]
        return float(np.mean(d)), d

    clone, clone_d = stats("B_lam0", "start")
    l0, l0_d = stats("B_lam0", "final")
    l01, l01_d = stats("C_lam0.1", "final")
    l1, l1_d = stats("D_lam1", "final")
    labels = ["Heuristic", "Behaviour clone", "DQN fine-tune, λ=0", "DQN + imitation, λ=0.1",
              "DQN + imitation, λ=1"]
    vals = [HEURISTIC_VAL, clone, l0, l01, l1]
    dots = [None, clone_d, l0_d, l01_d, l1_d]
    _hbars(b, labels, vals, [C_HEUR, C_CLONE, C_L0, C_L01, C_L1], dots=dots, ref=HEURISTIC_VAL)
    b.set_title("(b) Clone the heuristic, then improve with RL", loc="left", fontsize=13, fontweight="bold")
    b.text(0, -1.75, "50 validation games, 10k RL steps; bar = mean of 2 training seeds, dots = each seed",
           fontsize=9, color=MUTED)
    fig.suptitle("Mean game score by agent", x=0.01, ha="left", fontsize=16, fontweight="bold", y=1.03)
    fig.tight_layout()
    _save(fig, out, "results_overview")


def imitation_tradeoff(res: Path, out: Path):
    _, arms, clone_topout = _load(res)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.0))
    names = ["Clone\n(start)", "λ=0", "λ=0.1", "λ=1"]
    cols = [C_CLONE, C_L0, C_L01, C_L1]
    arm_keys = [None, "B_lam0", "C_lam0.1", "D_lam1"]

    def panel(ax, start_key, final_key, start_override, fmt, title, ylab):
        means, dots = [], []
        for ak in arm_keys:
            if ak is None:
                d = start_override if start_override else [arms["B_lam0"][s][start_key] for s in (42, 43)]
            else:
                d = [arms[ak][s][final_key] for s in (42, 43)]
            dots.append(d)
            means.append(float(np.mean(d)))
        x = np.arange(4)
        ax.bar(x, means, color=cols, width=0.62, zorder=2)
        for xi, d, m in zip(x, dots, means):
            ax.scatter([xi - 0.07, xi + 0.07], d, s=26, color="white", edgecolor=INK, lw=1.1, zorder=4)
            ax.text(xi, max(d + [m]) + max(means) * 0.05, fmt(m), ha="center", fontsize=11, fontweight="bold")
        ax.set_xticks(x, names, fontsize=11)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
        ax.set_ylabel(ylab, fontsize=10, color=MUTED)
        ax.set_ylim(0, max(max(d) for d in dots) * 1.25)
        ax.grid(axis="y", color=FAINT, zorder=0)

    panel(a, "start_hs_top1", "final_hs_top1", None, lambda v: f"{v:.2f}",
          "Agreement with the heuristic", "top-1 agreement")
    panel(b, None, "final_top_out", [clone_topout[42], clone_topout[43]], lambda v: f"{v:.0%}",
          "Games that top out before 500 pieces", "top-out rate")
    b.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.suptitle("What the imitation loss protects (after 10k RL steps)", x=0.01, ha="left",
                 fontsize=15, fontweight="bold", y=1.04)
    fig.tight_layout()
    _save(fig, out, "imitation_tradeoff")


def training_pipeline(out: Path, res: Optional[Path] = None):
    fig = plt.figure(figsize=(14.5, 4.4), facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 145)
    ax.set_ylim(0, 44)
    ax.axis("off")

    def box(x, y, w, h, title, sub, color, fc="#F9FAFB"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc=fc, ec=color, lw=2.0))
        ax.text(x + w / 2, y + h - 3.4, title, ha="center", va="center", fontsize=12.5, fontweight="bold")
        for i, line in enumerate(sub):
            ax.text(x + w / 2, y + h - 7.6 - i * 3.3, line, ha="center", va="center", fontsize=10, color=MUTED)

    def arrow(p0, p1, color=INK, rad=0.0, lw=1.8):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14, lw=lw, color=color,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))

    ax.text(2, 41, "Training recipe behind the best agent", fontsize=16, fontweight="bold", ha="left")
    y0, h = 14, 20
    box(2, y0, 26, h, "Heuristic", ["hand-tuned weights:", "lines, height,", "holes, bumpiness"], C_HEUR)
    box(36, y0, 26, h, "Expert demos", ["25,000 decisions", "50 games", "(seeds 300000+)"], C_HEUR)
    box(70, y0, 26, h, "Behaviour clone", ["3,000 supervised steps", "cross-entropy over", "legal placements"], C_CLONE)
    box(104, y0, 39, h, "Double DQN fine-tune", ["10,000 RL steps, ε = 0.05", "L = L_TD + λ · L_imitation", "λ = 0.1 works best"], C_L01,
        fc="#F0FAF8")
    for x0, x1 in ((28, 36), (62, 70), (96, 104)):
        arrow((x0 + 0.3, y0 + h / 2), (x1 - 0.3, y0 + h / 2))
    # demos also feed the persistent imitation loss
    arrow((49, y0), (123.5, y0 - 0.2), color=C_HEUR, rad=0.22, lw=1.5)
    ax.text(86, 1.2, "same demos stay in the loss for all of RL (persistent imitation term)",
            ha="center", fontsize=10, color=C_HEUR, style="italic")

    ab, arms, _ = _load(res) if res else (None, None, None)
    if arms:
        c = np.mean([arms["B_lam0"][s]["start"] for s in (42, 43)])
        r = np.mean([arms["C_lam0.1"][s]["final"] for s in (42, 43)])
        ax.text(83, y0 + h + 3.6, f"clone ≈ {c:,.0f}", ha="center", fontsize=11, fontweight="bold", color=C_CLONE)
        ax.text(123.5, y0 + h + 3.6, f"≈ {r:,.0f}  (+{(r / c - 1):.0%})", ha="center", fontsize=11,
                fontweight="bold", color=C_L01)
    _save(fig, out, "training_pipeline")


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "docs/images"))
    args = ap.parse_args(argv)
    res, out = Path(args.results_dir), Path(args.out_dir)
    results_overview(res, out)
    imitation_tradeoff(res, out)
    training_pipeline(out, res)
    print("wrote results_overview, imitation_tradeoff, training_pipeline (.png/.svg) to", out)


if __name__ == "__main__":
    main()

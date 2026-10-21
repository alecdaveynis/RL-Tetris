#!/usr/bin/env python3
"""Analyse the imitation-regularised DQN experiment (results/exp_imitreg)."""

from __future__ import annotations

import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TAG = "exp_imitreg"
RES = ROOT / "results" / TAG
MOD = ROOT / "models" / TAG
OUT = RES / "analysis"
ARMS = ("B_lam0", "C_lam0.1", "D_lam1")
LAM = {"B_lam0": 0.0, "C_lam0.1": 0.1, "D_lam1": 1.0}
NAME = {"B_lam0": "B: standard DQN (λ=0)", "C_lam0.1": "C: imitation λ=0.1", "D_lam1": "D: imitation λ=1.0"}
COL = {"B_lam0": "tab:red", "C_lam0.1": "tab:blue", "D_lam1": "tab:green"}
SEEDS = (42, 43)
HEURISTIC_REF = 21422.0  # same 50 validation seeds (results/reports/guided_dqn_experiment.md)


def read(path: Path):
    return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(path.open())]


def col(rows, k):
    return np.array([r.get(k, np.nan) for r in rows], float)


# ---- per-game evaluation (parallel) ------------------------------------------------


def _eval_chunk(args):
    ckpt, seeds = args
    import torch

    torch.set_num_threads(1)
    from tetris_rl.agents.dqn_agent import DQNAgent
    from tetris_rl.rewards import RewardConfig
    from tetris_rl.training import diagnostics as dg

    agent = DQNAgent.from_checkpoint(ckpt, device="cpu", explore=False)
    return [dg.evaluate_greedy_extended(agent, 500, RewardConfig(), seeds=[s])["mean_score"] for s in seeds]


def per_game_scores(ckpts: dict[str, Path], workers: int = 9) -> dict[str, np.ndarray]:
    from tetris_rl.training.diagnostics import VALIDATION_SEEDS

    cache = OUT / "per_game_scores.json"
    have = json.loads(cache.read_text()) if cache.exists() else {}
    todo = {k: p for k, p in ckpts.items() if k not in have}
    jobs, keys = [], []
    for k, p in todo.items():
        for chunk in np.array_split(np.array(VALIDATION_SEEDS), 5):
            jobs.append((str(p), [int(x) for x in chunk]))
            keys.append(k)
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            outs = list(ex.map(_eval_chunk, jobs))
        for k in todo:
            have[k] = [x for kk, o in zip(keys, outs) if kk == k for x in o]
        cache.write_text(json.dumps(have))
    return {k: np.array(have[k]) for k in ckpts}


def boot_ci(diff, n=5000):
    rng = np.random.default_rng(0)
    means = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---- plotting ----------------------------------------------------------------------


def line_fig(ev, panels, fname, suptitle, hlines=None, extra=None):
    """panels: list of (key, ylabel). One row per panel, one column per seed."""
    fig, axes = plt.subplots(len(panels), 2, figsize=(11, 2.9 * len(panels)), squeeze=False)
    for i, (key, ylabel) in enumerate(panels):
        for j, s in enumerate(SEEDS):
            ax = axes[i][j]
            for a in ARMS:
                rows = ev[(a, s)]
                x, y = col(rows, "env_steps"), col(rows, key)
                ok = ~np.isnan(y)
                ax.plot(x[ok], y[ok], "o-", color=COL[a], label=NAME[a], ms=4)
            if hlines and key in hlines:
                for yv, lab, st in hlines[key](s):
                    ax.axhline(yv, color="k", ls=st, lw=0.9, label=lab)
            ax.set_title(f"{ylabel} - seed {s}", fontsize=9)
            ax.set_xlabel("RL env steps")
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=120)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ev = {(a, s): read(RES / a / f"seed{s}" / "eval_metrics.csv") for a in ARMS for s in SEEDS}
    for rows in ev.values():  # derived efficiency metrics (separate scoring style from survival)
        for r in rows:
            r["score_per_piece"] = r["mean_score"] / max(r["mean_pieces"], 1e-9)
            r["score_per_line"] = r["mean_score"] / max(r["mean_lines"], 1e-9)
    info = {(a, s): json.loads((RES / a / f"seed{s}" / "run_summary.json").read_text()) for a in ARMS for s in SEEDS}
    summ = json.loads((RES / "run_summary.json").read_text())
    out: dict = {"protected_unchanged": summ.get("protected_unchanged"),
                 "preexisting_model_files_unchanged": summ.get("preexisting_model_files_unchanged"),
                 "total_wall_s": summ.get("total_wall_s")}

    # ---- A: clone only = step 0; identical across arms? ---------------------------
    a_rows = {}
    for s in SEEDS:
        zero = [ev[(a, s)][0] for a in ARMS]
        assert all(r["env_steps"] == 0 for r in zero)
        scores = {a: ev[(a, s)][0]["mean_score"] for a in ARMS}
        out.setdefault("step0_identical_across_arms", {})[str(s)] = bool(len(set(scores.values())) == 1)
        out.setdefault("step0_scores", {})[str(s)] = scores
        a_rows[s] = zero[0]
    start = {s: a_rows[s]["mean_score"] for s in SEEDS}

    # ---- figures ---------------------------------------------------------------------
    hl_score = lambda s: [(start[s], "A: clone only (step 0)", "--"), (0.85 * start[s], "85% of clone", ":"),  # noqa: E731
                          (HEURISTIC_REF, "heuristic", "-.")]
    line_fig(ev, [("mean_score", "validation mean score"), ("median_score", "validation median score")],
             "fig1_val_score.png", "Validation score vs RL steps (50 fixed games, greedy, no heuristic)",
             hlines={"mean_score": hl_score})
    line_fig(ev, [("w_td_loss", "TD loss (train window mean)"), ("w_total_loss", "combined loss (TD + λ·CE)")],
             "fig2_td_loss.png", "TD / combined loss vs RL steps")
    line_fig(ev, [("w_imit_loss", "imitation CE on demo batches (train window)"),
                  ("demo_val_ce", "imitation CE on held-out demo states (fixed 2000)")],
             "fig3_imitation_loss.png", "Imitation loss vs RL steps (all arms log CE; λ=0 does not optimise it)",
             hlines={"demo_val_ce": lambda s: [(ev[("B_lam0", s)][0]["demo_val_ce"], "clone (step 0)", "--")]})
    line_fig(ev, [("hs_top1_agreement", "top-1 agreement with heuristic, fixed heuristic states"),
                  ("onpolicy_heuristic_agreement", "agreement on states the policy visits"),
                  ("demo_val_top1", "top-1 on held-out demo states")],
             "fig4_heuristic_agreement.png", "Heuristic action agreement vs RL steps")
    line_fig(ev, [("holes_created_per_move", "holes created per move"),
                  ("agg_height_delta_per_move", "aggregate-height change per move"),
                  ("mean_peak_height", "mean peak stack height")],
             "fig5_holes_height.png", "Hole creation and stack height vs RL steps")
    line_fig(ev, [("mean_lines", "mean lines cleared"), ("mean_pieces", "mean pieces survived"),
                  ("top_out_rate", "top-out rate"), ("score_per_piece", "score per piece placed"),
                  ("score_per_line", "score per cleared line (multi-line clear quality)")],
             "fig6_lines_pieces_topout.png", "Gameplay diagnostics vs RL steps")
    line_fig(ev, [("hs_q_max_mean", "mean max-Q (fixed heuristic states)"), ("hs_q_action_range", "Q range across legal actions"),
                  ("hs_spearman", "Spearman(Q, heuristic score)"), ("hs_heuristic_rank", "Q-rank of heuristic's best action (1=top)"),
                  ("demo_val_margin", "expert-Q minus best-alternative-Q (held-out demos)")],
             "fig7_q_diagnostics.png", "Q magnitude and ranking diagnostics")
    line_fig(ev, [("grad_norm_mean", "total grad norm (pre-clip, mean)"), ("grad_clip_frac", "fraction of steps clipped"),
                  ("w_gn_td", "‖∇ TD‖ (sampled every 50 steps)"), ("w_gn_imit_weighted", "‖∇ λ·CE‖ (sampled every 50 steps)"),
                  ("w_grad_cos_td_imit", "cos(∇TD, ∇CE)")],
             "fig8_gradients.png", "Gradient diagnostics")

    # ---- start / best / final ---------------------------------------------------------
    rows_tbl, ckpts = [], {}
    for s in SEEDS:
        ckpts[f"clone_s{s}"] = ROOT / "models" / "exp_guided" / "B_bc" / f"seed{s}" / "dqn_best.pt"
    summary_rows = {}
    for a in ARMS:
        for s in SEEDS:
            r = ev[(a, s)]
            steps, sc = col(r, "env_steps"), col(r, "mean_score")
            rl = steps > 0
            i_best_rl = int(np.argmax(np.where(rl, sc, -np.inf)))
            i_best_all = int(np.argmax(sc))
            fin = r[-1]
            d = dict(
                arm=a, seed=s, start=sc[0], final=sc[-1], final_step=int(steps[-1]),
                best_rl=sc[i_best_rl], best_rl_step=int(steps[i_best_rl]),
                best_overall=sc[i_best_all], best_overall_step=int(steps[i_best_all]),
                min_rl=float(sc[rl].min()), frac_final=sc[-1] / sc[0], frac_best_rl=sc[i_best_rl] / sc[0],
                frac_min=float(sc[rl].min() / sc[0]),
                passes_85=bool(sc[-1] >= 0.85 * sc[0]),
                final_top_out=fin["top_out_rate"], final_holes=fin["holes_created_per_move"],
                start_holes=r[0]["holes_created_per_move"], holes_ratio=fin["holes_created_per_move"] / max(r[0]["holes_created_per_move"], 1e-9),
                start_hs_top1=r[0]["hs_top1_agreement"], final_hs_top1=fin["hs_top1_agreement"],
                start_demo_val_top1=r[0]["demo_val_top1"], final_demo_val_top1=fin["demo_val_top1"],
                start_demo_val_ce=r[0]["demo_val_ce"], final_demo_val_ce=fin["demo_val_ce"],
                final_agg_h=fin["agg_height_delta_per_move"], start_agg_h=r[0]["agg_height_delta_per_move"],
                final_lines=fin["mean_lines"], final_pieces=fin["mean_pieces"], median_final=fin["median_score"],
                monotone_nonincreasing_after_start=bool(np.all(np.diff(sc) <= 0)),
            )
            summary_rows[(a, s)] = d
            ckpts[f"{a}_s{s}_final"] = MOD / a / f"seed{s}" / f"dqn_step{d['final_step']}.pt"
            ckpts[f"{a}_s{s}_bestrl"] = MOD / a / f"seed{s}" / f"dqn_step{d['best_rl_step']}.pt"
    out["start_best_final"] = {f"{a}/seed{s}": v for (a, s), v in summary_rows.items()}

    # ---- per-game paired comparisons (clone vs final/best, and vs λ=0 control) ------------
    pg = per_game_scores(ckpts)
    # sanity: per-game means reproduce the logged eval rows
    chk = {}
    for (a, s), d in summary_rows.items():
        chk[f"{a}/seed{s}/final"] = (float(pg[f"{a}_s{s}_final"].mean()), d["final"])
    out["per_game_reproduces_logged_final"] = chk
    paired = {}
    for s in SEEDS:
        clone = pg[f"clone_s{s}"]
        for a in ARMS:
            for kind in ("final", "bestrl"):
                x = pg[f"{a}_s{s}_{kind}"]
                lo, hi = boot_ci(x - clone)
                paired[f"{a}/seed{s}/{kind}_vs_clone"] = dict(diff=float((x - clone).mean()), lo=lo, hi=hi)
            if a != "B_lam0":
                x, c = pg[f"{a}_s{s}_final"], pg[f"B_lam0_s{s}_final"]
                lo, hi = boot_ci(x - c)
                paired[f"{a}/seed{s}/final_vs_lam0_final"] = dict(diff=float((x - c).mean()), lo=lo, hi=hi)
    out["paired_per_game"] = paired

    # ---- loss composition over the whole run --------------------------------------------------
    comp = {}
    for a in ARMS:
        for s in SEEDS:
            r = [x for x in ev[(a, s)] if x["env_steps"] > 0]
            n = col(r, "n_grad_steps_window")
            wm = lambda k: float(np.nansum(col(r, k) * n) / n.sum())  # noqa: E731
            td, ce, lam = wm("w_td_loss"), wm("w_imit_loss"), LAM[a]
            gtd, gim = float(np.nanmean(col(r, "w_gn_td"))), float(np.nanmean(col(r, "w_gn_imit_unweighted")))
            comp[f"{a}/seed{s}"] = dict(
                td_loss=td, imit_ce=ce, weighted_imit=lam * ce, imit_share_of_loss=lam * ce / (td + lam * ce),
                grad_norm_td=gtd, grad_norm_ce_unweighted=gim, grad_ratio_weighted_imit_over_td=lam * gim / gtd,
                grad_cos=float(np.nanmean(col(r, "w_grad_cos_td_imit"))),
                clip_frac=float(np.nansum(col(r, "grad_clip_frac") * n) / n.sum()),
                total_grad_norm=float(np.nansum(col(r, "grad_norm_mean") * n) / n.sum()),
            )
    out["loss_composition"] = comp

    # ---- budgets ---------------------------------------------------------------------------------
    out["budgets"] = {f"{a}/seed{s}": dict(
        rl_env_steps=info[(a, s)]["rl_env_steps"], grad_steps=info[(a, s)]["grad_steps"],
        expert_demo_transitions=info[(a, s)]["expert_demo_transitions_available"],
        imitation_state_samples=info[(a, s)]["imitation_state_samples"],
        wall_s=info[(a, s)]["elapsed_s"]) for a in ARMS for s in SEEDS}
    (OUT / "analysis.json").write_text(json.dumps(out, indent=2, default=float))

    # ---- markdown tables -------------------------------------------------------------------------
    L = []
    L.append("### Start / best / final validation score (50 games, greedy)\n")
    L.append("| Arm | Seed | Start (A: clone) | Best RL ckpt (step) | Final (10k) | Final / start | ≥85%? | Min over RL evals / start | Best RL / start |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for a in ARMS:
        for s in SEEDS:
            d = summary_rows[(a, s)]
            L.append(f"| {NAME[a]} | {s} | {d['start']:.0f} | {d['best_rl']:.0f} ({d['best_rl_step']}) | {d['final']:.0f} | "
                     f"{d['frac_final']:.1%} | {'yes' if d['passes_85'] else 'no'} | {d['frac_min']:.1%} | {d['frac_best_rl']:.1%} |")
    L.append("\n### Validation mean score by RL step\n")
    steps = [int(x) for x in col(ev[("B_lam0", 42)], "env_steps")]
    L.append("| Arm | Seed | " + " | ".join(str(x) for x in steps) + " |")
    L.append("|---|---|" + "---|" * len(steps))
    for a in ARMS:
        for s in SEEDS:
            L.append(f"| {NAME[a]} | {s} | " + " | ".join(f"{v:.0f}" for v in col(ev[(a, s)], "mean_score")) + " |")
    L.append("\n### Skill-preservation diagnostics, start → final\n")
    L.append("| Arm | Seed | HS top-1 agree | held-out demo top-1 | held-out demo CE | holes/move | ΔAggH/move | lines | pieces | top-out | score/piece | score/line |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in ARMS:
        for s in SEEDS:
            d, r0, r1 = summary_rows[(a, s)], ev[(a, s)][0], ev[(a, s)][-1]
            L.append(f"| {NAME[a]} | {s} | {d['start_hs_top1']:.2f} → {d['final_hs_top1']:.2f} | "
                     f"{d['start_demo_val_top1']:.2f} → {d['final_demo_val_top1']:.2f} | "
                     f"{d['start_demo_val_ce']:.2f} → {d['final_demo_val_ce']:.2f} | "
                     f"{d['start_holes']:.3f} → {d['final_holes']:.3f} ({d['holes_ratio']:.1f}×) | "
                     f"{d['start_agg_h']:.2f} → {d['final_agg_h']:.2f} | {r0['mean_lines']:.0f} → {r1['mean_lines']:.0f} | "
                     f"{r0['mean_pieces']:.0f} → {r1['mean_pieces']:.0f} | {r0['top_out_rate']:.2f} → {r1['top_out_rate']:.2f} | "
                     f"{r0['score_per_piece']:.1f} → {r1['score_per_piece']:.1f} | {r0['score_per_line']:.0f} → {r1['score_per_line']:.0f} |")
    L.append("\n### Effective contribution of each loss (means over the 10k RL steps)\n")
    L.append("| Arm | Seed | TD loss | imitation CE | λ·CE | λ·CE share of loss | ‖∇TD‖ | ‖∇CE‖ (unweighted) | λ‖∇CE‖/‖∇TD‖ | cos(∇TD,∇CE) | total ‖∇‖ pre-clip | clipped |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in ARMS:
        for s in SEEDS:
            c = comp[f"{a}/seed{s}"]
            L.append(f"| {NAME[a]} | {s} | {c['td_loss']:.3f} | {c['imit_ce']:.3f} | {c['weighted_imit']:.3f} | "
                     f"{c['imit_share_of_loss']:.0%} | {c['grad_norm_td']:.1f} | {c['grad_norm_ce_unweighted']:.1f} | "
                     f"{c['grad_ratio_weighted_imit_over_td']:.2f} | {c['grad_cos']:.2f} | {c['total_grad_norm']:.1f} | {c['clip_frac']:.0%} |")
    L.append("\n### Paired per-game differences (same 50 validation games; 95% bootstrap CI over games)\n")
    L.append("| Comparison | Seed | Mean diff | 95% CI |")
    L.append("|---|---|---|---|")
    for k, v in paired.items():
        a, sd, what = k.split("/")
        L.append(f"| {NAME[a]}: {what.replace('_', ' ')} | {sd[4:]} | {v['diff']:+.0f} | [{v['lo']:+.0f}, {v['hi']:+.0f}] |")
    (OUT / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(json.dumps({k: out[k] for k in ("step0_identical_across_arms", "per_game_reproduces_logged_final")}, indent=2))


if __name__ == "__main__":
    main()

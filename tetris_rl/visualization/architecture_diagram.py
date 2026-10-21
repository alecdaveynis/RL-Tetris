"""
Render the DQN architecture as a slide-ready SVG + PNG.

All dimensions, layer types and parameter counts come from the *loaded*
``QNetwork`` (``model_info.describe_network``) and from ``features.py``'s layout,
not from hard-coded numbers.

    python -m tetris_rl.visualization.architecture_diagram \
        --model models/exp_imitreg/C_lam0.1/seed43/dqn_best.pt \
        --out-dir docs/images
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from tetris_rl.engine import BOARD_HEIGHT, BOARD_WIDTH, PIECE_NAMES  # noqa: E402
from tetris_rl.environment import MAX_ACTIONS  # noqa: E402
from tetris_rl.visualization.analysis import (  # noqa: E402
    default_checkpoint,
    load_dqn_agent,
)
from tetris_rl.visualization.model_info import (  # noqa: E402
    SCALAR_FEATURES,
    NetworkInfo,
    describe_network,
    feature_groups,
)

# ---- palette ---------------------------------------------------------------
INK = "#1F2937"
MUTED = "#6B7280"
FAINT = "#D1D5DB"
PAPER = "#FFFFFF"
CORNELL = "#B31B1B"
C_BOARD = "#4C78A8"
C_CUR = "#F58518"
C_NEXT = "#E45756"
C_HEIGHT = "#54A24B"
C_METRIC = "#9D6FB0"
C_LN = "#2A9D8F"
C_HID = ["#2F6FDB", "#2F6FDB", "#2F6FDB", "#2F6FDB"]
C_OUT = "#F2A900"
GROUP_COLORS = {
    "board": C_BOARD,
    "current": C_CUR,
    "next": C_NEXT,
    "heights": C_HEIGHT,
    "scalars": C_METRIC,
}

plt.rcParams.update(
    {
        "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "svg.fonttype": "path",  # render identically everywhere (PowerPoint/Keynote/Slides)
        "figure.dpi": 100,
    }
)


def _text(ax, x, y, s, size=9, color=INK, weight="normal", ha="center", va="center", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, **kw)


def _box(ax, x, y, w, h, fc=PAPER, ec=FAINT, lw=1.0, r=1.0, z=1, alpha=1.0):
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha,
    )
    ax.add_patch(p)
    return p


def _arrow(ax, p0, p1, color=MUTED, lw=1.3, ms=9, z=3, style="-|>", rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle=style, mutation_scale=ms, lw=lw, color=color, zorder=z,
            connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0,
        )
    )


# ---- feature glyphs ----------------------------------------------------------


def _glyph_board(ax, x, y, w, h):
    """Tiny 10×20 board with a plausible stack (decorative)."""
    cw, ch = w / BOARD_WIDTH, h / BOARD_HEIGHT
    ax.add_patch(Rectangle((x, y), w, h, fc="#0F172A", ec="none", zorder=4))
    heights = [5, 6, 6, 4, 5, 7, 6, 3, 4, 5]
    holes = {(1, 3), (4, 1), (7, 1)}
    for c, hh in enumerate(heights):
        for r in range(hh):
            if (c, r) in holes:
                continue
            ax.add_patch(
                Rectangle((x + c * cw, y + r * ch), cw, ch, fc=C_BOARD, ec="#0F172A",
                          lw=0.3, zorder=5)
            )
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=INK, lw=0.8, zorder=6))


def _glyph_onehot(ax, x, y, w, h, hot: int, color: str):
    cw = w / 7
    for i in range(7):
        ax.add_patch(
            Rectangle((x + i * cw, y), cw, h, fc=color if i == hot else "#F3F4F6",
                      ec=INK, lw=0.5, zorder=5)
        )
        _text(ax, x + (i + 0.5) * cw, y + h / 2, PIECE_NAMES[i], size=5.5,
              color=PAPER if i == hot else MUTED, weight="bold", zorder=6)


def _glyph_heights(ax, x, y, w, h):
    vals = [0.45, 0.6, 0.55, 0.35, 0.5, 0.8, 0.65, 0.3, 0.4, 0.55]
    cw = w / len(vals)
    for i, v in enumerate(vals):
        ax.add_patch(
            Rectangle((x + i * cw + 0.12, y), cw - 0.24, v * h, fc=C_HEIGHT, ec="none", zorder=5)
        )
    ax.plot([x, x + w], [y, y], color=INK, lw=0.8, zorder=6)


# ---- main figure -------------------------------------------------------------


def _neuron_ys(yc: float, n_top: int = 5, n_bot: int = 2, gap: float = 4.4, dots: float = 7.2):
    """y-positions of drawn neurons: n_top, an ellipsis gap, n_bot (centered on yc)."""
    total = (n_top - 1 + n_bot - 1) * gap + dots
    top = yc + total / 2
    ys = [top - i * gap for i in range(n_top)]
    bot0 = ys[-1] - dots
    ys += [bot0 - i * gap for i in range(n_bot)]
    return ys, (ys[n_top - 1] + ys[n_top]) / 2


def draw_architecture(
    info: NetworkInfo,
    checkpoint_note: Optional[str] = None,
    out_svg: Optional[Path] = None,
    out_png: Optional[Path] = None,
    dpi: int = 220,
):
    W, H = 160.0, 90.0
    fig = plt.figure(figsize=(16, 9), facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # ---- header
    ax.add_patch(Rectangle((0, H - 1.6), W, 1.6, fc=CORNELL, ec="none", zorder=0))
    _text(ax, 4, H - 6.0, "Placement-Action Q-Network (DQN)", size=25, weight="bold", ha="left")
    _text(
        ax, 4, H - 10.9,
        r"$Q(s,a)=f_\theta(\varphi(s,a))$    scores one candidate placement per forward pass"
        f"    |    {info.total_params:,} trainable parameters",
        size=12, color=MUTED, ha="left",
    )
    _text(ax, W - 4, H - 6.0, "RL-Tetris  ·  Cornell CS 4701", size=11, color=MUTED, ha="right")

    # ---- left: input features ------------------------------------------------
    groups = feature_groups()
    gx, gw = 3.0, 47.0
    top, bottom = 70.5, 17.5
    heights_by_key = {"board": 15.5, "current": 7.4, "next": 7.4, "heights": 8.4, "scalars": 12.6}
    gap = (top - bottom - sum(heights_by_key.values())) / (len(groups) - 1)
    _text(ax, gx, top + 3.3, r"INPUT  $\varphi(s,a)\in\mathbb{R}^{%d}$  -  features of the board after the move"
          % info.input_dim, size=10.2, color=MUTED, weight="bold", ha="left")

    y = top
    anchors = []
    for g in groups:
        h = heights_by_key[g.key]
        y0 = y - h
        col = GROUP_COLORS[g.key]
        _box(ax, gx, y0, gw, h, fc="#FAFAFA", ec=FAINT, lw=0.9, r=0.8)
        ax.add_patch(Rectangle((gx, y0), 1.1, h, fc=col, ec="none", zorder=3))
        title = {
            "board": f"Resulting board ({BOARD_HEIGHT}\u00d7{BOARD_WIDTH})",
            "current": "Current piece",
            "next": "Next piece (preview)",
            "heights": "Column heights",
            "scalars": "Board metrics",
        }[g.key]
        tx = gx + 3.0
        _text(ax, tx, y0 + h - 2.0, title, size=10.5, weight="bold", ha="left")
        if g.key == "scalars":
            _text(ax, tx + 13.0, y0 + h - 2.0, f"{g.size} inputs", size=9.2, color=col,
                  weight="bold", ha="left")
        else:
            _text(ax, tx, y0 + h - 4.5, f"{g.size} inputs", size=9.2, color=col,
                  weight="bold", ha="left")
        if g.key == "board":
            _text(ax, tx, y0 + h - 7.6, "1 = filled cell, 0 = empty", size=8.2, color=MUTED, ha="left")
            _text(ax, tx, y0 + h - 10.0, "after the piece locks", size=8.2, color=MUTED, ha="left")
            _text(ax, tx, y0 + h - 12.4, "and full rows clear", size=8.2, color=MUTED, ha="left")
            _glyph_board(ax, gx + gw - 11.0, y0 + 1.2, 7.4, h - 2.4)
        elif g.key in ("current", "next"):
            _glyph_onehot(ax, gx + gw - 20.0, y0 + h / 2 - 1.8, 17.5, 3.6,
                          hot=2 if g.key == "current" else 5, color=col)
            _text(ax, gx + gw - 11.25, y0 + 0.9, "one-hot", size=7.4, color=MUTED)
        elif g.key == "heights":
            _text(ax, tx, y0 + 1.5, "per column, \u00f7 20", size=8.0, color=MUTED, ha="left")
            _glyph_heights(ax, gx + gw - 16.0, y0 + 1.5, 13.0, h - 3.6)
        else:
            lines = [lab for _, lab in SCALAR_FEATURES]
            for i, lab in enumerate(lines):
                cx = tx + (i // 4) * 21.5
                cy = y0 + h - 5.2 - (i % 4) * 1.95
                _text(ax, cx, cy, lab if i < 5 else lab, size=7.4, color=INK, ha="left")
        anchors.append((gx + gw, y0 + h / 2))
        y = y0 - gap

    # ---- concat bar
    bar_x, bar_w = 56.0, 2.8
    bar_y0, bar_y1 = bottom + 2.0, top - 1.0
    _box(ax, bar_x, bar_y0, bar_w, bar_y1 - bar_y0, fc="#111827", ec="none", r=1.2, z=3)
    for g, (ax_, ay_) in zip(groups, anchors):
        tgt_y = np.interp(ay_, [bottom, top], [bar_y0 + 2.5, bar_y1 - 2.5])
        _arrow(ax, (ax_ + 0.5, ay_), (bar_x - 0.2, tgt_y), color=GROUP_COLORS[g.key], lw=1.8,
               ms=9, rad=0.0)
    _text(ax, bar_x + bar_w / 2, bar_y1 + 3.7, "concatenate", size=9, color=MUTED)
    _text(ax, bar_x + bar_w / 2, bar_y1 + 1.2, f"{info.input_dim}", size=13, weight="bold")

    # ---- network columns -----------------------------------------------------
    stages = []  # (kind, title, n_units, card lines, color)
    if info.use_layer_norm:
        ln = next(l for l in info.layers if l.kind == "layernorm")
        stages.append(("layernorm", "LayerNorm", ln.out_dim,
                       [f"LayerNorm({ln.out_dim})", "per-sample normalization", f"{ln.params:,} params"], C_LN))
    hid = [l for l in info.layers if l.kind == "linear"]
    for i, l in enumerate(hid[:-1]):
        stages.append(("hidden", f"Hidden {i + 1}", l.out_dim,
                       [f"Linear {l.in_dim}\u2192{l.out_dim}", l.activation or "ReLU",
                        f"{l.params:,} params"], C_HID[i % len(C_HID)]))
    out = hid[-1]
    stages.append(("output", "Output", out.out_dim,
                   [f"Linear {out.in_dim}\u2192{out.out_dim}", "no activation",
                    f"{out.params:,} params"], C_OUT))

    n_cols = len(stages)
    x_first, x_last = 76.0, 148.0
    xs = np.linspace(x_first, x_last, n_cols)
    yc = 51.5
    R = 1.6
    col_ys: list[list[float]] = []
    for kind, *_ in stages:
        col_ys.append([yc] if kind == "output" else _neuron_ys(yc)[0])

    # connections (drawn first, under neurons)
    prev_x = bar_x + bar_w
    prev_pts = [(prev_x, y_) for y_ in np.linspace(bar_y1 - 8, bar_y0 + 8, 7)]  # top -> bottom
    first = True
    for ci, (kind, *_rest) in enumerate(stages):
        x = xs[ci]
        pts = [(x, y_) for y_ in col_ys[ci]]
        if kind == "layernorm":
            for (px, py), (qx, qy) in zip(prev_pts, pts):  # element-wise: 1 feature -> 1 unit
                ax.plot([px, qx - R], [py, qy], color=C_LN, lw=0.9, alpha=0.55, zorder=1)
        else:
            for (px, py) in prev_pts:
                for (qx, qy) in pts:
                    ax.plot([px + (0 if first else R), qx - (R * (1.9 if kind == "output" else 1))],
                            [py, qy], color="#64748B", lw=0.5, alpha=0.20, zorder=1)
        first = False
        prev_pts, prev_x = pts, x

    for ci, (kind, title, n_units, card, color) in enumerate(stages):
        x = xs[ci]
        ys = col_ys[ci]
        if kind == "output":
            ax.add_patch(Circle((x, ys[0]), R * 2.0, fc=color, ec=PAPER, lw=1.4, zorder=4))
            _text(ax, x, ys[0], "Q", size=15, color=PAPER, weight="bold", zorder=6)
            _text(ax, x, yc + 12.0, "Output", size=12, weight="bold")
            _text(ax, x, yc + 7.4, r"$Q(s,a)$", size=13, weight="bold")
            _text(ax, x, yc - 7.0, "one scalar", size=8.6, color=MUTED)
            _text(ax, x, yc - 9.1, "per placement", size=8.6, color=MUTED)
        else:
            for y_ in ys:
                ax.add_patch(Circle((x, y_), R, fc=color, ec=PAPER, lw=1.0, zorder=4, alpha=0.95))
            _, dots_y = _neuron_ys(yc)
            for k in (-1.7, 0, 1.7):
                ax.add_patch(Circle((x, dots_y + k), 0.34, fc=INK, ec="none", zorder=5))
            top_y, bot_y = ys[0] + R + 0.6, ys[-1] - R - 0.6
            bx = x + R + 1.5
            ax.plot([bx, bx], [bot_y, top_y], color=MUTED, lw=1.0, zorder=2)
            ax.plot([bx - 0.7, bx], [top_y, top_y], color=MUTED, lw=1.0, zorder=2)
            ax.plot([bx - 0.7, bx], [bot_y, bot_y], color=MUTED, lw=1.0, zorder=2)
            _text(ax, bx + 0.9, yc, f"{n_units}", size=12, weight="bold", color=color, ha="left")
            _text(ax, x, top_y + 3.6, title, size=12, weight="bold")

        # layer card
        cw_, chh = 17.2, 12.6
        cx0, cy0 = x - cw_ / 2, 17.0
        _box(ax, cx0, cy0, cw_, chh, fc="#F9FAFB", ec=FAINT, lw=0.9, r=0.8)
        ax.add_patch(Rectangle((cx0 + 0.7, cy0 + chh - 1.2), cw_ - 1.4, 0.7, fc=color, ec="none",
                               zorder=3))
        _text(ax, x, cy0 + chh - 3.7, card[0], size=9.4, weight="bold")
        if kind == "hidden":
            gxx = x - 3.4
            ax.plot([gxx, gxx + 1.7], [cy0 + 5.35, cy0 + 5.35], color=color, lw=2.0, zorder=5)
            ax.plot([gxx + 1.7, gxx + 3.5], [cy0 + 5.35, cy0 + 7.15], color=color, lw=2.0, zorder=5)
            _text(ax, gxx + 4.4, cy0 + 6.0, card[1], size=9, color=color, weight="bold", ha="left")
        else:
            _text(ax, x, cy0 + 6.0, card[1], size=8.2, color=INK)
        _text(ax, x, cy0 + 1.9, card[2], size=8.2, color=MUTED)

    # ---- bottom: how decisions are made ------------------------------------
    by0, bh = 2.2, 11.6
    _box(ax, 3, by0, W - 6, bh, fc="#F3F4F6", ec="none", r=1.4, z=0)
    _text(ax, 5.0, by0 + bh - 1.8, "AT DECISION TIME", size=8.4, weight="bold", color=MUTED, ha="left")

    hid_str = "\u2192".join(str(l.out_dim) for l in hid)
    items = [
        (r"$N$ legal placements", f"(rotation, column), N \u2264 {MAX_ACTIONS}", C_BOARD, 27),
        (r"$\varphi(s,a_i)$ for each", f"{info.input_dim} features per placement", C_METRIC, 25),
        ("same shared network", f"{info.input_dim}\u2192{hid_str}", C_HID[0], 27),
        (r"$Q_1,\ \ldots,\ Q_N$", "one value per placement", C_OUT, 25),
        (r"$a^*=\arg\max_a\,Q(s,a)$", "highest-Q placement is played", CORNELL, 30),
    ]
    gap_x = (W - 8 - sum(w for *_, w in items)) / (len(items) - 1)
    xcur = 4.0
    prev_right = None
    for label, sub, color, w in items:
        _box(ax, xcur, by0 + 0.9, w, 7.4, fc=PAPER, ec=color, lw=1.6, r=1.0, z=2)
        _text(ax, xcur + w / 2, by0 + 5.6, label, size=10, weight="bold", zorder=4)
        _text(ax, xcur + w / 2, by0 + 2.6, sub, size=7.8, color=MUTED, zorder=4)
        if prev_right is not None:
            _arrow(ax, (prev_right + 0.3, by0 + 4.6), (xcur - 0.3, by0 + 4.6), color=INK, lw=1.5, ms=10)
        prev_right = xcur + w
        xcur += w + gap_x

    if checkpoint_note:
        _text(ax, W - 3.5, 0.8, checkpoint_note, size=7, color=MUTED, ha="right")

    if out_svg:
        Path(out_svg).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_svg, format="svg", facecolor=PAPER)
    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, format="png", dpi=dpi, facecolor=PAPER)
    return fig


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default=None, help="checkpoint (default: best available)")
    ap.add_argument("--out-dir", default="docs/images")
    ap.add_argument("--name", default="dqn_architecture")
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args(argv)

    ckpt = Path(args.model) if args.model else default_checkpoint()
    if ckpt is None or not ckpt.exists():
        raise SystemExit("No checkpoint found; pass --model PATH")
    agent, meta = load_dqn_agent(ckpt)
    info = describe_network(agent.online)
    note = f"Dimensions read from checkpoint: {meta['path']}"
    out = Path(args.out_dir)
    draw_architecture(info, note, out / f"{args.name}.svg", out / f"{args.name}.png", dpi=args.dpi)
    print(f"wrote {out / (args.name + '.svg')}\nwrote {out / (args.name + '.png')}")


if __name__ == "__main__":
    main()

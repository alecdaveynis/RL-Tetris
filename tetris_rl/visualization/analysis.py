"""
Score every legal placement with the *real* DQN and heuristic agents.

This is the backend of the interactive explorer, but it is plain Python with no
HTTP dependency so it can be unit-tested and reused from notebooks.

Reuse (nothing re-implemented):
  * ``TetrisEngine`` / ``TetrisEnv``    legal placements, landing rows, simulate, step
  * ``features.placement_feature_batch`` the exact φ(s, a) the DQN is trained on
  * ``DQNAgent.q_values / select_action`` Q-values and the greedy choice
  * ``HeuristicAgent.evaluate_placement`` the baseline's per-placement score
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from tetris_rl.agents.dqn_agent import DQNAgent, DQNConfig
from tetris_rl.agents.heuristic_agent import HeuristicAgent
from tetris_rl.agents.random_agent import RandomAgent
from tetris_rl.engine import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    PIECE_NAMES,
    PieceType,
    clear_lines,
    get_shape,
)
from tetris_rl.environment import TetrisEnv
from tetris_rl.features import (
    board_metrics,
    placement_feature_batch,
    state_action_feature_size,
)
from tetris_rl.visualization.model_info import (
    describe_network,
    feature_groups,
    feature_names,
    forward_with_activations,
    linear_layers,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"

#: Preferred default checkpoints, first existing one wins.
DEFAULT_CHECKPOINT_CANDIDATES = (
    "models/exp_imitreg/C_lam0.1/seed43/dqn_best.pt",  # imitation-regularised Double DQN
    "models/improved/dqn_best.pt",  # README default "improved" DQN
    "models/dqn_best.pt",
)


# --------------------------------------------------------------------------
# Checkpoint handling
# --------------------------------------------------------------------------


def default_checkpoint() -> Optional[Path]:
    for rel in DEFAULT_CHECKPOINT_CANDIDATES:
        p = REPO_ROOT / rel
        if p.exists():
            return p
    found = sorted(MODELS_DIR.rglob("dqn_best.pt")) if MODELS_DIR.exists() else []
    return found[0] if found else None


#: Human-readable names for the checkpoints that matter for the project story.
FEATURED_CHECKPOINTS = {
    "models/exp_imitreg/C_lam0.1/seed43/dqn_best.pt": "Imitation-regularised Double DQN (λ=0.1, seed 43) - best",
    "models/exp_imitreg/C_lam0.1/seed42/dqn_best.pt": "Imitation-regularised Double DQN (λ=0.1, seed 42) - best",
    "models/exp_guided/B_bc/seed43/dqn_best.pt": "Behaviour clone of the heuristic (seed 43)",
    "models/exp_guided/B_bc/seed42/dqn_best.pt": "Behaviour clone of the heuristic (seed 42)",
    "models/improved/dqn_best.pt": "Improved Double DQN, trained from scratch",
    "models/original/dqn_best.pt": "Original DQN, trained from scratch",
}


def list_checkpoints(models_dir: Path = MODELS_DIR) -> list[dict[str, Any]]:
    """All ``*.pt`` files under ``models/`` (path relative to the repo, size in MB)."""
    out: list[dict[str, Any]] = []
    if not models_dir.exists():
        return out
    for p in sorted(models_dir.rglob("*.pt")):
        rel = str(p.relative_to(REPO_ROOT))
        out.append(
            {
                "path": rel,
                "size_mb": round(p.stat().st_size / 1e6, 1),
                "featured": FEATURED_CHECKPOINTS.get(rel),
            }
        )
    return out


def load_dqn_agent(path: str | Path) -> tuple[DQNAgent, dict[str, Any]]:
    """
    Load the *trained* online network from a checkpoint into a real ``DQNAgent``.

    The architecture is read from the saved weights (``state_dict`` shapes), so it is
    correct even if the stored config were missing or stale. The replay buffer and
    optimizer in the checkpoint are not touched (they can be hundreds of MB).
    Returns ``(agent, meta)`` with ε forced to 0 (greedy).
    """
    path = Path(path).resolve()
    ckpt: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["online"]

    lin_keys = sorted(
        (int(k.split(".")[1]), k) for k in sd if k.startswith("net.") and k.endswith(".weight")
    )
    weights = [sd[k] for _, k in lin_keys]
    in_dim = int(weights[0].shape[1])
    expected = state_action_feature_size()
    if in_dim != expected:
        raise ValueError(
            f"{path.name}: first layer expects {in_dim} inputs but the current feature "
            f"extractor produces {expected}; checkpoint is from an incompatible version."
        )
    hidden = tuple(int(w.shape[0]) for w in weights[:-1])
    use_ln = "input_norm.weight" in sd

    valid = {f for f in DQNConfig.__dataclass_fields__}
    cfg_dict = {k: v for k, v in (ckpt.get("config") or {}).items() if k in valid}
    cfg = DQNConfig(**cfg_dict)
    stored = (tuple(cfg.hidden_sizes), bool(cfg.use_layer_norm))
    cfg.hidden_sizes = hidden
    cfg.use_layer_norm = use_ln
    cfg.device = "cpu"
    cfg.seed = None

    agent = DQNAgent(cfg)
    agent.online.load_state_dict(sd)
    agent.online.eval()
    agent.epsilon = 0.0

    meta = {
        "path": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
        "name": path.name,
        "size_mb": round(path.stat().st_size / 1e6, 1),
        "checkpoint_version": ckpt.get("checkpoint_version"),
        "total_steps": ckpt.get("total_steps"),
        "grad_steps": ckpt.get("grad_steps"),
        "episodes_trained": ckpt.get("episodes_trained"),
        "best_eval_score": _finite(ckpt.get("best_eval_score")),
        "double_dqn": bool(cfg.double_dqn),
        "lr": cfg.lr,
        "gamma": cfg.gamma,
        "config_matches_weights": stored == (hidden, use_ln),
    }
    return agent, meta


def _finite(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


# --------------------------------------------------------------------------
# State <-> JSON helpers
# --------------------------------------------------------------------------


def board_to_str(board: np.ndarray) -> str:
    return "".join(str(int(v)) for v in board.ravel())


def parse_board(data: Any) -> np.ndarray:
    """Accept a 200-char string, a flat list, or a 20×10 nested list."""
    if isinstance(data, str):
        flat = [int(ch) for ch in data]
    else:
        flat = [int(v) for v in np.asarray(data).ravel()]
    if len(flat) != BOARD_HEIGHT * BOARD_WIDTH:
        raise ValueError(f"board must have {BOARD_HEIGHT * BOARD_WIDTH} cells, got {len(flat)}")
    arr = np.array(flat, dtype=np.int8).reshape(BOARD_HEIGHT, BOARD_WIDTH)
    if arr.min() < 0 or arr.max() > 8:
        raise ValueError("board cells must be in 0..8")
    return arr


def make_env(board: np.ndarray, current: int, nxt: int, seed: Optional[int] = None) -> TetrisEnv:
    """
    Build a ``TetrisEnv`` positioned on an arbitrary (board, current, next) state.

    The env/engine are the project's own; we only overwrite the engine's state
    fields and ask the env to re-enumerate legal placements (exactly what
    ``TetrisEnv.reset``/``step`` do after any change).
    """
    env = TetrisEnv(seed=seed)
    env.reset(seed=seed)
    eng = env.engine
    eng.board = board.astype(np.int8).copy()
    eng.current_piece = PieceType(int(current))
    eng.next_piece = PieceType(int(nxt))
    eng.game_over = False
    eng._bag = []  # later draws start a fresh 7-bag
    env._refresh_legal()
    return env


def state_dict_of(env: TetrisEnv, **extra: Any) -> dict[str, Any]:
    e = env.engine
    return {
        "board": board_to_str(e.board),
        "current": int(e.current_piece),
        "next": int(e.next_piece),
        "pieces_placed": int(e.pieces_placed),
        "score": int(e.score),
        "lines": int(e.lines_cleared_total),
        "game_over": bool(e.game_over),
        **extra,
    }


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode("ascii")


# --------------------------------------------------------------------------
# The explorer backend
# --------------------------------------------------------------------------


class PlacementExplorer:
    """Holds one loaded DQN checkpoint and answers analysis queries."""

    def __init__(self, checkpoint: str | Path) -> None:
        self.heuristic = HeuristicAgent(seed=0)
        self.load(checkpoint)

    # ---- model ----------------------------------------------------------
    def load(self, checkpoint: str | Path) -> None:
        agent, meta = load_dqn_agent(checkpoint)
        self.agent, self.meta = agent, meta
        self.net_info = describe_network(agent.online)

    def info(self) -> dict[str, Any]:
        return {
            "checkpoint": self.meta,
            "network": self.net_info.to_dict(),
            "feature_groups": [
                {"key": g.key, "label": g.label, "start": g.start, "size": g.size, "detail": g.detail}
                for g in feature_groups()
            ],
            "feature_names": feature_names(),
            "piece_names": PIECE_NAMES,
            "piece_shapes": {
                name: [[int(r), int(c)] for r, c in get_shape(PieceType(i), 0)]
                for i, name in enumerate(PIECE_NAMES)
            },
            "board": {"height": BOARD_HEIGHT, "width": BOARD_WIDTH},
            "heuristic_weights": dict(self.heuristic.weights.__dict__),
        }

    def weights(self) -> dict[str, Any]:
        """Real trained parameters (float32, base64) for client-side connection views."""
        net = self.agent.online
        out: dict[str, Any] = {"layers": []}
        for lin in linear_layers(net):
            out["layers"].append(
                {
                    "in": lin.in_features,
                    "out": lin.out_features,
                    "W": _b64(lin.weight.detach().numpy()),  # (out, in) row-major
                    "b": _b64(lin.bias.detach().numpy()),
                }
            )
        return out

    # ---- analysis -------------------------------------------------------
    def analyze(self, board: Any, current: int, nxt: int) -> dict[str, Any]:
        b = parse_board(board)
        env = make_env(b, current, nxt, seed=0)
        return self._analyze_env(env)

    def _analyze_env(self, env: TetrisEnv) -> dict[str, Any]:
        eng = env.engine
        placements = env.legal_placements()
        n = len(placements)
        base = {
            "board": board_to_str(eng.board),
            "current": int(eng.current_piece),
            "next": int(eng.next_piece),
            "n_legal": n,
            "stages": [],
            "placements": [],
            "dqn_choice": None,
            "heuristic_choices": [],
            "agree": None,
        }
        if n == 0:
            base["game_over"] = True
            return base

        feats = placement_feature_batch(eng, placements)
        q = self.agent.q_values(feats)
        dqn_idx = int(self.agent.select_action(env, explore=False))
        assert dqn_idx == int(np.argmax(q))

        h_scores = np.array(
            [self.heuristic.evaluate_placement(env, i) for i in range(n)], dtype=np.float64
        )
        h_best = float(h_scores.max())
        h_ties = [int(i) for i in np.flatnonzero(np.isclose(h_scores, h_best))]

        q_order = np.argsort(-q, kind="stable")
        q_rank = np.empty(n, dtype=int)
        q_rank[q_order] = np.arange(1, n + 1)
        h_order = np.argsort(-h_scores, kind="stable")
        h_rank = np.empty(n, dtype=int)
        h_rank[h_order] = np.arange(1, n + 1)

        scalar_start = feature_groups()[-1].start
        plist = []
        for i, pl in enumerate(placements):
            landing = eng.hard_drop_row(pl.piece, pl.rotation, pl.column)
            cells = eng.cells_at(pl.piece, pl.rotation, landing, pl.column)
            new_board, lines, score = eng.simulate_placement(pl)
            # Where do the piece's cells end up *after* line clears? Mark them with a
            # sentinel value and let the engine's own clear_lines move them.
            marked = eng.board.copy()
            for r, c in cells:
                marked[r, c] = 9
            _, marked = clear_lines(marked)
            placed = np.argwhere(marked == 9)
            m = board_metrics(new_board)
            plist.append(
                {
                    "index": i,
                    "rotation": int(pl.rotation),
                    "column": int(pl.column),
                    "cells": [[int(r), int(c)] for r, c in cells],
                    "result_board": board_to_str(new_board),
                    "placed_after_clear": [[int(r), int(c)] for r, c in placed],
                    "lines_cleared": int(lines),
                    "score_gained": int(score),
                    "q": float(q[i]),
                    "q_rank": int(q_rank[i]),
                    "heuristic_score": float(h_scores[i]),
                    "heuristic_rank": int(h_rank[i]),
                    "heights": [int(v) for v in m["heights"]],
                    "holes": int(m["holes"]),
                    "aggregate_height": int(m["aggregate_height"]),
                    "bumpiness": int(m["bumpiness"]),
                    "max_height": int(m["max_height"]),
                    "scalar_features": [round(float(v), 5) for v in feats[i, scalar_start:]],
                }
            )

        base.update(
            placements=plist,
            dqn_choice=dqn_idx,
            heuristic_choices=h_ties,
            agree=dqn_idx in h_ties,
            q_min=float(q.min()),
            q_max=float(q.max()),
            stages=self._stage_activations(feats, q),
            game_over=False,
        )
        return base

    def _stage_activations(self, feats: np.ndarray, q: np.ndarray) -> list[dict[str, Any]]:
        """
        Per-stage activations for every candidate: input, LayerNorm output,
        post-ReLU hidden layers, and Q. ``values[i]`` belongs to placement ``i``.
        """
        net = self.agent.online
        trace = forward_with_activations(net, feats)
        stages: list[dict[str, Any]] = []
        hidden_no = 0
        for t in trace:
            v = t["values"]
            if t["kind"] == "input":
                stages.append({"name": "input", "label": "Input φ(s,a)", "values": _round(v, 4)})
            elif t["kind"] == "layernorm":
                stages.append({"name": "layernorm", "label": "LayerNorm", "values": _round(v, 4)})
            elif t["kind"] == "act":
                hidden_no += 1
                stages.append(
                    {"name": f"hidden{hidden_no}", "label": f"Hidden {hidden_no} (ReLU)",
                     "values": _round(v, 4)}
                )
        # The final Linear's output is Q (no activation); use the agent's own q_values.
        stages.append({"name": "output", "label": "Q(s,a)", "values": [[float(x)] for x in q]})
        # Q should equal the traced final linear output exactly.
        last = trace[-1]["values"].reshape(-1)
        if not np.allclose(last, q, atol=1e-5):
            raise RuntimeError("traced forward pass disagrees with agent.q_values")
        return stages

    # ---- state generation ----------------------------------------------
    def sample_state(self, kind: str, seed: int, n_pieces: int) -> dict[str, Any]:
        """Play ``n_pieces`` moves from an empty board with the chosen agent."""
        env = TetrisEnv(seed=seed)
        env.reset(seed=seed)
        policy = self._policy(kind, seed)
        last_good = state_dict_of(env)
        for _ in range(max(0, n_pieces)):
            if env.engine.game_over or not env.legal_placements():
                break
            env.step(policy.select_action(env, explore=False))
            if env.engine.game_over or not env.legal_placements():
                break
            last_good = state_dict_of(env)
        return last_good

    def step(self, board: Any, current: int, nxt: int, index: int,
             seed: Optional[int] = None) -> dict[str, Any]:
        """Lock legal placement ``index``; the new next-piece is drawn at random."""
        env = make_env(parse_board(board), current, nxt, seed=seed)
        if not 0 <= index < len(env.legal_placements()):
            raise ValueError(f"placement index {index} out of range")
        _, _, terminated, _, info = env.step(index)
        return state_dict_of(
            env,
            lines_cleared=int(info.get("lines_cleared", 0)),
            score_gained=int(info.get("score_gained", 0)),
            terminated=bool(terminated),
        )

    def next_disagreement(self, board: Any, current: int, nxt: int,
                          seed: Optional[int] = None, max_steps: int = 300) -> dict[str, Any]:
        """
        Roll the *heuristic* forward from the given state and return the first later
        state where the DQN's greedy choice differs from the heuristic's.
        """
        env = make_env(parse_board(board), current, nxt, seed=seed)
        heur = HeuristicAgent(seed=seed)
        for step in range(1, max_steps + 1):
            if env.engine.game_over or not env.legal_placements():
                return {"found": False, "reason": "game over before a disagreement"}
            env.step(heur.select_action(env, explore=False))
            if env.engine.game_over or not env.legal_placements():
                return {"found": False, "reason": "game over before a disagreement"}
            feats = placement_feature_batch(env.engine, env.legal_placements())
            dqn_idx = int(np.argmax(self.agent.q_values(feats)))
            scores = np.array(
                [self.heuristic.evaluate_placement(env, i) for i in range(len(feats))]
            )
            ties = np.flatnonzero(np.isclose(scores, scores.max()))
            if dqn_idx not in ties:
                return {"found": True, "after_moves": step, "state": state_dict_of(env)}
        return {"found": False, "reason": f"no disagreement within {max_steps} moves"}

    def _policy(self, kind: str, seed: int):
        if kind == "dqn":
            return self.agent
        if kind == "random":
            return RandomAgent(seed=seed)
        return HeuristicAgent(seed=seed)


def _round(a: np.ndarray, nd: int) -> list[list[float]]:
    return np.round(a.astype(np.float64), nd).tolist()


__all__ = [
    "PlacementExplorer",
    "load_dqn_agent",
    "default_checkpoint",
    "list_checkpoints",
    "make_env",
    "parse_board",
    "board_to_str",
]

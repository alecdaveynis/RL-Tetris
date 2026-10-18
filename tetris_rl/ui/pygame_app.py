"""
Pygame Tetris UI: human play + AI demo (random / heuristic / DQN).

Independent of training — safe to run while training headlessly elsewhere.

Controls (human mode):
  Left/Right  – move
  Up          – rotate
  Down        – soft drop one row (preview only; lock via Space)
  Space       – hard drop / lock
  P           – pause
  R           – restart
  1/2/3/4     – Human / Random / Heuristic / DQN
  +/-         – AI playback speed
  Esc         – quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from tetris_rl.agents.dqn_agent import DQNAgent
from tetris_rl.agents.heuristic_agent import HeuristicAgent
from tetris_rl.agents.random_agent import RandomAgent
from tetris_rl.engine import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    PIECE_NAMES,
    PieceType,
    Placement,
    TetrisEngine,
    get_shape,
)
from tetris_rl.environment import TetrisEnv

# Colors
BLACK = (12, 14, 20)
GRID = (40, 44, 56)
WHITE = (230, 232, 240)
MUTED = (140, 148, 168)
ACCENT = (80, 180, 220)

PIECE_COLORS = {
    0: (0, 0, 0),
    int(PieceType.I) + 1: (0, 220, 220),
    int(PieceType.O) + 1: (220, 200, 40),
    int(PieceType.T) + 1: (180, 80, 200),
    int(PieceType.S) + 1: (80, 200, 80),
    int(PieceType.Z) + 1: (220, 60, 60),
    int(PieceType.J) + 1: (60, 100, 220),
    int(PieceType.L) + 1: (230, 140, 40),
}

CELL = 30
MARGIN = 24


class GameApp:
    def __init__(
        self,
        mode: str = "human",
        model_path: Optional[str] = None,
        seed: int = 42,
        device: str = "cpu",
        ai_delay_ms: int = 250,
    ) -> None:
        self.mode = mode
        self.model_path = model_path
        self.seed = seed
        self.device = device
        self.ai_delay_ms = ai_delay_ms
        self.paused = False
        self.engine = TetrisEngine(seed=seed)
        self.env = TetrisEnv(seed=seed)
        self.env.reset(seed=seed)
        self.engine = self.env.engine

        # Human ghost piece state
        self.human_rot = 0
        self.human_col = 3
        self._sync_human_from_current()

        self.agents = {
            "random": RandomAgent(seed=seed),
            "heuristic": HeuristicAgent(seed=seed),
        }
        self.dqn: Optional[DQNAgent] = None
        if model_path and Path(model_path).exists():
            self.dqn = DQNAgent.from_checkpoint(model_path, device=device, explore=False)
            self.agents["dqn"] = self.dqn

        self._last_ai_ms = 0
        self._pending_placement: Optional[Placement] = None
        self._anim_board: Optional[np.ndarray] = None

    def _sync_human_from_current(self) -> None:
        self.human_rot = 0
        # Center-ish spawn column
        shape = get_shape(self.engine.current_piece, 0)
        width = max(c for _, c in shape) - min(c for _, c in shape) + 1
        self.human_col = max(0, (BOARD_WIDTH - width) // 2 - min(c for _, c in shape))

    def restart(self) -> None:
        self.seed += 1
        self.env = TetrisEnv(seed=self.seed)
        self.env.reset(seed=self.seed)
        self.engine = self.env.engine
        self._sync_human_from_current()
        self.paused = False
        self._pending_placement = None

    def current_agent_label(self) -> str:
        return {
            "human": "Human",
            "random": "Random AI",
            "heuristic": "Heuristic AI",
            "dqn": "Trained DQN",
        }.get(self.mode, self.mode)

    def try_move_human(self, d_col: int = 0, d_rot: int = 0) -> None:
        if self.engine.game_over or self.mode != "human":
            return
        new_rot = (self.human_rot + d_rot) % 4
        new_col = self.human_col + d_col
        # Validate by checking hard-drop exists (horizontal bounds)
        if self.engine.hard_drop_row(self.engine.current_piece, new_rot, new_col) is not None:
            self.human_rot = new_rot
            self.human_col = new_col

    def hard_drop_human(self) -> None:
        if self.engine.game_over or self.mode != "human":
            return
        placement = Placement(
            rotation=self.human_rot,
            column=self.human_col,
            piece=self.engine.current_piece,
        )
        # Map to env legal index if present; otherwise step engine directly via env
        legal = self.env.legal_placements()
        for i, p in enumerate(legal):
            if (
                p.rotation == placement.rotation
                and p.column == placement.column
                and p.piece == placement.piece
            ):
                self.env.step(i)
                self.engine = self.env.engine
                self._sync_human_from_current()
                return
        # Fallback: find placement with same locked cells
        landing = self.engine.hard_drop_row(placement.piece, placement.rotation, placement.column)
        if landing is None:
            return
        target = set(self.engine.cells_at(placement.piece, placement.rotation, landing, placement.column))
        for i, p in enumerate(legal):
            l2 = self.engine.hard_drop_row(p.piece, p.rotation, p.column)
            if l2 is None:
                continue
            cells = set(self.engine.cells_at(p.piece, p.rotation, l2, p.column))
            if cells == target:
                self.env.step(i)
                self.engine = self.env.engine
                self._sync_human_from_current()
                return

    def ai_step(self) -> None:
        if self.engine.game_over or self.mode == "human" or self.paused:
            return
        if self.mode == "dqn" and self.dqn is None:
            return
        agent = self.agents.get(self.mode)
        if agent is None:
            return
        if self.env.legal_placements():
            action = agent.select_action(self.env, explore=False)
            self.env.step(action)
            self.engine = self.env.engine
            self._sync_human_from_current()

    def run(self) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise SystemExit(
                "pygame is required for the UI. Install with: pip install pygame"
            ) from exc

        pygame.init()
        sidebar = 220
        width = MARGIN * 2 + BOARD_WIDTH * CELL + sidebar
        height = MARGIN * 2 + BOARD_HEIGHT * CELL
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Tetris-RL")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("menlo", 18)
        font_sm = pygame.font.SysFont("menlo", 14)
        font_lg = pygame.font.SysFont("menlo", 22, bold=True)

        running = True
        while running:
            now = pygame.time.get_ticks()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_p:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        self.restart()
                    elif event.key == pygame.K_1:
                        self.mode = "human"
                    elif event.key == pygame.K_2:
                        self.mode = "random"
                    elif event.key == pygame.K_3:
                        self.mode = "heuristic"
                    elif event.key == pygame.K_4:
                        if self.dqn is not None:
                            self.mode = "dqn"
                    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                        self.ai_delay_ms = max(50, self.ai_delay_ms - 50)
                    elif event.key == pygame.K_MINUS:
                        self.ai_delay_ms = min(1000, self.ai_delay_ms + 50)
                    elif self.mode == "human" and not self.paused:
                        if event.key == pygame.K_LEFT:
                            self.try_move_human(d_col=-1)
                        elif event.key == pygame.K_RIGHT:
                            self.try_move_human(d_col=1)
                        elif event.key == pygame.K_UP:
                            self.try_move_human(d_rot=1)
                        elif event.key == pygame.K_SPACE:
                            self.hard_drop_human()
                        elif event.key == pygame.K_DOWN:
                            # Soft drop visual: shift ghost down is hard-drop only;
                            # nudge column is N/A — treat as hard drop preview confirm
                            pass

            if self.mode != "human" and not self.paused and not self.engine.game_over:
                if now - self._last_ai_ms >= self.ai_delay_ms:
                    self.ai_step()
                    self._last_ai_ms = now

            # Draw
            screen.fill(BLACK)
            ox, oy = MARGIN, MARGIN

            # Board background
            pygame.draw.rect(
                screen,
                GRID,
                pygame.Rect(ox - 2, oy - 2, BOARD_WIDTH * CELL + 4, BOARD_HEIGHT * CELL + 4),
                border_radius=4,
            )

            board = self.engine.board
            for r in range(BOARD_HEIGHT):
                for c in range(BOARD_WIDTH):
                    val = int(board[r, c])
                    color = PIECE_COLORS.get(val, MUTED) if val else (22, 26, 34)
                    rect = pygame.Rect(ox + c * CELL, oy + r * CELL, CELL - 1, CELL - 1)
                    pygame.draw.rect(screen, color, rect, border_radius=3)

            # Ghost / active piece for human
            if self.mode == "human" and not self.engine.game_over:
                landing = self.engine.hard_drop_row(
                    self.engine.current_piece, self.human_rot, self.human_col
                )
                if landing is not None:
                    cells = self.engine.cells_at(
                        self.engine.current_piece, self.human_rot, landing, self.human_col
                    )
                    ghost_color = PIECE_COLORS[int(self.engine.current_piece) + 1]
                    for r, c in cells:
                        if 0 <= r < BOARD_HEIGHT and 0 <= c < BOARD_WIDTH:
                            rect = pygame.Rect(ox + c * CELL, oy + r * CELL, CELL - 1, CELL - 1)
                            s = pygame.Surface((CELL - 1, CELL - 1), pygame.SRCALPHA)
                            s.fill((*ghost_color, 90))
                            screen.blit(s, rect.topleft)

            # Sidebar
            sx = ox + BOARD_WIDTH * CELL + 20
            y = oy
            title = font_lg.render("Tetris-RL", True, WHITE)
            screen.blit(title, (sx, y))
            y += 36
            agent_surf = font.render(self.current_agent_label(), True, ACCENT)
            screen.blit(agent_surf, (sx, y))
            y += 28
            for label, value in [
                ("Score", str(self.engine.score)),
                ("Lines", str(self.engine.lines_cleared_total)),
                ("Pieces", str(self.engine.pieces_placed)),
                ("Next", PIECE_NAMES[int(self.engine.next_piece)]),
                ("Speed", f"{self.ai_delay_ms} ms"),
            ]:
                screen.blit(font_sm.render(f"{label}: {value}", True, MUTED), (sx, y))
                y += 22

            y += 10
            # Next piece preview
            screen.blit(font_sm.render("Preview", True, WHITE), (sx, y))
            y += 24
            preview = get_shape(self.engine.next_piece, 0)
            pc = PIECE_COLORS[int(self.engine.next_piece) + 1]
            for dr, dc in preview:
                rect = pygame.Rect(sx + dc * 20, y + dr * 20, 18, 18)
                pygame.draw.rect(screen, pc, rect, border_radius=3)
            y += 90

            help_lines = [
                "1 Human  2 Random",
                "3 Heuristic  4 DQN",
                "Arrows/Space play",
                "P pause  R restart",
                "+/- AI speed",
            ]
            for line in help_lines:
                screen.blit(font_sm.render(line, True, MUTED), (sx, y))
                y += 18

            if self.paused:
                overlay = font_lg.render("PAUSED", True, WHITE)
                screen.blit(overlay, (ox + 60, oy + BOARD_HEIGHT * CELL // 2))
            if self.engine.game_over:
                overlay = font_lg.render("GAME OVER", True, (220, 80, 80))
                screen.blit(overlay, (ox + 40, oy + BOARD_HEIGHT * CELL // 2))
                screen.blit(
                    font_sm.render("Press R to restart", True, MUTED),
                    (ox + 50, oy + BOARD_HEIGHT * CELL // 2 + 28),
                )
            if self.mode == "dqn" and self.dqn is None:
                screen.blit(
                    font_sm.render("No DQN model loaded", True, (220, 120, 80)),
                    (sx, height - 40),
                )

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tetris-RL Pygame UI")
    p.add_argument(
        "--mode",
        type=str,
        default="human",
        choices=["human", "random", "heuristic", "dqn"],
    )
    p.add_argument("--model", type=str, default="models/dqn_best.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--ai-delay-ms", type=int, default=250)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    app = GameApp(
        mode=args.mode,
        model_path=args.model,
        seed=args.seed,
        device=args.device,
        ai_delay_ms=args.ai_delay_ms,
    )
    app.run()


if __name__ == "__main__":
    main()

"""
Local browser app to explore the trained placement DQN.

    python -m tetris_rl.visualization.explorer                       # default checkpoint
    python -m tetris_rl.visualization.explorer --model models/improved/dqn_best.pt

Serves a single self-contained page (no CDN / internet needed) plus a small JSON
API on 127.0.0.1. All numbers shown come from the real checkpoint, engine and
agents via ``analysis.PlacementExplorer``.
"""

from __future__ import annotations

import argparse
import json
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from tetris_rl.visualization.analysis import (
    MODELS_DIR,
    REPO_ROOT,
    PlacementExplorer,
    default_checkpoint,
    list_checkpoints,
)

STATIC_DIR = Path(__file__).parent / "static"


def _resolve_model(rel: str) -> Path:
    """Only allow loading ``*.pt`` files inside the repo's ``models/`` directory."""
    p = (REPO_ROOT / rel).resolve()
    if p.suffix != ".pt" or not p.is_file():
        raise ValueError(f"not a checkpoint file: {rel}")
    if not p.is_relative_to(MODELS_DIR.resolve()):
        raise ValueError("checkpoints must live under models/")
    return p


class _State:
    def __init__(self, explorer: PlacementExplorer) -> None:
        self.explorer = explorer
        self.lock = threading.Lock()  # torch + engine objects are not shared across requests


def make_handler(state: _State):
    class Handler(BaseHTTPRequestHandler):
        server_version = "TetrisDQNExplorer/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet
            pass

        # ---- helpers ----
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: Any, code: int = 200) -> None:
            self._send(code, json.dumps(obj, allow_nan=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _body(self) -> dict[str, Any]:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        # ---- routes ----
        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                if path in ("/", "/index.html"):
                    html = (STATIC_DIR / "explorer.html").read_bytes()
                    self._send(200, html, "text/html; charset=utf-8")
                elif path == "/api/info":
                    with state.lock:
                        self._json(state.explorer.info())
                elif path == "/api/weights":
                    with state.lock:
                        self._json(state.explorer.weights())
                elif path == "/api/checkpoints":
                    self._json({"checkpoints": list_checkpoints(),
                                "current": state.explorer.meta["path"]})
                else:
                    self._send(404, b"not found", "text/plain")
            except Exception as e:  # pragma: no cover - defensive
                traceback.print_exc()
                self._json({"error": str(e)}, 500)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                req = self._body()
                with state.lock:
                    ex = state.explorer
                    if path == "/api/analyze":
                        out = ex.analyze(req["board"], req["current"], req["next"])
                    elif path == "/api/sample":
                        out = ex.sample_state(req.get("agent", "heuristic"),
                                              int(req.get("seed", 0)), int(req.get("pieces", 0)))
                    elif path == "/api/step":
                        out = ex.step(req["board"], req["current"], req["next"],
                                      int(req["index"]), seed=req.get("seed"))
                    elif path == "/api/disagreement":
                        out = ex.next_disagreement(req["board"], req["current"], req["next"],
                                                   seed=req.get("seed"))
                    elif path == "/api/load":
                        ex.load(_resolve_model(req["path"]))
                        out = {"ok": True}
                    else:
                        self._send(404, b"not found", "text/plain")
                        return
                self._json(out)
            except (KeyError, ValueError) as e:
                self._json({"error": str(e)}, 400)
            except Exception as e:  # pragma: no cover - defensive
                traceback.print_exc()
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    return Handler


def serve(checkpoint: Path, host: str = "127.0.0.1", port: int = 8765,
          open_browser: bool = True) -> None:
    print(f"Loading checkpoint: {checkpoint}")
    explorer = PlacementExplorer(checkpoint)
    m = explorer.meta
    print(f"  architecture: {explorer.net_info.input_dim} -> "
          f"{'LayerNorm -> ' if explorer.net_info.use_layer_norm else ''}"
          f"{' -> '.join(str(h) for h in explorer.net_info.hidden_sizes)} -> 1 "
          f"({explorer.net_info.total_params:,} params)")
    print(f"  trained steps={m['total_steps']}, best eval score={m['best_eval_score']}")
    try:
        httpd = ThreadingHTTPServer((host, port), make_handler(_State(explorer)))
    except OSError as e:
        raise SystemExit(f"Cannot listen on {host}:{port} ({e}). Try --port {port + 1}.")
    url = f"http://{host}:{port}/"
    print(f"\nTetris DQN Explorer running at {url}   (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        httpd.server_close()


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default=None, help="checkpoint (default: best available under models/)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = ap.parse_args(argv)

    ckpt = Path(args.model) if args.model else default_checkpoint()
    if ckpt is None or not ckpt.exists():
        raise SystemExit("No checkpoint found. Train one or pass --model PATH (see README).")
    serve(ckpt, args.host, args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()

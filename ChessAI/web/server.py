"""Web server for the ChessAI browser interface.

Serves the static files in ``web/`` and exposes a JSON endpoint
``POST /api/ai-move`` that uses the existing ChessAI classical engine
to choose a move for the requested position.

Usage:
    python -m web.server [--port 8000] [--depth 3] [--no-engine]
    (run from the ChessAI project root)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional
from urllib.parse import urlparse

import chess

# Make project imports work regardless of the current working directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))

# Content types for static file serving.
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".json": "application/json",
    ".ico": "image/x-icon",
}


class _EngineWrapper:
    """Lazy wrapper around the classical engine agent so imports stay fast."""

    def __init__(self, depth: int, enabled: bool) -> None:
        self.depth = depth
        self.enabled = enabled
        self._agent = None

    def _ensure(self):
        if self._agent is None:
            from src.agents.classical_engine import ClassicalEngineAgent

            self._agent = ClassicalEngineAgent(depth=self.depth)
        return self._agent

    def _search(self, fen: str):
        """Search for the best move and evaluate the given position.

        Returns:
            (uci_move, score_cp): uci move (or None) and the evaluation in
            centipawns from White's perspective (positive = good for White).
            The score uses the engine's static evaluator, which is correct
            at any search depth.
        """
        if not self.enabled:
            return None, 0
        board = chess.Board(fen)
        if board.is_game_over() or not board.legal_moves:
            return None, 0
        try:
            from src.engine.board import Board as EngineBoard

            agent = self._ensure()
            wrapper = EngineBoard.from_raw(board)
            move = agent.select_move(board)
            score = agent.engine.evaluate_position(wrapper)
            return move.uci() if move else None, score
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[server] engine error: {exc}", flush=True)
            return None, 0


_ENGINE: _EngineWrapper = None  # type: ignore


def _read_body(handler: BaseHTTPRequestHandler) -> Dict:
    length = int(handler.headers.get("Content-Length") or 0)
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return {}


class _Handler(BaseHTTPRequestHandler):
    server_version = "ChessAI/1.0"

    def log_message(self, fmt, *args):  # keep console clean
        print("[%s] %s" % (self.log_date_time_string(), fmt % args), flush=True)

    # ---- routing -------------------------------------------------------

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/" or path == "":
            path = "/index.html"
        elif path.startswith("/api/"):
            self._send_json(404, {"error": "not found"})
            return
        self._serve_static(path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/ai-move":
            self._handle_ai_move()
        elif parsed.path == "/api/eval":
            self._handle_eval()
        else:
            self._send_json(404, {"error": "not found"})

    # ---- handlers ------------------------------------------------------

    def _handle_ai_move(self) -> None:
        body = _read_body(self)
        fen = body.get("fen")
        depth = int(body.get("depth") or 0)
        if not fen:
            self._send_json(400, {"error": "fen required"})
            return
        if not _ENGINE.enabled:
            self._send_json(200, {"move": None, "engine": False})
            return
        move, score = _ENGINE._search(fen)
        self._send_json(200, {"move": move, "score": score, "engine": True})

    def _handle_eval(self) -> None:
        body = _read_body(self)
        fen = body.get("fen")
        if not fen:
            self._send_json(400, {"error": "fen required"})
            return
        if not _ENGINE.enabled:
            self._send_json(200, {"score": None})
            return
        _, score = _ENGINE._search(fen)
        self._send_json(200, {"score": score})

    def _serve_static(self, path: str) -> None:
        # Prevent path traversal.
        safe = os.path.normpath(path).lstrip("/")
        full = os.path.join(WEB_DIR, safe)
        if not os.path.realpath(full).startswith(os.path.realpath(WEB_DIR)):
            self._send_json(403, {"error": "forbidden"})
            return
        if not os.path.isfile(full):
            self._send_error(404, "Not Found")
            return
        ext = os.path.splitext(full)[1].lower()
        ctype = CONTENT_TYPES.get(ext, "application/octet-stream")
        with open(full, "rb") as fh:
            data = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, status: int, payload: Dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, status: int, message: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(message.encode("utf-8"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ChessAI web server")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address.")
    parser.add_argument("--depth", type=int, default=3, help="Classical engine search depth.")
    parser.add_argument("--no-engine", action="store_true", help="Serve UI without engine moves (fallback JS engine is used).")
    args = parser.parse_args(argv)

    global _ENGINE
    _ENGINE = _EngineWrapper(depth=args.depth, enabled=not args.no_engine)

    httpd = ThreadingHTTPServer((args.host, args.port), _Handler)
    # Show localhost/127.0.0.1 in the URL (not the bind address 0.0.0.0)
    display_host = "localhost" if args.host == "0.0.0.0" else args.host
    print(f"ChessAI web UI: http://{display_host}:{args.port}", flush=True)
    print(f"Engine: {'classical depth=%d' % args.depth if _ENGINE.enabled else 'disabled (JS fallback)'}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

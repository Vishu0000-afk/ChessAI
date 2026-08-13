/* ==========================================================================
   AI module — pluggable chess engine interface.
   ==========================================================================
   Architecture note: the UI never talks to the engine directly. It asks
   an `AIController` for a move on behalf of a given side. The default
   implementation calls the local Python engine backend (`/api/ai-move`)
   exposed by `server.py`. If the backend is unreachable it falls back to
   a lightweight built-in search so the app still works standalone.
   ========================================================================== */

class AIController {
  constructor() {
    this.backendAvailable = false;
    this.depth = 3;
    this._checkingBackend = false;
  }

  async init() {
    if (this._checkingBackend) return;
    this._checkingBackend = true;
    try {
      const res = await fetch("api/ai-move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fen: undefined }),
      });
      this.backendAvailable = res.ok;
    } catch (e) {
      this.backendAvailable = false;
    }
    this._checkingBackend = false;
    return this.backendAvailable;
  }

  /* Compute a move for `side` ('w'|'b') from a chess.js game.
     Returns a Promise resolving to { from, to, promotion } or null. */
  async getMove(game, side, depth) {
    if (this.backendAvailable) {
      try {
        return await this._remoteMove(game, side, depth);
      } catch (e) {
        /* fall through to local engine */
      }
    }
    return this._localMove(game, depth);
  }

  /* Evaluation of the current position in centipawns from White's
     perspective (positive = good for White). Uses the Python engine when
     reachable, otherwise a fast built-in material + piece-square eval. */
  async getEval(game) {
    if (this.backendAvailable) {
      try {
        const res = await fetch("api/eval", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ fen: game.fen() }),
        });
        if (res.ok) {
          const data = await res.json();
          if (data && data.score !== null && data.score !== undefined) return data.score;
        }
      } catch (e) {
        /* fall through to JS eval */
      }
    }
    return this._jsEval(game.fen());
  }

  async _remoteMove(game, side, depth) {
    const res = await fetch("api/ai-move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fen: game.fen(), side, depth: depth || this.depth }),
    });
    if (!res.ok) throw new Error("backend error");
    const data = await res.json();
    if (!data.move) return null;
    const m = data.move; // uci string e.g. "e2e4" or "e7e8q"
    return {
      from: m.slice(0, 2),
      to: m.slice(2, 4),
      promotion: m.length > 4 ? m.slice(4) : undefined,
    };
  }

  /* Minimal negamax search — standalone fallback when no backend is running. */
  _localMove(game, depth) {
    const d = depth || 2;
    const legal = game.moves({ verbose: true });
    if (legal.length === 0) return null;

    const negamax = (g, remaining, alpha, beta, color) => {
      if (remaining === 0) {
        return color * this._eval(g);
      }
      const moves = g.moves({ verbose: true });
      if (moves.length === 0) {
        if (g.inCheck()) return -100000 * color; // mated
        return 0; // stalemate
      }
      let best = -Infinity;
      for (const m of moves) {
        g.move(m);
        let score;
        if (g.isCheckmate()) {
          score = 100000 * color;
        } else if (g.isDraw()) {
          score = 0;
        } else {
          score = -negamax(g, remaining - 1, -beta, -alpha, -color);
        }
        g.undo();
        if (score > best) best = score;
        if (best > alpha) alpha = best;
        if (alpha >= beta) break;
      }
      return best;
    };

    const clone = () => {
      const c = new Chess(game.fen());
      return c;
    };

    const g = clone();
    const color = g.turn() === "w" ? 1 : -1;
    let bestScore = -Infinity;
    let bestMove = null;
    for (const m of legal) {
      g.move(m);
      let score;
      if (g.isCheckmate()) {
        score = 100000 * color;
      } else if (g.isDraw()) {
        score = 0;
      } else {
        score = -negamax(g, d - 1, -Infinity, Infinity, -color);
      }
      g.undo();
      if (score > bestScore) {
        bestScore = score;
        bestMove = m;
      }
    }
    if (!bestMove) {
      const r = legal[Math.floor(Math.random() * legal.length)];
      return { from: r.from, to: r.to, promotion: r.promotion };
    }
    return { from: bestMove.from, to: bestMove.to, promotion: bestMove.promotion };
  }

  /* Static material + positional evaluation, scaled to [0..1]. */
  _eval(g) {
    const values = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };
    let score = 0;
    const board = g.board();
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const sq = board[r][c];
        if (!sq) continue;
        const v = values[sq.type];
        const sign = sq.color === "w" ? 1 : -1;
        score += sign * v;
      }
    }
    return score;
  }

  /* Centipawn evaluation (White's perspective) with piece-square tables.
     Positive = good for White; consistent with the backend's centipawns. */
  _jsEval(fen) {
    const values = { p: 100, n: 320, b: 330, r: 500, q: 900, k: 0 };
    const rows = fen.split(" ")[0].split("/");
    let score = 0;
    for (let r = 0; r < 8; r++) {
      let c = 0;
      for (const ch of rows[r]) {
        if (/\d/.test(ch)) {
          c += parseInt(ch, 10);
          continue;
        }
        const lower = ch.toLowerCase();
        const isWhite = ch === ch.toUpperCase();
        const idx = isWhite ? r * 8 + c : (7 - r) * 8 + c;
        const pst = (PST[lower] || [])[idx] || 0;
        score += isWhite ? values[lower] + pst : -(values[lower] + pst);
        c++;
      }
    }
    return score;
  }
}

/* Simple piece-square tables (White's perspective, mirrored for Black). */
const PST = {
  p: [
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
  ],
  n: [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
  ],
  b: [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
  ],
  r: [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    0, 0, 0, 5, 5, 0, 0, 0,
  ],
  q: [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
  ],
  k: [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20,
  ],
};

window.AIController = AIController;
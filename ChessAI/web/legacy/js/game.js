/* ==========================================================================
   Game — chess game state management built on chess.js.
   Owns the position, move history, and navigation. The UI (app.js)
   subscribes via onChange(state) fired after every position change.
   ========================================================================== */

class Game {
  constructor() {
    this.reset();
  }

  reset() {
    this.chess = new Chess();
    this.startFen = this.chess.fen();
    this.moves = []; // full list of moves played this game: {from,to,promotion,san,color,captured,flags}
    this.pointer = 0; // how many moves are applied to the current position
    this.onChange = null;
    this.resigned = false;
  }

  load(fen) {
    this.chess = new Chess(fen);
    this.startFen = fen;
    this.moves = [];
    this.pointer = 0;
    this.resigned = false;
  }

  /* ---- queries ---- */

  fen() {
    return this.chess.fen();
  }

  turn() {
    return this.chess.turn();
  }

  isGameOver() {
    return this.chess.isGameOver();
  }

  isCheck() {
    return this.chess.inCheck();
  }

  isCheckmate() {
    return this.chess.isCheckmate();
  }

  isStalemate() {
    return this.chess.isStalemate();
  }

  isDraw() {
    return this.chess.isDraw();
  }

  legalMovesFor(square) {
    return this.chess
      .moves({ square, verbose: true })
      .map((m) => ({ from: m.from, to: m.to, capture: !!m.captured, promotion: m.promotion }));
  }

  /* Legal moves for the piece on `square` regardless of whose turn it is.
     Used for premoves (validating the side that is NOT to move). The board
     layout is identical; we just flip the active color in a throwaway clone. */
  premoveMovesFor(square) {
    const parts = this.chess.fen().split(" ");
    parts[1] = parts[1] === "w" ? "b" : "w";
    const flipped = new Chess(parts.join(" "));
    return flipped
      .moves({ square, verbose: true })
      .map((m) => ({ from: m.from, to: m.to, capture: !!m.captured, promotion: m.promotion }));
  }

  findKing(side) {
    const board = this.chess.board();
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const sq = board[r][c];
        if (sq && sq.type === "k" && sq.color === side) return sq.square;
      }
    }
    return null;
  }

  /* ---- mutations ---- */

  makeMove({ from, to, promotion }) {
    let move;
    try {
      move = this.chess.move({ from, to, promotion });
    } catch (e) {
      return null;
    }
    if (!move) return null;
    this.moves.push({
      from: move.from,
      to: move.to,
      promotion: move.promotion,
      san: move.san,
      color: move.color,
      captured: move.captured,
      flags: move.flags,
    });
    this.pointer++;
    return move;
  }

  canUndo() {
    return this.pointer > 0;
  }

  canRedo() {
    return this.pointer < this.moves.length;
  }

  goTo(moveCount) {
    if (moveCount < 0 || moveCount > this.moves.length) return;
    this.pointer = moveCount;
    this._rebuild();
  }

  _rebuild() {
    this.chess = new Chess(this.startFen);
    for (let i = 0; i < this.pointer; i++) {
      const m = this.moves[i];
      this.chess.move({ from: m.from, to: m.to, promotion: m.promotion });
    }
  }

  /* ---- state snapshot for the UI ---- */

  state() {
    return {
      fen: this.fen(),
      turn: this.turn(),
      moves: this.moves,
      pointer: this.pointer,
      check: this.isCheck(),
      checkmate: this.isCheckmate(),
      stalemate: this.isStalemate(),
      draw: this.isDraw(),
      gameOver: this.isGameOver() || this.resigned,
    };
  }

  /* Pieces captured BY `side` ('w'|'b'): pieces the mover removed. */
  capturedBy(side) {
    const out = [];
    for (const m of this.moves.slice(0, this.pointer)) {
      if (m.captured && m.color === side) out.push(m.captured); // captured piece type
    }
    return out;
  }
}

window.Game = Game;
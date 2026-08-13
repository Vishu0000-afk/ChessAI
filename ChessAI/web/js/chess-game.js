/* ChessGame — game state and move validation, backed by the vendored chess.js. */
class ChessGame {
  constructor() {
    this._chess = new Chess();
    this._history = [];
  }

  reset() {
    this._chess.reset();
    this._history = [];
  }

  load(fen) {
    this._chess.load(fen);
    this._history = [];
  }

  fen() {
    return this._chess.fen();
  }

  turn() {
    return this._chess.turn();
  }

  isCheck() {
    return this._chess.isCheck();
  }

  isCheckmate() {
    return this._chess.isCheckmate();
  }

  isStalemate() {
    return this._chess.isStalemate();
  }

  isGameOver() {
    return this._chess.isGameOver();
  }

  isThreefoldRepetition() {
    return this._chess.isThreefoldRepetition();
  }

  board() {
    return this._chess.board();
  }

  get(square) {
    return this._chess.get(square);
  }

  legalMovesFor(square) {
    return this._chess.moves({ square, verbose: true }).map((m) => new Move(m));
  }

  allMoves() {
    return this._chess.moves({ verbose: true }).map((m) => new Move(m));
  }

  makeMove(from, to, promotion) {
    const m = this._chess.move({ from, to, promotion });
    if (!m) return null;
    this._history.push(m);
    return new Move(m);
  }

  undo() {
    const m = this._chess.undo();
    if (!m) return null;
    this._history.pop();
    return new Move(m);
  }

  lastMove() {
    if (!this._history.length) return null;
    return new Move(this._history[this._history.length - 1]);
  }
}
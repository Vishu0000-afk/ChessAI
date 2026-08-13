/* UI/App — window wiring, resize, flip control, game flow. */
class App {
  constructor() {
    this.game = new ChessGame();
    this.boardEl = document.getElementById("board");
    this.coordsEl = document.getElementById("board-coords");
    this.promoEl = document.getElementById("promo-overlay");
    this.bannerEl = document.getElementById("banner");
    this.flipBtn = document.getElementById("flip-btn");

    this.coords = new BoardCoordinates(this.coordsEl);
    this.board = new ChessBoard({
      el: this.boardEl,
      game: this.game,
      promoEl: this.promoEl,
      bannerEl: this.bannerEl,
      onMove: (from, to, promotion) => this._onMove(from, to, promotion),
    });

    this.flipBtn.addEventListener("click", () => this._flip());
    window.addEventListener("keydown", (e) => {
      if (e.key === "r" || e.key === "R") this._flip();
    });
  }

  _flip() {
    this.board.flip();
    this.coords.setOrientation(this.board.orientation);
  }

  _onMove(from, to, promotion) {
    const move = this.game.makeMove(from, to, promotion);
    if (!move) return;
    this.board.applyMove(move);
    this._checkEnd();
  }

  _checkEnd() {
    if (this.game.isCheckmate()) {
      const winner = this.game.turn() === "w" ? "Black" : "White";
      this.board.showBanner("Checkmate — " + winner + " wins");
      this.board.setLocked(true);
    } else if (this.game.isStalemate()) {
      this.board.showBanner("Stalemate — Draw");
      this.board.setLocked(true);
    } else if (this.game.isGameOver()) {
      this.board.showBanner("Draw");
      this.board.setLocked(true);
    }
  }
}

window.addEventListener("DOMContentLoaded", () => {
  window.app = new App();
});
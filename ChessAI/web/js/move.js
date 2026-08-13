/* Move — lightweight wrapper around a chess.js verbose move object. */
class Move {
  constructor(m) {
    this.from = m.from;
    this.to = m.to;
    this.promotion = m.promotion || null;
    this.san = m.san;
    this.lan = m.lan;
    this.color = m.color;
    this.piece = m.piece;
    this.captured = m.captured || null;
    this.flags = m.flags;
    this.isCapture = this.flags.indexOf("c") !== -1;
    this.isEnPassant = this.flags.indexOf("e") !== -1;
    this.isPromotion = this.flags.indexOf("p") !== -1;
    this.isCastle = this.flags.indexOf("k") !== -1 || this.flags.indexOf("q") !== -1;
  }

  uci() {
    return this.from + this.to + (this.promotion || "");
  }

  equals(other) {
    return !!other &&
      this.from === other.from &&
      this.to === other.to &&
      (this.promotion || "") === (other.promotion || "");
  }
}
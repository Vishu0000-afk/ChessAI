/* ChessPiece — a single piece element: rendering, positioning, animation. */
class ChessPiece {
  constructor(type, color, square, board) {
    this.type = type;
    this.color = color;
    this.square = square;
    this._board = board;

    this.el = document.createElement("div");
    this.el.className = "piece " + color;
    this.el.innerHTML = PIECE_SVGS[type + "_" + color];
    board.el.appendChild(this.el);

    this._place(false);
  }

  _place(animate) {
    const t = this._board.transformFor(this.square);
    if (animate) {
      this.el.classList.add("animating");
      this.el.style.transform = t;
    } else {
      this.el.style.transition = "none";
      this.el.style.transform = t;
      void this.el.offsetWidth; /* flush so the initial paint is not animated */
      this.el.style.transition = "";
    }
  }

  moveTo(square) {
    this.square = square;
    this._place(true);
  }

  refresh() {
    this._place(false);
  }

  setDragging(dragging) {
    this.el.classList.toggle("dragging", dragging);
  }

  setGhost(ghost) {
    this.el.classList.toggle("ghost", ghost);
  }

  fadeOut() {
    this.el.classList.add("captured");
    const el = this.el;
    setTimeout(() => el.remove(), 100);
  }

  remove() {
    this.el.remove();
  }
}
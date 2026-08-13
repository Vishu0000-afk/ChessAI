/* ChessBoard — square grid, piece placement, interaction, highlights, flip. */
class ChessBoard {
  constructor(options) {
    this.el = options.el;
    this.game = options.game;
    this.promoEl = options.promoEl;
    this.bannerEl = options.bannerEl;
    this.onMove = options.onMove || function () {};

    this._orientation = "w";
    this._pieces = [];
    this._selected = null;
    this._legalMoves = [];
    this._lastMove = null;
    this._checkSquare = null;
    this._animating = false;
    this._drag = null;
    this._locked = false;

    this._buildSquares();
    this._bindPointer();
    this.setPosition();
  }

  /* ---------- layout ---------- */

  _buildSquares() {
    this._squares = [];
    for (let i = 0; i < 64; i++) {
      const row = Math.floor(i / 8);
      const col = i % 8;
      const light = (row + col) % 2 === 0;
      const sq = document.createElement("div");
      sq.className = "square" + (light ? " light" : " dark");
      this.el.appendChild(sq);
      this._squares.push(sq);
    }
  }

  squareToCR(square) {
    const file = square.charCodeAt(0) - 97;
    const rank = square.charCodeAt(1) - 49;
    if (this._orientation === "w") return { col: file, row: 7 - rank };
    return { col: 7 - file, row: rank };
  }

  transformFor(square) {
    const p = this.squareToCR(square);
    return "translate(" + (p.col * 100) + "%, " + (p.row * 100) + "%)";
  }

  squareFromPoint(clientX, clientY) {
    const rect = this.el.getBoundingClientRect();
    const col = Math.floor((clientX - rect.left) / (rect.width / 8));
    const row = Math.floor((clientY - rect.top) / (rect.height / 8));
    if (col < 0 || col > 7 || row < 0 || row > 7) return null;
    const fileIdx = this._orientation === "w" ? col : 7 - col;
    const rankIdx = this._orientation === "w" ? 7 - row : row;
    return String.fromCharCode(97 + fileIdx) + (rankIdx + 1);
  }

  get orientation() {
    return this._orientation;
  }

  setOrientation(orientation) {
    this._orientation = orientation;
    for (const piece of this._pieces) piece.refresh();
  }

  flip() {
    this.setOrientation(this._orientation === "w" ? "b" : "w");
  }

  /* ---------- position ---------- */

  setPosition() {
    for (const piece of this._pieces) piece.remove();
    this._pieces = [];
    const grid = this.game.board();
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const p = grid[r][c];
        if (!p) continue;
        const square = String.fromCharCode(97 + c) + (8 - r);
        this._pieces.push(new ChessPiece(p.type, p.color, square, this));
      }
    }
  }

  _pieceAt(square) {
    for (const piece of this._pieces) {
      if (piece.square === square) return piece;
    }
    return null;
  }

  _piecesOnSquare(square) {
    const out = [];
    for (const piece of this._pieces) {
      if (piece.square === square) out.push(piece);
    }
    return out;
  }

  /* ---------- highlights ---------- */

  setLastMove(from, to) {
    this.clearLastMove();
    this._lastMove = { from, to };
    this._squareEl(from).classList.add("lastmove");
    this._squareEl(to).classList.add("lastmove");
  }

  clearLastMove() {
    if (!this._lastMove) return;
    this._squareEl(this._lastMove.from).classList.remove("lastmove");
    this._squareEl(this._lastMove.to).classList.remove("lastmove");
    this._lastMove = null;
  }

  setCheck(square) {
    this.clearCheck();
    if (!square) return;
    this._checkSquare = square;
    this._squareEl(square).classList.add("check");
  }

  clearCheck() {
    if (!this._checkSquare) return;
    this._squareEl(this._checkSquare).classList.remove("check");
    this._checkSquare = null;
  }

  clearHighlights() {
    this.clearLastMove();
    this.clearCheck();
    this.clearMarkers();
    this.deselect();
  }

  _squareEl(square) {
    const p = this.squareToCR(square);
    return this._squares[p.row * 8 + p.col];
  }

  /* ---------- selection & legal moves ---------- */

  select(square) {
    this.clearMarkers();
    this._selected = square;
    this._squareEl(square).classList.add("selected");
    this._legalMoves = this.game.legalMovesFor(square);
    for (const m of this._legalMoves) {
      const el = this._squareEl(m.to);
      const marker = document.createElement("div");
      marker.className = "marker " + (m.isCapture ? "ring" : "dot");
      el.appendChild(marker);
    }
  }

  deselect() {
    if (!this._selected) return;
    this._squareEl(this._selected).classList.remove("selected");
    this._selected = null;
    this._legalMoves = [];
  }

  clearMarkers() {
    const els = this.el.querySelectorAll(".marker");
    for (let i = 0; i < els.length; i++) els[i].remove();
  }

  isLegalTarget(from, to) {
    return this.game.legalMovesFor(from).some((m) => m.to === to);
  }

  _resolveMove(from, to) {
    const legal = this.game.legalMovesFor(from);
    const candidates = legal.filter((m) => m.to === to);
    if (!candidates.length) return false;
    if (candidates.some((m) => m.isPromotion)) {
      this._showPromotion(from, to);
      return true;
    }
    this.onMove(from, to, null);
    return true;
  }

  /* ---------- promotion ---------- */

  _showPromotion(from, to) {
    this._promo = { from, to };
    this.promoEl.innerHTML = "";
    const mover = this.game.get(from).color;
    const box = document.createElement("div");
    box.id = "promo-box";
    const types = ["q", "r", "b", "n"];
    for (const t of types) {
      const btn = document.createElement("button");
      btn.innerHTML = PIECE_SVGS[t + "_" + mover];
      btn.addEventListener("click", () => this._pickPromotion(t));
      box.appendChild(btn);
    }
    this.promoEl.appendChild(box);
    this.promoEl.classList.remove("hidden");
  }

  _pickPromotion(type) {
    this.promoEl.classList.add("hidden");
    this.promoEl.innerHTML = "";
    const p = this._promo;
    this._promo = null;
    this.onMove(p.from, p.to, type);
  }

  /* ---------- move animation ---------- */

  applyMove(move) {
    this._animating = true;
    this.clearMarkers();
    this.deselect();
    this.clearLastMove();
    this.clearCheck();

    if (move.isEnPassant) {
      const capSq = move.to[0] + move.from[1];
      const cap = this._pieceAt(capSq);
      if (cap) cap.fadeOut();
      this._pieces = this._pieces.filter((p) => p.square !== capSq);
    } else if (move.isCapture) {
      const cap = this._pieceAt(move.to);
      if (cap) cap.fadeOut();
      this._pieces = this._pieces.filter((p) => p.square !== move.to);
    }

    if (move.isCastle) {
      const white = move.color === "w";
      const rank = white ? "1" : "8";
      const kingFrom = move.from;
      const kingTo = move.to;
      const rookFrom = kingTo === "g" + rank ? "h" + rank : "a" + rank;
      const rookTo = kingTo === "g" + rank ? "f" + rank : "d" + rank;
      const rook = this._pieceAt(rookFrom);
      if (rook) rook.moveTo(rookTo);
    }

    const mover = this._pieceAt(move.from);
    if (mover) mover.moveTo(move.to);

    if (move.isPromotion) {
      const mover = this._pieceAt(move.to);
      if (mover) {
        mover.remove();
        this._pieces = this._pieces.filter((p) => p.square !== move.to);
        this._pieces.push(new ChessPiece(move.promotion, move.color, move.to, this));
      }
    }

    this.setLastMove(move.from, move.to);
    const opponent = move.color === "w" ? "b" : "w";
    if (this.game.isCheck()) {
      const kingSquare = this._findKing(opponent);
      this.setCheck(kingSquare);
    }

    setTimeout(() => { this._animating = false; }, 160);
  }

  _findKing(color) {
    const grid = this.game.board();
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const p = grid[r][c];
        if (p && p.type === "k" && p.color === color) {
          return String.fromCharCode(97 + c) + (8 - r);
        }
      }
    }
    return null;
  }

  /* ---------- pointer interaction ---------- */

  _bindPointer() {
    this.el.addEventListener("pointerdown", (e) => this._onPointerDown(e));
    this.el.addEventListener("pointermove", (e) => this._onPointerMove(e));
    this.el.addEventListener("pointerup", (e) => this._onPointerUp(e));
    this.el.addEventListener("pointercancel", () => this._cancelDrag());
  }

  setLocked(locked) {
    this._locked = locked;
    if (locked) {
      this.clearMarkers();
      this.deselect();
      this._cancelDrag();
    }
  }

  _onPointerDown(e) {
    if (this._animating || this._promo || this._locked) return;
    const square = this.squareFromPoint(e.clientX, e.clientY);
    if (!square) return;

    if (this._selected) {
      if (square === this._selected) {
        this.deselect();
        return;
      }
      if (this.isLegalTarget(this._selected, square)) {
        this._resolveMove(this._selected, square);
        return;
      }
      const p = this.game.get(square);
      if (p && p.color === this.game.turn()) {
        this.deselect();
        this.select(square);
      } else {
        this.deselect();
        return;
      }
    } else {
      const p = this.game.get(square);
      if (!p || p.color !== this.game.turn()) return;
      this.select(square);
    }

    this._drag = {
      from: square,
      pointerId: e.pointerId,
      moved: false,
      startX: e.clientX,
      startY: e.clientY,
    };
    try { this.el.setPointerCapture(e.pointerId); } catch (err) {}
  }

  _onPointerMove(e) {
    if (!this._drag) return;
    const d = this._drag;
    const dist = Math.hypot(e.clientX - d.startX, e.clientY - d.startY);
    if (!d.moved && dist > 4) {
      d.moved = true;
      const piece = this._pieceAt(d.from);
      if (piece) piece.setDragging(true);
    }
    if (!d.moved) return;
    const rect = this.el.getBoundingClientRect();
    const x = e.clientX - rect.left - rect.width / 16;
    const y = e.clientY - rect.top - rect.height / 16;
    const piece = this._pieceAt(d.from);
    if (piece) piece.el.style.transform = "translate(" + x + "px, " + y + "px)";
  }

  _onPointerUp(e) {
    const d = this._drag;
    if (!d) return;
    this._drag = null;
    const piece = this._pieceAt(d.from);
    if (!d.moved) return;
    piece.setDragging(false);
    const square = this.squareFromPoint(e.clientX, e.clientY);
    if (square && this.isLegalTarget(d.from, square)) {
      this._resolveMove(d.from, square);
    } else {
      piece._place(false);
    }
  }

  _cancelDrag() {
    if (!this._drag) return;
    const d = this._drag;
    this._drag = null;
    if (d.moved) {
      const piece = this._pieceAt(d.from);
      if (piece) {
        piece.setDragging(false);
        piece._place(false);
      }
    }
  }

  /* ---------- end banner ---------- */

  showBanner(text) {
    this.bannerEl.textContent = text;
    this.bannerEl.classList.remove("hidden");
  }

  hideBanner() {
    this.bannerEl.classList.add("hidden");
  }
}
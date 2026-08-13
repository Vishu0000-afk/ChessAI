/* ==========================================================================
   Board — reusable chessboard component.
   Renders squares, coordinates, pieces, highlights; handles click-to-move,
   drag-and-drop (via a ghost piece), right-click marks, hover move preview,
   premoves, and a last-move arrow. Talks to the outside world purely
   through callbacks:
     - onMoveRequest({ from, to, promotion })  -> Promise<boolean> success
     - onSelect(square | null)
     - onQueryMoves(square) -> [{from,to,capture}]  (optional)
   All chess rules live in the Game/chess.js layer, never here.
   ========================================================================== */

class Board {
  constructor(container) {
    this.container = container;
    this.orientation = "w"; // 'w' = white at bottom, 'b' = black at bottom
    this.selected = null;
    this.lastMove = null;
    this.checkSquare = null;
    this.hintSquares = [];
    this.legalMoves = []; // { from, to, capture }
    this.allowInput = true;
    this.showCoords = true;
    this.pieces = {}; // square -> element
    this.marks = new Set(); // squares with right-click analysis marks
    this.onMoveRequest = null;
    this.onSelect = null;
    this.onQueryMoves = null;
    this._drag = null;
    this._hoverSquare = null;
    this._hoverMarkers = [];
    this._arrowMove = null;
    this._animCounter = 0;

    this._buildDOM();
  }

  _buildDOM() {
    this.squares = [];
    for (let i = 0; i < 64; i++) {
      const sq = document.createElement("div");
      sq.className = "square";
      this.container.appendChild(sq);
      this.squares.push(sq);
    }
    this._applySquareColors();

    /* move arrow overlay */
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.classList.add("move-arrow-svg");
    this._arrowSvg = svg;
    this.container.appendChild(svg);

    /* pointer events: drag + click */
    this.container.addEventListener("pointerdown", (e) => this._onPointerDown(e));
    this.container.addEventListener("pointermove", (e) => this._onPointerMove(e));
    this.container.addEventListener("pointerup", (e) => this._onPointerUp(e));
    this.container.addEventListener("pointercancel", () => this._endDrag());
    this.container.addEventListener("contextmenu", (e) => this._onContextMenu(e));

    window.addEventListener("resize", () => this._resyncPieceSizes());
    if (typeof ResizeObserver !== "undefined") {
      this._observer = new ResizeObserver(() => this._resyncPieceSizes());
      this._observer.observe(this.container);
    }
  }

  _applySquareColors() {
    const fileNames = "abcdefgh";
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        /* r = 0 is the TOP row visually (rank 8 from white's view). */
        const idx = this.orientation === "w" ? r * 8 + c : (7 - r) * 8 + (7 - c);
        const sq = this.squares[idx];
        const light = (r + c) % 2 === 0;
        sq.classList.toggle("light", light);
        sq.classList.toggle("dark", !light);
        sq.dataset.rank = r;
        sq.dataset.file = c;
        sq.dataset.square = this._indexToSquare(r, c);
        this._setCoords(sq, r, c, fileNames);
      }
    }
  }

  _setCoords(sq, rank, file, fileNames) {
    sq.querySelectorAll(".coord").forEach((n) => n.remove());
    if (!this.showCoords) return;
    const rankNum = this.orientation === "w" ? 8 - rank : rank + 1;
    const fileChar = this.orientation === "w" ? fileNames[file] : fileNames[7 - file];
    const coordFile = document.createElement("span");
    coordFile.className = "coord coord-file";
    coordFile.textContent = fileChar;
    const coordRank = document.createElement("span");
    coordRank.className = "coord coord-rank";
    coordRank.textContent = rankNum;
    sq.appendChild(coordFile);
    sq.appendChild(coordRank);
  }

  setCoordsVisible(visible) {
    this.showCoords = !!visible;
    this._applySquareColors();
  }

  /* ---- coordinate helpers ----
     Conventions:
       - `r` (0..7) is the visual grid row, 0 = top, 7 = bottom.
       - `c` (0..7) is the visual grid column, 0 = left, 7 = right.
       - A chess square is "abcdefgh"[file] + (1..8 rank).
  */

  _indexToSquare(r, c) {
    if (this.orientation === "w") {
      return "abcdefgh"[c] + (8 - r);
    }
    return "abcdefgh"[7 - c] + (r + 1);
  }

  _visibleIndex(sqName) {
    const file = sqName.charCodeAt(0) - 97;
    const rank = parseInt(sqName[1], 10) - 1;
    if (this.orientation === "w") {
      const r = 7 - rank; // visual row of a chess rank
      return r * 8 + file;
    }
    const r = rank;
    return r * 8 + (7 - file);
  }

  _squareFromEvent(e) {
    const rect = this.container.getBoundingClientRect();
    const size = rect.width;
    const x = (e.clientX - rect.left) / size;
    const y = (e.clientY - rect.top) / size;
    if (x < 0 || x > 1 || y < 0 || y > 1) return null;
    const c = Math.floor(x * 8);
    const r = Math.floor(y * 8);
    return this._indexToSquare(r, c);
  }

  /* ---- pixel geometry ---- */

  _size() {
    return this.container.clientWidth;
  }

  _sqSize() {
    return this._size() / 8;
  }

  /* Top-left corner of a square in board pixels. */
  _posFor(sq) {
    const size = this._size();
    const sqS = size / 8;
    const file = sq.charCodeAt(0) - 97;
    const rank = parseInt(sq[1], 10) - 1;
    const c = this.orientation === "w" ? file : 7 - file;
    const r = this.orientation === "w" ? 7 - rank : rank;
    return { x: c * sqS, y: r * sqS };
  }

  /* Centre of a square in board pixels. */
  _centerFor(sq) {
    const { x, y } = this._posFor(sq);
    const sqS = this._sqSize();
    return { x: x + sqS / 2, y: y + sqS / 2 };
  }

  _applyPiecePos(el, sq) {
    const size = this._size();
    const sqS = size / 8;
    const { x, y } = this._posFor(sq);
    const off = (sqS - size * 0.88) / 2;
    el.style.transform = `translate3d(${x + off}px, ${y + off}px, 0)`;
  }

  _placeGhost(ghost, px, py) {
    const size = this._size();
    const w = size * 0.88;
    const gx = Math.max(0, Math.min(size - w, px - w / 2));
    const gy = Math.max(0, Math.min(size - w, py - w / 2));
    ghost.style.transform = `translate3d(${gx}px, ${gy}px, 0)`;
  }

  /* ---- state setters ---- */

  setPosition(fen) {
    const rows = fen.split(" ")[0].split("/");
    const newPieces = {};
    for (let r = 0; r < 8; r++) {
      let c = 0;
      for (const ch of rows[r]) {
        if (/\d/.test(ch)) {
          c += parseInt(ch, 10);
        } else {
          const sq = "abcdefgh"[c] + (8 - r);
          newPieces[sq] = ch;
          c++;
        }
      }
    }

    this._clearMarkers();
    this._clearHoverPreview();

    /* remove pieces not present */
    for (const sq of Object.keys(this.pieces)) {
      if (!(sq in newPieces)) {
        this.pieces[sq].remove();
        delete this.pieces[sq];
      }
    }
    /* add / update pieces */
    for (const [sq, ch] of Object.entries(newPieces)) {
      if (this.pieces[sq]) {
        if (this.pieces[sq].dataset.piece === ch) continue;
        this.pieces[sq].remove();
        delete this.pieces[sq];
      }
      this._addPiece(sq, ch);
    }
    this._resyncPieceSizes();
  }

  _addPiece(sq, ch) {
    const el = document.createElement("div");
    el.className = "piece";
    el.dataset.piece = ch;
    const size = this._size();
    el.style.width = `${size * 0.88}px`;
    el.style.height = `${size * 0.88}px`;
    this._applyPiecePos(el, sq);
    el.innerHTML = PIECE_SVGS[ch === ch.toLowerCase() ? `${ch}_b` : `${ch.toLowerCase()}_w`];
    this.container.appendChild(el);
    this.pieces[sq] = el;
  }

  _resyncPieceSizes() {
    const size = this._size();
    for (const [sq, el] of Object.entries(this.pieces)) {
      if (!el) continue;
      el.style.width = `${size * 0.88}px`;
      el.style.height = `${size * 0.88}px`;
      this._applyPiecePos(el, sq);
    }
    if (this._drag?.ghost) this._placeGhost(this._drag.ghost, this._drag.x, this._drag.y);
    this._renderArrow();
  }

  setOrientation(o) {
    if (o === this.orientation) return;
    this.orientation = o;
    this.clearHighlights();
    this._applySquareColors();
    for (const sq of Object.keys(this.pieces)) {
      const el = this.pieces[sq];
      if (el) this._applyPiecePos(el, sq);
    }
    this._resyncPieceSizes();
  }

  /* ---- highlighting ---- */

  setSelected(sq) {
    if (this.selected === sq) return;
    this._clearSquareClass(this.selected, "selected");
    this.selected = sq;
    this._clearHoverPreview();
    if (sq) this._squareEl(sq).classList.add("selected");
  }

  setLastMove(from, to) {
    this._clearSquareClass(this.lastMove?.from, "last-move");
    this._clearSquareClass(this.lastMove?.to, "last-move");
    this.lastMove = { from, to };
    if (from) this._squareEl(from).classList.add("last-move");
    if (to) this._squareEl(to).classList.add("last-move");
  }

  setCheck(sq) {
    if (this.checkSquare === sq) return;
    this._clearSquareClass(this.checkSquare, "check");
    this.checkSquare = sq;
    if (sq) this._squareEl(sq).classList.add("check");
  }

  setLegalMoves(moves) {
    this._clearMarkers();
    this._clearHoverPreview();
    this.legalMoves = moves || [];
    for (const m of this.legalMoves) {
      const el = this._squareEl(m.to);
      const marker = document.createElement("div");
      marker.className = `sq-marker ${m.capture ? "capture" : "move"}`;
      el.appendChild(marker);
    }
  }

  setHints(squares) {
    for (const sq of this.hintSquares) this._clearSquareClass(sq, "hint-target");
    this.hintSquares = squares || [];
    for (const sq of this.hintSquares) this._squareEl(sq).classList.add("hint-target");
  }

  /* ---- premove ---- */

  setPremove(from, to) {
    this._clearSquareClass(this.premove?.from, "premove");
    this._clearSquareClass(this.premove?.to, "premove");
    this.premove = { from, to };
    if (from) this._squareEl(from).classList.add("premove");
    if (to) this._squareEl(to).classList.add("premove");
  }

  clearPremove() {
    if (!this.premove) return;
    this._clearSquareClass(this.premove.from, "premove");
    this._clearSquareClass(this.premove.to, "premove");
    this.premove = null;
  }

  /* ---- move arrow ---- */

  setMoveArrow(from, to) {
    this._arrowMove = from && to ? { from, to } : null;
    this._renderArrow();
  }

  _renderArrow() {
    const svg = this._arrowSvg;
    if (!svg) return;
    svg.innerHTML = "";
    if (!this._arrowMove?.from || !this._arrowMove?.to) return;
    const { from, to } = this._arrowMove;
    const sqS = this._sqSize();
    const a = this._centerFor(from);
    const b = this._centerFor(to);
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len = Math.hypot(dx, dy);
    if (len < 1) return;
    const ux = dx / len;
    const uy = dy / len;

    /* shorten the line so the arrowhead sits on the destination square */
    const headLen = sqS * 0.3;
    const headW = sqS * 0.16;
    const tipX = b.x - ux * (sqS * 0.12);
    const tipY = b.y - uy * (sqS * 0.12);
    const baseX = tipX - ux * headLen;
    const baseY = tipY - uy * headLen;
    const px = -uy;
    const py = ux;

    const line = document.createElementNS("http://www.w3.org/2000/svg", "path");
    line.classList.add("move-arrow-line");
    line.setAttribute("d", `M ${a.x} ${a.y} L ${baseX} ${baseY}`);
    line.setAttribute("stroke-width", Math.max(2, sqS * 0.09));
    svg.appendChild(line);

    const head = document.createElementNS("http://www.w3.org/2000/svg", "path");
    head.classList.add("move-arrow-head");
    head.setAttribute(
      "d",
      `M ${tipX} ${tipY} L ${baseX + px * headW} ${baseY + py * headW} ` +
        `L ${baseX - px * headW} ${baseY - py * headW} Z`
    );
    svg.appendChild(head);
  }

  /* ---- marks (analysis) ---- */

  _onContextMenu(e) {
    e.preventDefault();
    if (this._drag) return;
    const sq = this._squareFromEvent(e);
    if (!sq) return;
    const el = this._squareEl(sq);
    if (this.marks.has(sq)) {
      this.marks.delete(sq);
      el.classList.remove("mark");
    } else {
      this.marks.add(sq);
      el.classList.add("mark");
    }
  }

  clearMarks() {
    for (const sq of this.marks) {
      this._squareEl(sq)?.classList.remove("mark");
    }
    this.marks.clear();
  }

  clearHighlights() {
    this.setSelected(null);
    this.setLegalMoves([]);
    this.setHints([]);
    this.setCheck(null);
    this.clearPremove();
    this._clearHoverPreview();
  }

  _clearSquareClass(sq, cls) {
    if (!sq) return;
    const el = this._squareEl(sq);
    if (el) el.classList.remove(cls);
  }

  _squareEl(sq) {
    return this.squares[this._visibleIndex(sq)];
  }

  _clearMarkers() {
    this.container.querySelectorAll(".sq-marker").forEach((n) => n.remove());
  }

  /* ---- hover move preview ---- */

  _updateHoverPreview(sq) {
    if (sq === this._hoverSquare) return;
    this._clearHoverPreview();
    this._hoverSquare = sq;
    if (!sq || this.selected || !this.allowInput || !this.onQueryMoves) return;
    const ch = this.pieces[sq]?.dataset.piece;
    if (!ch) return;
    const moves = this.onQueryMoves(sq) || [];
    if (!moves.length) return;
    for (const m of moves) {
      const el = this._squareEl(m.to);
      const marker = document.createElement("div");
      marker.className = `sq-marker preview ${m.capture ? "capture" : "move"}`;
      el.appendChild(marker);
      this._hoverMarkers.push(marker);
    }
  }

  _clearHoverPreview() {
    this._hoverMarkers.forEach((n) => n.remove());
    this._hoverMarkers = [];
    this._hoverSquare = null;
  }

  /* ---- interactions ---- */

  _onPointerDown(e) {
    if (e.button !== 0) return;
    if (!this.allowInput) return;
    const sq = this._squareFromEvent(e);
    if (!sq) return;
    const ch = this.pieces[sq]?.dataset.piece;
    if (!ch) {
      this._handleEmptyClick(sq);
      return;
    }
    e.preventDefault();
    this._clearHoverPreview();
    const rect = this.container.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const source = this.pieces[sq];
    const ghost = source.cloneNode(true);
    ghost.className = "piece ghost";
    const size = this._size();
    ghost.style.width = `${size * 0.88}px`;
    ghost.style.height = `${size * 0.88}px`;
    this._placeGhost(ghost, px, py);
    this.container.appendChild(ghost);
    source.classList.add("dragging-source");
    this._drag = { sq, ghost, source, offsetX: px, offsetY: py, x: px, y: py, moved: false };
    this.container.setPointerCapture(e.pointerId);
  }

  _onPointerMove(e) {
    if (this._drag) {
      const rect = this.container.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      this._placeGhost(this._drag.ghost, px, py);
      this._drag.moved = true;
      this._drag.x = px;
      this._drag.y = py;

      /* hover highlight */
      this.container.querySelectorAll(".square.hover").forEach((n) => n.classList.remove("hover"));
      const target = this._squareFromEvent(e);
      if (target) this._squareEl(target).classList.add("hover");
    } else {
      this._updateHoverPreview(this._squareFromEvent(e));
    }
  }

  _onPointerUp(e) {
    if (!this._drag) return;
    const drag = this._drag;
    this._endDrag();

    const target = this._squareFromEvent(e);
    if (!target) return;

    if (this._dragMoved(drag)) {
      this._tryMove(drag.sq, target);
    } else {
      this._handleClick(drag.sq);
    }
  }

  _dragMoved(drag) {
    const sqSize = this._sqSize();
    return (
      Math.abs(drag.offsetX - (drag.x ?? drag.offsetX)) > sqSize * 0.35 ||
      Math.abs(drag.offsetY - (drag.y ?? drag.offsetY)) > sqSize * 0.35
    );
  }

  _endDrag() {
    if (this._drag) {
      this._drag.ghost.remove();
      this._drag.source.classList.remove("dragging-source");
      this._drag = null;
    }
    this.container.querySelectorAll(".square.hover").forEach((n) => n.classList.remove("hover"));
    this._clearHoverPreview();
  }

  _handleEmptyClick(sq) {
    if (this.selected) {
      this._tryMove(this.selected, sq);
    } else {
      if (this.onSelect) this.onSelect(null);
    }
  }

  _handleClick(sq) {
    const ch = this.pieces[sq]?.dataset.piece;
    if (!ch) {
      this._handleEmptyClick(sq);
      return;
    }
    if (this.selected === sq) {
      if (this.onSelect) this.onSelect(null);
      return;
    }
    if (this.selected) {
      this._tryMove(this.selected, sq);
    } else {
      if (this.onSelect) this.onSelect(sq);
    }
  }

  _tryMove(from, to) {
    if (!this.onMoveRequest) return;
    const computed = this.onQueryMoves ? (this.onQueryMoves(from) || []) : this.legalMoves;
    this.legalMoves = computed;
    const legal = computed.find((m) => m.from === from && m.to === to);
    if (!legal) {
      /* clicked another of our own pieces — switch selection */
      const ch = this.pieces[to]?.dataset.piece;
      if (ch) {
        if (this.onSelect) this.onSelect(to);
      } else if (this.onSelect) {
        this.onSelect(null);
      }
      return;
    }
    if (this.onMoveRequest) {
      this.onMoveRequest({ from, to, promotion: null });
    }
  }

  /* ---- animation ---- */

  animateMove(from, to, isCapture, cb) {
    const el = this.pieces[from];
    if (!el) {
      if (cb) cb();
      return;
    }
    this.container.appendChild(el); // bring to front
    el.style.transition = "transform 0.2s cubic-bezier(0.2,0.8,0.25,1)";
    this._applyPiecePos(el, to);
    el.dataset.anim = ++this._animCounter;

    const finish = () => {
      el.style.transition = "";
      this.pieces[from] = null;
      delete this.pieces[from];
      if (this.pieces[to]) {
        this.pieces[to].remove();
        delete this.pieces[to];
      }
      this.pieces[to] = el;
      if (cb) cb();
    };

    const target = this.pieces[to];
    let capturedEl = null;
    if (target) {
      capturedEl = target;
      target.classList.add("capturing");
    }

    setTimeout(() => {
      if (capturedEl) capturedEl.remove();
      finish();
    }, isCapture ? 220 : 200);
  }
}

window.Board = Board;
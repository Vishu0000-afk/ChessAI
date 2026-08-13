/* ==========================================================================
   App — main application controller.
   Wires the Board, Game, ChessClock, and AIController together. Handles
   game modes (HvH, HvAI, AIvAI), promotion UI, premoves, move history
   panel, player bars, clocks, evaluation bar, themes, sounds, analysis
   marks, and game-over messaging.
   ========================================================================== */

class App {
  constructor() {
    this.board = new Board(document.getElementById("board"));
    this.game = new Game();
    this.clock = new ChessClock();
    this.ai = new AIController();

    this.mode = "hvh"; // hvh | hvai | aiai
    this.humanSide = "w";
    this.aiDepth = 3;
    this.animating = false;
    this.pendingPromotion = null;

    /* settings (persisted) */
    this.autoQueen = localStorage.getItem("chessai.autoQueen") === "1";
    this.theme = localStorage.getItem("chessai.theme") || "crimson";
    this.board.showCoords = localStorage.getItem("chessai.coords") !== "0";

    /* premove state */
    this.premove = null;
    this._premoveFrom = null;
    this._evalToken = 0;

    this._bindEvents();
    this._bindBoard();
    this._setupSettings();
    this._init();
  }

  /* ---- setup ---- */

  _bindEvents() {
    this.game.onChange = () => this._refresh();

    document.getElementById("btn-new-game").addEventListener("click", () => this.newGame());
    document.getElementById("btn-flip").addEventListener("click", () => this.flipBoard());
    document.getElementById("btn-hint").addEventListener("click", () => this.requestHint());
    document.getElementById("btn-pause").addEventListener("click", () => this.togglePause());

    document.getElementById("btn-undo").addEventListener("click", () => this.undoMove());
    document.getElementById("btn-redo").addEventListener("click", () => this.redoMove());
    document.getElementById("btn-restart").addEventListener("click", () => this.newGame());
    document.getElementById("btn-resign").addEventListener("click", () => this.resign());
    document.getElementById("btn-clear-marks").addEventListener("click", () => this.board.clearMarks());

    document.getElementById("nav-start").addEventListener("click", () => this.navigateTo(0));
    document.getElementById("nav-prev").addEventListener("click", () => this.navigateTo(this.game.pointer - 1));
    document.getElementById("nav-next").addEventListener("click", () => this.navigateTo(this.game.pointer + 1));
    document.getElementById("nav-end").addEventListener("click", () => this.navigateTo(this.game.moves.length));

    document.querySelectorAll(".tc-btn").forEach((b) =>
      b.addEventListener("click", () => this._setTimeControl(b.dataset.time))
    );

    document.querySelectorAll(".mode-btn").forEach((b) =>
      b.addEventListener("click", () => this._setMode(b.dataset.mode))
    );

    document.getElementById("move-list").addEventListener("click", (e) => {
      const san = e.target.closest(".move-san");
      if (!san) return;
      this.navigateTo(parseInt(san.dataset.index, 10));
    });
  }

  _bindBoard() {
    this.board.onMoveRequest = ({ from, to, promotion }) => this._onMoveRequest(from, to, promotion);
    this.board.onSelect = (sq) => this._onSelect(sq);
    this.board.onQueryMoves = (sq) => {
      if (this.mode === "hvai" && this.game.turn() !== this.humanSide) {
        const ch = this.game.chess.get(sq);
        if (!ch || ch.color !== this.humanSide) return [];
        return this.game.premoveMovesFor(sq);
      }
      return this.game.legalMovesFor(sq);
    };
  }

  _setupSettings() {
    const themes = ["crimson", "classic", "emerald", "obsidian"];
    const box = document.getElementById("theme-select");
    box.innerHTML = "";
    themes.forEach((t) => {
      const btn = document.createElement("button");
      btn.className = "theme-btn";
      btn.dataset.theme = t;
      btn.innerHTML = `<span class="theme-swatch"></span>${t[0].toUpperCase() + t.slice(1)}`;
      btn.addEventListener("click", () => this._setTheme(t));
      box.appendChild(btn);
    });
    this._setTheme(this.theme, true);

    this._bindSwitch(
      "opt-autoqueen",
      () => this.autoQueen,
      (v) => {
        this.autoQueen = v;
        localStorage.setItem("chessai.autoQueen", v ? "1" : "0");
      }
    );
    this._bindSwitch(
      "opt-coords",
      () => this.board.showCoords,
      (v) => {
        this.board.setCoordsVisible(v);
        localStorage.setItem("chessai.coords", v ? "1" : "0");
      }
    );
    this._bindSwitch("opt-sound", () => Sound.enabled, (v) => (Sound.enabled = v));

    /* apply the persisted coordinate visibility to the already-built board */
    this.board.setCoordsVisible(this.board.showCoords);
  }

  _bindSwitch(id, get, set) {
    const el = document.getElementById(id);
    if (!el) return;
    const sync = () => el.classList.toggle("on", !!get());
    sync();
    el.addEventListener("click", () => {
      set(!get());
      sync();
    });
    el.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        set(!get());
        sync();
      }
    });
  }

  _setTheme(name, init) {
    this.theme = name;
    if (!init) localStorage.setItem("chessai.theme", name);
    document.documentElement.dataset.theme = name;
    document.querySelectorAll(".theme-btn").forEach((b) => b.classList.toggle("active", b.dataset.theme === name));
  }

  async _init() {
    await this.ai.init();
    this.newGame();
  }

  /* ---- game lifecycle ---- */

  newGame() {
    this.game.reset();
    this.clock.reset();
    this.clock.setControl(this.clock.preset, undefined, undefined);
    this.board.setPosition(this.game.fen());
    this.board.clearHighlights();
    this.premove = null;
    this._premoveFrom = null;
    this.board.allowInput = true;
    this._resetUI();
    this.clock.start("w");
    this._refresh();
    if (this._aiToMove()) this._scheduleAI();
  }

  _resetUI() {
    this.game = new Game();
    this.game.onChange = () => this._refresh();
    document.getElementById("game-message").classList.remove("show", "win");
    this._setClockDisplay();
    this._updatePlayerBars();
  }

  _aiToMove() {
    if (this.game.isGameOver()) return false;
    const turn = this.game.turn();
    if (this.mode === "aiai") return true;
    if (this.mode === "hvai") return turn !== this.humanSide;
    return false;
  }

  _scheduleAI() {
    if (this.animating) {
      setTimeout(() => {
        if (this._aiToMove() && !this.animating) this._scheduleAI();
      }, 250);
      return;
    }
    setTimeout(() => this._makeAIMove(), 350);
  }

  async _makeAIMove() {
    if (!this._aiToMove()) return;
    const side = this.game.turn();
    const move = await this.ai.getMove(this.game.chess, side, this.aiDepth);
    if (!move) return;
    if (!this._aiToMove()) return;
    this._executeMove(move);
  }

  /* ---- input handling ---- */

  _onSelect(sq) {
    if (this.animating) return;
    if (this.game.isGameOver()) return;
    if (this.mode === "aiai") return;

    const aiTurnInHvai = this.mode === "hvai" && this.game.turn() !== this.humanSide;
    if (aiTurnInHvai) {
      this._onPremoveSelect(sq);
      return;
    }

    this.board.setSelected(sq);
    if (!sq) {
      this.board.setLegalMoves([]);
      return;
    }
    const moves = this.game.legalMovesFor(sq);
    if (moves.length === 0) {
      this.board.setSelected(null);
      this.board.setLegalMoves([]);
      return;
    }
    this.board.setLegalMoves(moves);
  }

  _onPremoveSelect(sq) {
    if (!sq) {
      this._clearPremove();
      return;
    }
    const ch = this.game.chess.get(sq);
    if (!ch || ch.color !== this.humanSide) {
      this._clearPremove();
      return;
    }
    const moves = this.game.premoveMovesFor(sq);
    if (!moves.length) {
      this._clearPremove();
      return;
    }
    if (this.premove && this.premove.from === sq) {
      this._clearPremove();
      return;
    }
    this._clearPremove();
    this._premoveFrom = sq;
    this.board.setSelected(sq);
    this.board.setLegalMoves(moves);
  }

  _clearPremove() {
    this.premove = null;
    this._premoveFrom = null;
    this.board.clearPremove();
    this.board.setSelected(null);
    this.board.setLegalMoves([]);
  }

  _onMoveRequest(from, to, promotion) {
    if (this.animating) return;
    if (this.game.isGameOver()) return;

    const aiTurnInHvai = this.mode === "hvai" && this.game.turn() !== this.humanSide;
    if (aiTurnInHvai) {
      this._queuePremove(from, to, promotion);
      return;
    }
    if (this.mode === "aiai") return;

    /* clear any stray premove before committing a live move */
    this.board.clearPremove();
    this.premove = null;
    this._premoveFrom = null;

    /* promotion handling: if piece moving to last rank is a pawn, ask user */
    const needPromotion =
      !promotion &&
      this.game.chess.get(from)?.type === "p" &&
      (to[1] === "8" || to[1] === "1");

    if (needPromotion && this.autoQueen) {
      this._executeMove({ from, to, promotion: "q" });
      return;
    }
    if (needPromotion) {
      this._requestPromotion(from, to);
      return;
    }

    this._executeMove({ from, to, promotion });
  }

  _queuePremove(from, to, promotion) {
    const ch = this.game.chess.get(from);
    if (!ch || ch.color !== this.humanSide) return;
    const legal = this.game.premoveMovesFor(from);
    const match = legal.find((m) => m.from === from && m.to === to);
    if (!match) {
      this._toast("Illegal premove");
      this._clearPremove();
      return;
    }
    let promo = promotion;
    if (!promo && ch.type === "p" && (to[1] === "8" || to[1] === "1")) promo = "q";
    this.premove = { from, to, promotion: promo };
    this._premoveFrom = null;
    this.board.setSelected(null);
    this.board.setLegalMoves([]);
    this.board.setPremove(from, to);
  }

  _executeMove({ from, to, promotion }) {
    if (this.animating) return;
    const move = this.game.makeMove({ from, to, promotion });
    if (!move) {
      this._toast("Illegal move");
      this.board.clearHighlights();
      return;
    }
    this._playMove(move);
  }

  _playMove(move) {
    this.animating = true;
    this.board.allowInput = false;
    this.board.setSelected(null);
    this.board.setLegalMoves([]);
    this.board.setHints([]);

    const isCapture = !!move.captured;
    Sound.play(isCapture ? "capture" : "move");

    /* clock switch happens at animation start; simple + consistent */
    this.clock.switchSide();

    this.board.animateMove(move.from, move.to, isCapture, () => {
      this.animating = false;
      this.board.allowInput = true;
      this._afterMove();
    });
  }

  _afterMove() {
    this.game.onChange(); // triggers _refresh which redraws position
    const s = this.game.state();
    this._handleGameEnd(s);

    if (!s.gameOver && this.premove && this.game.turn() === this.humanSide) {
      this._executePremove();
      return;
    }
    if (!s.gameOver && this._aiToMove()) this._scheduleAI();
  }

  _executePremove() {
    const p = this.premove;
    this._clearPremove();
    const move = this.game.makeMove(p);
    if (!move) {
      this._toast("Premove was illegal");
      this.game.onChange();
      if (this._aiToMove()) this._scheduleAI();
      return;
    }
    this._playMove(move);
  }

  _handleGameEnd(s) {
    const msg = document.getElementById("game-message");
    if (s.checkmate) {
      const winner = s.turn === "w" ? "Black" : "White";
      msg.textContent = `Checkmate — ${winner} wins`;
      msg.classList.add("show", "win");
      this.clock.stop();
      this.board.allowInput = false;
      Sound.play("gameover");
    } else if (s.stalemate) {
      msg.textContent = "Stalemate — Draw";
      msg.classList.add("show");
      this.clock.stop();
      this.board.allowInput = false;
      Sound.play("gameover");
    } else if (s.draw) {
      msg.textContent = "Draw";
      msg.classList.add("show");
      this.clock.stop();
      this.board.allowInput = false;
      Sound.play("gameover");
    } else if (s.check) {
      msg.textContent = "Check!";
      Sound.play("check");
    } else {
      msg.classList.remove("show", "win");
    }
  }

  /* ---- promotion ---- */

  _requestPromotion(from, to) {
    this.pendingPromotion = { from, to };
    const color = this.game.turn();
    const choices = ["q", "r", "b", "n"].map((p) => {
      const ch = color === "w" ? p.toUpperCase() : p;
      const el = document.createElement("button");
      el.className = "promo-choice";
      el.innerHTML = PIECE_SVGS[color === "w" ? `${p}_w` : `${p}_b`];
      el.addEventListener("click", () => {
        document.getElementById("promotion-overlay").classList.add("hidden");
        this.pendingPromotion = null;
        this._executeMove({ from, to, promotion: p });
      });
      return el;
    });
    const box = document.getElementById("promotion-choices");
    box.innerHTML = "";
    choices.forEach((c) => box.appendChild(c));
    document.getElementById("promotion-overlay").classList.remove("hidden");
  }

  /* ---- navigation ---- */

  navigateTo(count) {
    if (this.animating) return;
    if (this.mode === "aiai") return;
    this._clearPremove();
    this.game.goTo(count);
    this.board.setPosition(this.game.fen());
    this.board.clearHighlights();
    this._syncClockToNavigation();
    this._refresh();
  }

  undoMove() {
    if (this.mode === "aiai") return;
    if (this.animating) return;
    if (!this.game.canUndo()) return;
    this.navigateTo(this.game.pointer - 1);
  }

  redoMove() {
    if (this.mode === "aiai") return;
    if (this.animating) return;
    if (!this.game.canRedo()) return;
    this.navigateTo(this.game.pointer + 1);
  }

  flipBoard() {
    this.board.setOrientation(this.board.orientation === "w" ? "b" : "w");
  }

  resign() {
    if (this.game.isGameOver()) return;
    this._clearPremove();
    const loser = this.game.turn() === "w" ? "White" : "Black";
    const msg = document.getElementById("game-message");
    msg.textContent = `${loser} resigns — ${loser === "White" ? "Black" : "White"} wins`;
    msg.classList.add("show", "win");
    this.clock.stop();
    this.game.resigned = true;
    this.board.allowInput = false;
    Sound.play("gameover");
    this._refresh();
  }

  togglePause() {
    if (this.clock.running) {
      this.clock.pause();
      this._toast("Paused");
    } else {
      this.clock.start(this.clock.side || this.game.turn());
      this._toast("Resumed");
    }
  }

  async requestHint() {
    const s = this.game.state();
    if (s.gameOver) return;
    const move = await this.ai.getMove(this.game.chess, s.turn, this.aiDepth);
    if (!move) return;
    this.board.setHints([move.from, move.to]);
    this._toast(`Hint: ${move.from}-${move.to}`);
  }

  /* ---- mode & time control ---- */

  _setMode(mode) {
    if (this.animating) return;
    document.querySelectorAll(".mode-btn").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
    this.mode = mode;
    if (mode === "hvai") {
      this.humanSide = "w";
    }
    this.newGame();
  }

  _setTimeControl(preset) {
    document.querySelectorAll(".tc-btn").forEach((b) => b.classList.toggle("active", b.dataset.time === preset));
    if (preset === "custom") {
      /* simple custom: 5 min + 5s — could prompt, keep minimal */
      this.clock.setControl("custom", 5, 5);
      document.getElementById("tc-value").textContent = "5:00 +5s";
    } else {
      this.clock.setControl(preset);
      document.getElementById("tc-value").textContent = this.clock.format("w");
    }
    this._refresh();
  }

  /* ---- rendering ---- */

  _refresh() {
    const s = this.game.state();
    this.board.setPosition(s.fen);
    this.board.setCheck(s.check ? this.game.findKing(s.turn) : null);

    const last = s.moves[s.pointer - 1];
    if (last) {
      this.board.setLastMove(last.from, last.to);
      this.board.setMoveArrow(last.from, last.to);
    } else {
      this.board.setLastMove(null, null);
      this.board.setMoveArrow(null, null);
    }

    this._renderMoveList();
    this._renderCaptured();
    this._updatePlayerBars();
    this._setClockDisplay();
    this._updateStatus(s);
    this._updateEval();

    /* trigger AI moves when browsing back to a live position in hvai/aiai */
    if (!this.animating && this._aiToMove()) this._scheduleAI();
  }

  _updateStatus(s) {
    const el = document.getElementById("status");
    el.classList.toggle("check", s.check);
    if (s.checkmate) el.textContent = "Checkmate";
    else if (s.stalemate) el.textContent = "Stalemate";
    else if (s.draw) el.textContent = "Draw";
    else if (s.check) el.textContent = `${s.turn === "w" ? "White" : "Black"} is in check`;
    else el.textContent = `${s.turn === "w" ? "White" : "Black"} to move`;
  }

  _renderMoveList() {
    const list = document.getElementById("move-list");
    list.innerHTML = "";
    const s = this.game.state();
    for (let i = 0; i < s.moves.length; i++) {
      const num = document.createElement("li");
      num.className = "move-num";
      num.textContent = `${Math.floor(i / 2) + 1}.`;

      const san = document.createElement("li");
      san.className = `move-san${i % 2 === 0 ? " white" : " black"}`;
      if (s.moves[i].captured) san.classList.add("captured-san");
      san.textContent = s.moves[i].san;
      san.dataset.index = i;
      if (i === s.pointer - 1) san.classList.add("current");

      list.appendChild(num);
      list.appendChild(san);
      if (i % 2 === 1) list.appendChild(document.createElement("li"));
    }
    list.scrollTop = list.scrollHeight;
  }

  _renderCaptured() {
    const pMap = { p: "♙", n: "♘", b: "♗", r: "♖", q: "♕" };
    const white = document.getElementById("white-captured");
    const black = document.getElementById("black-captured");

    const wc = this.game.capturedBy("w"); // pieces White captured (black pieces)
    const bc = this.game.capturedBy("b");

    white.innerHTML = wc
      .slice()
      .sort((a, b) => "pnbrq".indexOf(b) - "pnbrq".indexOf(a))
      .map((t) => `<span class="cap-black">${pMap[t]}</span>`)
      .join("");
    black.innerHTML = bc
      .slice()
      .sort((a, b) => "pnbrq".indexOf(b) - "pnbrq".indexOf(a))
      .map((t) => `<span class="cap-white">${pMap[t]}</span>`)
      .join("");
  }

  _updatePlayerBars() {
    const turn = this.game.state().turn;
    document.getElementById("clock-white").closest(".player-bar").classList.toggle("active", turn === "w");
    document.getElementById("clock-black").closest(".player-bar").classList.toggle("active", turn === "b");
  }

  _setClockDisplay() {
    const low = Math.max(this.clock.whiteMs, this.clock.blackMs) / 1000 < 30;
    const cw = document.getElementById("clock-white");
    const cb = document.getElementById("clock-black");
    cw.textContent = this.clock.format("w");
    cb.textContent = this.clock.format("b");
    cw.classList.toggle("low", low);
    cb.classList.toggle("low", low);
  }

  _syncClockToNavigation() {
    this.clock.seek(this.game.pointer);
    this._setClockDisplay();
  }

  /* ---- evaluation bar ---- */

  _updateEval() {
    const token = ++this._evalToken;
    this.ai
      .getEval(this.game)
      .then((cp) => {
        if (token === this._evalToken) this._renderEval(cp);
      })
      .catch(() => {});
  }

  _renderEval(cp) {
    const fill = document.getElementById("eval-fill-white");
    const label = document.getElementById("eval-label");
    if (!fill || !label) return;
    const pct = 50 + 50 * Math.tanh((cp || 0) / 300);
    fill.style.height = `${Math.max(0, Math.min(100, pct))}%`;
    if (Math.abs(cp) > 2000) {
      label.textContent = cp > 0 ? "M" : "-M";
    } else {
      label.textContent = `${cp >= 0 ? "+" : ""}${(cp / 100).toFixed(1)}`;
    }
  }

  _toast(msg) {
    const t = document.getElementById("toast");
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(this._toastTimer);
    this._toastTimer = setTimeout(() => t.classList.remove("show"), 1800);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  window.app = new App();
});
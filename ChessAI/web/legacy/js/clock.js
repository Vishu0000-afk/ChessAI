/* ==========================================================================
   ChessClock — functional chess timers with time controls.
   ========================================================================== */

const TIME_CONTROLS = {
  bullet: { label: "Bullet", main: 60, increment: 0 },
  blitz: { label: "Blitz", main: 180, increment: 0 },
  rapid: { label: "Rapid", main: 600, increment: 0 },
  custom: { label: "Custom", main: 300, increment: 5 },
};

class ChessClock {
  constructor() {
    this.reset();
    this.preset = "rapid";
    this.mainMs = TIME_CONTROLS.rapid.main * 1000;
    this.incrementMs = TIME_CONTROLS.rapid.increment * 1000;
    this.interval = null;
    this.onExpire = null;
    this.onTick = null;
  }

  reset() {
    this.whiteMs = this.mainMs;
    this.blackMs = this.mainMs;
    this.side = null;
    this.running = false;
    this.expiredSide = null;
    /* clock state after each ply; snapshots[ply] = state with `ply` moves played */
    this.snapshots = [{ whiteMs: this.mainMs, blackMs: this.mainMs }];
  }

  setControl(preset, mainMinutes, incrementSeconds) {
    if (preset === "custom") {
      this.mainMs = (mainMinutes || 5) * 60 * 1000;
      this.incrementMs = (incrementSeconds || 0) * 1000;
    } else {
      const t = TIME_CONTROLS[preset];
      this.mainMs = t.main * 1000;
      this.incrementMs = t.increment * 1000;
    }
    this.preset = preset;
    this.reset();
  }

  start(side) {
    this.stop();
    this.side = side;
    this.running = true;
    this._lastTick = performance.now();
    this.interval = setInterval(() => this._tick(), 200);
  }

  stop() {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }
    this.running = false;
  }

  pause() {
    if (this.running) {
      this._applyElapsed();
      this.stop();
    }
  }

  _applyElapsed() {
    if (!this.side || !this.running) return;
    const now = performance.now();
    const elapsed = now - this._lastTick;
    this._lastTick = now;
    this._decrement(this.side, elapsed);
  }

  _decrement(side, ms) {
    if (side === "w") {
      this.whiteMs = Math.max(0, this.whiteMs - ms);
      if (this.whiteMs === 0) this._expire("w");
    } else {
      this.blackMs = Math.max(0, this.blackMs - ms);
      if (this.blackMs === 0) this._expire("b");
    }
    if (this.onTick) this.onTick();
  }

  _tick() {
    this._applyElapsed();
  }

  _expire(side) {
    this.expiredSide = side;
    this.stop();
    if (this.onExpire) this.onExpire(side);
  }

  /* Called after a move completes: switch clocks and add increment. */
  switchSide() {
    this._applyElapsed();
    if (this.side === "w") {
      this.whiteMs += this.incrementMs;
      this.side = "b";
    } else if (this.side === "b") {
      this.blackMs += this.incrementMs;
      this.side = "w";
    }
    this.snapshots.push({ whiteMs: this.whiteMs, blackMs: this.blackMs });
    if (this.onTick) this.onTick();
  }

  /* Restore clock state to a given ply (for move-history navigation). */
  seek(ply) {
    this.stop();
    const idx = Math.max(0, Math.min(ply, this.snapshots.length - 1));
    const s = this.snapshots[idx] || { whiteMs: this.mainMs, blackMs: this.mainMs };
    this.whiteMs = s.whiteMs;
    this.blackMs = s.blackMs;
    this.side = null;
    this.running = false;
    this.expiredSide = null;
    if (this.onTick) this.onTick();
  }

  ms(side) {
    return side === "w" ? this.whiteMs : this.blackMs;
  }

  format(side) {
    const ms = this.ms(side);
    const totalSec = Math.ceil(ms / 1000);
    const m = Math.floor(totalSec / 60);
    const s = totalSec % 60;
    return `${m}:${s.toString().padStart(2, "0")}`;
  }
}

window.ChessClock = ChessClock;
window.TIME_CONTROLS = TIME_CONTROLS;
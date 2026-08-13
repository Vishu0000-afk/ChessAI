/* ==========================================================================
   Sound — tiny WebAudio synth for UI feedback (no audio assets needed).
   Exposes `window.Sound` (a singleton instance) with:
     Sound.enabled            (bool, persisted in localStorage)
     Sound.toggle() -> bool
     Sound.play(name)         name in move|capture|check|gameover|select
   ========================================================================== */

window.Sound = new (class Sound {
  constructor() {
    this._ctx = null;
    this._enabled = localStorage.getItem("chessai.sound") !== "0";
  }

  get enabled() {
    return this._enabled;
  }

  set enabled(v) {
    this._enabled = !!v;
    localStorage.setItem("chessai.sound", this._enabled ? "1" : "0");
  }

  toggle() {
    this.enabled = !this._enabled;
    if (this._enabled) this.play("select");
    return this._enabled;
  }

  _ctxNow() {
    if (!this._ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return null;
      this._ctx = new AC();
    }
    if (this._ctx.state === "suspended") this._ctx.resume();
    return this._ctx;
  }

  _tone(freq, dur, type, gain, when) {
    const ctx = this._ctxNow();
    if (!ctx) return;
    const t0 = ctx.currentTime + (when || 0);
    const osc = ctx.createOscillator();
    const g = ctx.createGain();
    osc.type = type || "sine";
    osc.frequency.setValueAtTime(freq, t0);
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.exponentialRampToValueAtTime(gain || 0.15, t0 + 0.008);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
    osc.connect(g);
    g.connect(ctx.destination);
    osc.start(t0);
    osc.stop(t0 + dur + 0.02);
  }

  play(name) {
    if (!this._enabled) return;
    switch (name) {
      case "move":
        this._tone(340, 0.07, "triangle", 0.18);
        break;
      case "capture":
        this._tone(190, 0.09, "square", 0.12);
        this._tone(120, 0.11, "sine", 0.2, 0.01);
        break;
      case "check":
        this._tone(880, 0.09, "sine", 0.14);
        this._tone(660, 0.12, "sine", 0.14, 0.09);
        break;
      case "gameover":
        this._tone(523.25, 0.16, "sine", 0.14);
        this._tone(659.25, 0.16, "sine", 0.14, 0.12);
        this._tone(783.99, 0.26, "sine", 0.14, 0.24);
        break;
      case "select":
        this._tone(220, 0.04, "sine", 0.08);
        break;
      default:
        break;
    }
  }
})();
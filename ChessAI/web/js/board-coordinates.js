/* BoardCoordinates — file/rank labels rendered around the board, orientation-aware. */
class BoardCoordinates {
  constructor(container) {
    this._container = container;
    this._orientation = "w";
    this.render();
  }

  setOrientation(orientation) {
    this._orientation = orientation;
    this.render();
  }

  get orientation() {
    return this._orientation;
  }

  render() {
    this._container.innerHTML = "";
    const files = ["a", "b", "c", "d", "e", "f", "g", "h"];
    const ranks = ["1", "2", "3", "4", "5", "6", "7", "8"];

    for (let c = 0; c < 8; c++) {
      const file = files[this._orientation === "w" ? c : 7 - c];
      const rank = ranks[this._orientation === "w" ? 7 - c : c];

      this._add("file-top", file, c);
      this._add("file-bot", file, c);
      this._add("rank-left", rank, c);
      this._add("rank-right", rank, c);
    }
  }

  _add(cls, text, index) {
    const s = document.createElement("span");
    s.className = cls;
    s.textContent = text;
    if (cls.indexOf("file") !== -1) {
      s.style.left = (index * 12.5) + "%";
    } else {
      s.style.top = (index * 12.5) + "%";
    }
    this._container.appendChild(s);
  }
}
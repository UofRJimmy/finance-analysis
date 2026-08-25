from __future__ import annotations

from pathlib import Path


class Watchlist:
    def __init__(self, path: Path):
        self.path = path
        self._mtime: float | None = None
        self._tickers: list[str] = []

    def get(self) -> list[str]:
        mtime = self.path.stat().st_mtime if self.path.exists() else None
        if mtime != self._mtime:
            self._tickers = self._load()
            self._mtime = mtime
        return self._tickers

    def _load(self) -> list[str]:
        if not self.path.exists():
            return []
        tickers: list[str] = []
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip().upper()
            if line:
                tickers.append(line)
        return list(dict.fromkeys(tickers))

    def add(self, ticker: str) -> bool:
        ticker = ticker.strip().upper()
        if not ticker or ticker in self.get():
            return False
        lines = self.path.read_text(encoding="utf-8").splitlines() if self.path.exists() else []
        lines.append(ticker)
        self._write_lines(lines)
        return True

    def remove(self, ticker: str) -> bool:
        ticker = ticker.strip().upper()
        if ticker not in self.get():
            return False
        lines = self.path.read_text(encoding="utf-8").splitlines()
        kept = [line for line in lines if line.split("#", 1)[0].strip().upper() != ticker]
        self._write_lines(kept)
        return True

    def _write_lines(self, lines: list[str]) -> None:
        # Replace atomically so the polling thread never reads a half-written watchlist.
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temp_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        temp_path.replace(self.path)
        self._mtime = None

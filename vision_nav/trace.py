"""Per-run artefacts: every screenshot the model saw plus the decision it made.
These double as debugging aid and as evidence for the observations write-up."""

from __future__ import annotations

import json
import time
from pathlib import Path


class Tracer:
    def __init__(self, root: Path | None):
        self.dir: Path | None = None
        if root is not None:
            self.dir = root / time.strftime("%Y%m%d-%H%M%S")
            self.dir.mkdir(parents=True, exist_ok=True)

    def image(self, name: str, png: bytes) -> str | None:
        if not self.dir:
            return None
        path = self.dir / name
        path.write_bytes(png)
        return str(path)

    def event(self, record: dict) -> None:
        if not self.dir:
            return
        with (self.dir / "steps.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def result(self, data: dict) -> None:
        if self.dir:
            (self.dir / "result.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

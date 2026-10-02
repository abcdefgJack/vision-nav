#!/usr/bin/env python
"""End-to-end scenario runner.

Runs navigate.py on a set of scenarios and scores the result against ground truth
from the GitHub REST API. The API is used ONLY here, to grade the agent -- the agent
itself never sees it.

  python scripts/run_scenarios.py                 # all scenarios
  python scripts/run_scenarios.py openclaw-som    # just one
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "scenarios"

SCENARIOS = [
    # name, navigate.py args, repo to grade against (None = expect a graceful failure)
    ("openclaw-som", ["--repo", "openclaw/openclaw"], "openclaw/openclaw"),
    ("openclaw-coords", ["--repo", "openclaw/openclaw", "--grounding", "coords"], "openclaw/openclaw"),
    ("openclaw-prompt", ["--prompt", "search for openclaw and get the current release and related tags"], "openclaw/openclaw"),
    ("react-features", ["--prompt", "Find the latest React release and list its key features"], "facebook/react"),
    ("vscode-som", ["--repo", "microsoft/vscode"], "microsoft/vscode"),
    ("no-releases", ["--repo", "torvalds/linux", "--max-steps", "12"], None),
]


def gh(path: str):
    req = urllib.request.Request(f"https://api.github.com/{path}", headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def ground_truth(repo: str) -> dict:
    rel = gh(f"repos/{repo}/releases/latest")
    sha = gh(f"repos/{repo}/commits/{rel['tag_name']}")["sha"]
    return {"version": rel["tag_name"], "sha": sha, "author": rel["author"]["login"]}


def grade(out: dict, truth: dict | None) -> dict:
    rel = (out or {}).get("latest_release") or {}
    if truth is None:   # expected: no release found, handled gracefully
        ok = out.get("run", {}).get("status") in ("failed", "partial") and not rel.get("marked_latest")
        return {"pass": ok}
    author = (rel.get("author") or "").lower()
    checks = {
        "version": rel.get("version") == truth["version"],
        "tag": bool(rel.get("tag")) and truth["sha"].startswith(rel["tag"].lower()),
        # The UI shows bots without the "[bot]" suffix the API uses.
        "author": author == truth["author"].lower().removesuffix("[bot]"),
    }
    checks["pass"] = all(checks.values())
    return checks


def main(names: list[str]) -> int:
    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    for name, args, repo in SCENARIOS:
        if names and name not in names:
            continue
        print(f"== {name}", flush=True)
        truth = ground_truth(repo) if repo else None
        out_file = OUT_DIR / f"{name}.json"
        t = time.monotonic()
        proc = subprocess.run([sys.executable, str(ROOT / "navigate.py"), *args, "--out", str(out_file)],
                              cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        elapsed = time.monotonic() - t
        out = json.loads(out_file.read_text(encoding="utf-8")) if out_file.exists() and proc.returncode != 1 else {}
        g = grade(out, truth)
        run = out.get("run", {})
        rows.append({"scenario": name, "exit": proc.returncode, "status": run.get("status"), "steps": run.get("steps"),
                     "model_calls": run.get("model_calls"), "seconds": round(elapsed), "truth": truth, "grade": g,
                     "error": proc.stderr.strip().splitlines()[-1] if proc.returncode == 1 and proc.stderr else None})
        print(json.dumps(rows[-1], indent=2), flush=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    (OUT_DIR / f"results-{stamp}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    passed = sum(r["grade"].get("pass", False) for r in rows)
    print(f"\n{passed}/{len(rows)} scenarios passed")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

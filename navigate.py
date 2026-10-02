#!/usr/bin/env python
"""Vision-driven GitHub navigator.

  python navigate.py --repo openclaw/openclaw
  python navigate.py --prompt "search for openclaw and get the current release and related tags"
  python navigate.py --prompt "Find the latest React release and list its key features" --headed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from vision_nav.agent import Agent
from vision_nav.browser import BrowserSession
from vision_nav.extract import asset_links, normalise_repo, verify
from vision_nav.model import DEFAULT_MODELS, GeminiModel, ModelError
from vision_nav.trace import Tracer

ROOT = Path(__file__).resolve().parent


def build_goal(repo: str | None, prompt: str | None) -> str:
    if prompt:
        return prompt
    name = repo.split("/")[-1]
    return (f"Starting from the GitHub homepage, use GitHub's search to search for \"{name}\", open the repository "
            f"{repo} (exact owner and name), open its Releases page, and read the release that GitHub marks as "
            f"Latest: its version, tag commit, and author.")


def _relative(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def to_output(result, model, args, started: float, verification, tracer: Tracer) -> dict:
    ex = result.extraction
    out: dict = {}
    if ex:
        rel = ex.latest_release
        repo = normalise_repo(ex.repository) or args.repo
        out = {
            "repository": repo,
            "latest_release": {
                "version": rel.version,
                "tag": rel.tag,
                "author": rel.author,
                "title": rel.title,
                "published": rel.published,
                "marked_latest": rel.marked_latest,
                "release_notes_summary": rel.release_notes_summary,
                "assets": asset_links(repo, rel.version, rel.asset_names),
            },
        }
        if ex.answer and args.prompt:      # free-form answers only make sense for --prompt
            out["answer"] = ex.answer
    out["run"] = {
        "status": result.status,
        "reason": result.reason,
        "source_url": result.final_url,
        "steps": result.steps,
        "model_calls": model.calls,
        "model_usage": model.usage,
        "grounding": args.grounding,
        "duration_s": round(time.monotonic() - started, 1),
        "confidence": ex.confidence if ex else None,
        "extractor_issues": ex.issues if ex else None,
        "verification": verification,
        "trace_dir": _relative(tracer.dir),
    }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--repo", help="owner/name, e.g. openclaw/openclaw")
    target.add_argument("--prompt", help="natural-language task")
    p.add_argument("--url", default="https://github.com", help="start page (default: github.com)")
    p.add_argument("--model", action="append", help=f"Gemini model; repeat for a fallback chain (default: {DEFAULT_MODELS})")
    p.add_argument("--grounding", choices=["som", "coords"], default="som",
                   help="som: numbered boxes on the screenshot; coords: model returns pixel boxes itself")
    p.add_argument("--max-steps", type=int, default=25)
    p.add_argument("--timeout", type=float, default=600, help="wall-clock budget in seconds")
    p.add_argument("--headed", action="store_true", help="show the browser window")
    p.add_argument("--out", type=Path, help="write the JSON result to this file")
    p.add_argument("--trace-dir", type=Path, default=ROOT / "runs", help="where step screenshots go")
    p.add_argument("--no-trace", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    load_dotenv(ROOT / ".env")

    if args.repo and args.repo.count("/") != 1:
        p.error("--repo must look like owner/name")

    started = time.monotonic()
    tracer = Tracer(None if args.no_trace else args.trace_dir)
    try:
        model = GeminiModel(models=args.model)
    except ModelError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    goal = build_goal(args.repo, args.prompt)
    verification = None
    try:
        with BrowserSession(headless=not args.headed) as browser:
            browser.goto(args.url)
            agent = Agent(browser, model, goal, grounding=args.grounding, max_steps=args.max_steps,
                          deadline_s=args.timeout, tracer=tracer)
            result = agent.run()
            if result.extraction:
                verification = verify(result.extraction, browser.visible_text())
    except ModelError as e:
        print(f"error: model unavailable: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # browser crashes, navigation timeouts, ...
        logging.exception("run aborted")
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    out = to_output(result, model, args, started, verification, tracer)
    tracer.result(out)
    text = json.dumps(out, indent=2, ensure_ascii=False)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252
    print(text)
    return 0 if result.status == "success" else 2


if __name__ == "__main__":
    sys.exit(main())

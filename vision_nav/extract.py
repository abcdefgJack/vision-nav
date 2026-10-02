"""Turn the final page into structured data, then sanity-check it.

Extraction is vision-only. Verification afterwards compares the extracted strings
with the page's visible text: it never fills in values, it only flags ones that do
not appear on the page (i.e. likely misreads or hallucinations).
"""

from __future__ import annotations

import re

from .browser import BrowserSession
from .model import VisionModel
from .prompts import EXTRACT_TEMPLATE
from .schemas import Extraction


def extract(browser: BrowserSession, model: VisionModel, goal: str, pages: int = 3) -> tuple[Extraction, list[bytes]]:
    y = browser.page.evaluate("() => window.scrollY")
    shots = browser.screenshots_down(pages)
    browser.page.evaluate("(y) => window.scrollTo(0, y)", y)
    result = model.generate(EXTRACT_TEMPLATE.format(goal=goal), shots, Extraction)
    return result, shots


def verify(ex: Extraction, page_text: str) -> dict[str, bool | None]:
    """field -> True (found verbatim on page), False (not found: suspicious), None (not extracted)."""
    text = page_text.lower()
    rel = ex.latest_release
    checks: dict[str, bool | None] = {}
    for field in ("version", "tag", "author"):
        value = getattr(rel, field)
        checks[field] = None if not value else value.strip().lower() in text
    if ex.repository:
        owner, _, name = ex.repository.partition("/")
        checks["repository"] = owner.lower() in text and name.lower() in text
    return checks


def normalise_repo(repo: str | None) -> str | None:
    if not repo:
        return None
    repo = re.sub(r"\s+", "", repo)
    m = re.search(r"([\w.-]+/[\w.-]+)$", repo)
    return m.group(1) if m else repo


def asset_links(repo: str | None, version: str | None, names: list[str]) -> list[dict]:
    """Download URLs follow GitHub's documented, stable pattern. They are *derived*,
    not read off the page (link targets are not visible in a screenshot)."""
    if not (repo and version):
        return [{"name": n} for n in names]
    return [{"name": n, "download_url": f"https://github.com/{repo}/releases/download/{version}/{n}"}
            for n in names]

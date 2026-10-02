from __future__ import annotations

NAV_TEMPLATE = """You are a web-navigation agent driving a real Chromium browser. You only see screenshots
and act with the mouse and keyboard, like a careful human.

GOAL: {goal}

Current URL: {url}
Page title: {title}
The screenshot is the 1280x800 viewport.
{grounding}

Rules:
- Use the site's own UI (search box, links, tabs, menus). You cannot type URLs.
- Exactly one action per step.
- To search: click the search control, then `type` the query with submit=true.
- When several similarly named results appear, choose the one whose owner/name matches the target exactly
  (forks and mirrors often share the name).
- Use `extract` only when the information the goal asks for is visible right now. For "latest release"
  goals the release on screen must carry GitHub's "Latest" badge: the top entry of the releases list is
  NOT necessarily the one GitHub marks as Latest (e.g. maintenance/backport releases can be newer).
- Prefer one decisive click over long scrolling. A repository's front page usually summarises its
  Latest release in a sidebar box; clicking that entry opens the release directly. Release list pages
  can be extremely long, so open a specific release via its title link rather than scrolling through notes.
- If a click on a list entry does not change what you see, do not click it again; try another route.
- If a dialog, banner or overlay blocks the page, dismiss it (Escape or its close button).
- If feedback says your last action had no effect or you are repeating yourself, try a different approach.
- Use `fail` only when the goal is clearly impossible (e.g. the repository has no releases); explain why.

Recent steps (oldest first):
{history}
{feedback}"""

GROUNDING_SOM = """Interactive elements are outlined with coloured boxes, each with a number at its top-left corner.
To act on an element set `mark` to that number. If the element you need has no box, set `box_2d` instead
([ymin, xmin, ymax, xmax], normalised 0-1000)."""

GROUNDING_COORDS = """To act on an element set `box_2d` to its bounding box as [ymin, xmin, ymax, xmax],
normalised to 0-1000 over the screenshot."""

EXTRACT_TEMPLATE = """You are reading screenshots of a GitHub page (top to bottom, consecutive viewport captures).

The user's request: {goal}

Fill the schema with values read EXACTLY as displayed. Never infer, complete or invent a value: use null
when it is not visible or not legible, and mention it in `issues`.

Field guide:
- latest_release.version: the release's tag name, shown next to a tag icon (e.g. "v2026.1.29").
- latest_release.tag: the short commit SHA shown next to a commit icon (7 hex chars, e.g. "77e703c").
- latest_release.author: the account shown as having released it.
- latest_release.marked_latest: true ONLY if a "Latest" badge is attached to this specific release.
  If several releases are visible, describe the one with the "Latest" badge; if none has the badge,
  describe the most prominent release and set marked_latest=false.
- answer: only if the request asks something beyond these fields (e.g. "list its key features").
"""


def format_history(lines: list[str]) -> str:
    return "\n".join(lines) if lines else "(none yet)"

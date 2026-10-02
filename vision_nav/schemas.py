"""Structured contracts between the agent and the vision model."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    click = "click"
    type = "type"            # focus the target (if given), type text, optionally press Enter
    press = "press"          # a single key, e.g. Escape
    scroll = "scroll"
    back = "back"
    extract = "extract"      # the answer is on screen now -> hand off to the extractor
    fail = "fail"            # the model believes the goal is unreachable


class Decision(BaseModel):
    """One step of the navigation policy. Kept flat: Gemini's JSON mode is most
    reliable with simple, non-union schemas."""

    observation: str = Field(description="One sentence: what is on screen right now.")
    reasoning: str = Field(description="One or two sentences: why the chosen action moves toward the goal.")
    action: ActionType
    mark: Optional[int] = Field(None, description="Numbered box to act on (set-of-marks grounding).")
    box_2d: Optional[list[int]] = Field(
        None, description="[ymin, xmin, ymax, xmax] of the target, normalised 0-1000 (coords grounding).")
    text: Optional[str] = Field(None, description="Text to type (type) or key name (press).")
    submit: Optional[bool] = Field(None, description="Press Enter after typing.")
    scroll_direction: Optional[str] = Field(None, description="'up' or 'down'.")
    scroll_pages: Optional[float] = Field(None, description="How far to scroll, in screen heights (0.5-5, default 0.8).")
    expectation: str = Field(description="What should be visible after this action if it works.")


class LatestRelease(BaseModel):
    version: Optional[str] = Field(None, description="Release tag name exactly as shown, e.g. 'v2026.1.29'.")
    tag: Optional[str] = Field(None, description="Short commit SHA shown next to the tag (the value beside the commit icon).")
    author: Optional[str] = Field(None, description="Login of the account that published the release.")
    title: Optional[str] = Field(None, description="Release title/heading.")
    published: Optional[str] = Field(None, description="Publish date/time as displayed (relative or absolute).")
    marked_latest: bool = Field(description="True only if a 'Latest' badge is visibly attached to THIS release.")
    release_notes_summary: Optional[str] = Field(None, description="2-4 sentence summary of the visible notes.")
    asset_names: list[str] = Field(default_factory=list, description="File names listed under Assets, if visible.")


class Extraction(BaseModel):
    repository: Optional[str] = Field(None, description="owner/name as shown on the page.")
    latest_release: LatestRelease
    answer: Optional[str] = Field(None, description="Direct answer to the user's question, if it asked something beyond the core fields.")
    confidence: str = Field(description="'high', 'medium' or 'low'.")
    issues: Optional[str] = Field(None, description="Anything ambiguous, cut off or unreadable.")

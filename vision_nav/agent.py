"""Observe -> decide -> act loop with guard rails.

Guard rails (each one exists because of a failure seen while building this):
- step budget and wall-clock deadline
- "no visible effect" detection: the screen is identical after a click/type
- repetition detection: same action on the same URL several times
- invalid targets (mark id that does not exist, missing box) become feedback, not crashes
- extraction is re-checked: if the release read is not marked "Latest", the agent is sent back
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from .browser import VIEWPORT, Observation
from .extract import extract as default_extract
from .model import VisionModel
from .prompts import GROUNDING_COORDS, GROUNDING_SOM, NAV_TEMPLATE, format_history
from .schemas import ActionType, Decision, Extraction
from .trace import Tracer

log = logging.getLogger(__name__)

ALLOWED_KEYS = {"Enter", "Escape", "Tab", "ArrowDown", "ArrowUp", "PageDown", "PageUp", "Home", "End", "Backspace"}


class InvalidAction(ValueError):
    pass


@dataclass
class RunResult:
    status: str                         # "success" | "partial" | "failed"
    extraction: Extraction | None
    final_url: str | None
    steps: int
    reason: str | None = None
    history: list[str] = field(default_factory=list)


class Agent:
    def __init__(self, browser, model: VisionModel, goal: str, *, grounding: str = "som",
                 max_steps: int = 25, deadline_s: float = 300, max_extract_attempts: int = 3,
                 tracer: Tracer | None = None,
                 extractor: Callable = default_extract):
        if grounding not in ("som", "coords"):
            raise ValueError("grounding must be 'som' or 'coords'")
        self.browser = browser
        self.model = model
        self.goal = goal
        self.grounding = grounding
        self.max_steps = max_steps
        self.deadline_s = deadline_s
        self.max_extract_attempts = max_extract_attempts
        self.tracer = tracer or Tracer(None)
        self.extractor = extractor
        self.history: list[str] = []
        self._recent_sigs: deque[tuple] = deque(maxlen=6)

    # ------------------------------------------------------------------ loop
    def run(self) -> RunResult:
        start = time.monotonic()
        feedback: list[str] = []
        prev: tuple[str, ActionType] | None = None     # (fingerprint before action, action)
        no_effect_streak = 0
        extract_attempts = 0
        best: Extraction | None = None

        for step in range(1, self.max_steps + 1):
            if time.monotonic() - start > self.deadline_s:
                return self._finish("partial" if best else "failed", best, f"deadline of {self.deadline_s}s exceeded", step - 1)

            obs = self.browser.observe(with_marks=self.grounding == "som")
            shot_path = self.tracer.image(f"step_{step:02d}.png", obs.screenshot)

            if prev and prev[1] in (ActionType.click, ActionType.type, ActionType.press) and prev[0] == obs.fingerprint:
                no_effect_streak += 1
                feedback.append("Your last action had NO visible effect (the screen is pixel-identical). "
                                "The target may be wrong or not interactive; choose a different element or approach.")
                if no_effect_streak >= 4:
                    return self._finish("partial" if best else "failed", best, "stuck: repeated actions with no effect", step)
            else:
                no_effect_streak = 0

            decision = self.model.generate(self._nav_prompt(obs, feedback), [obs.screenshot], Decision)
            feedback = []
            log.info("step %d @ %s -> %s %s | %s", step, obs.url, decision.action.value,
                     self._target_str(decision), decision.reasoning)

            outcome = "ok"
            if decision.action == ActionType.fail:
                self._record(step, obs, decision, "agent gave up", shot_path)
                return self._finish("partial" if best else "failed", best, f"agent reported failure: {decision.reasoning}", step)

            if decision.action == ActionType.extract:
                extract_attempts += 1
                result, shots = self.extractor(self.browser, self.model, self.goal)
                for i, png in enumerate(shots):
                    self.tracer.image(f"step_{step:02d}_extract_{i}.png", png)
                best = result if best is None or result.latest_release.marked_latest else best
                self._record(step, obs, decision, f"extracted (marked_latest={result.latest_release.marked_latest})",
                             shot_path, extraction=result.model_dump())
                if result.latest_release.marked_latest or not self._goal_wants_latest():
                    return self._finish("success", result, None, step)
                if extract_attempts >= self.max_extract_attempts:
                    return self._finish("partial", best, "could not find a release marked 'Latest'", step)
                feedback.append(
                    f"The extractor read release '{result.latest_release.version}' but it is NOT marked 'Latest'. "
                    "Find the release that carries the 'Latest' badge (check the release list, scroll, or open it) "
                    "and extract again.")
                prev = None
                continue

            try:
                outcome = self._execute(decision, obs)
            except InvalidAction as e:
                outcome = f"invalid action: {e}"
                feedback.append(f"Your last action was invalid: {e}")

            sig = (obs.url, decision.action, decision.mark, tuple(decision.box_2d or ()), decision.text)
            self._recent_sigs.append(sig)
            if self._recent_sigs.count(sig) >= 3:
                feedback.append("You have repeated the same action on this page 3 times. It is not working; "
                                "change strategy (different element, scroll, go back).")

            self._record(step, obs, decision, outcome, shot_path)
            prev = (obs.fingerprint, decision.action)

        return self._finish("partial" if best else "failed", best, f"step budget of {self.max_steps} exhausted", self.max_steps)

    # ------------------------------------------------------------- helpers
    def _goal_wants_latest(self) -> bool:
        g = self.goal.lower()
        return "latest" in g or "current" in g or "newest" in g

    def _nav_prompt(self, obs: Observation, feedback: list[str]) -> str:
        fb = ("\nFeedback from the harness about your previous action:\n- " + "\n- ".join(feedback)) if feedback else ""
        return NAV_TEMPLATE.format(
            goal=self.goal, url=obs.url, title=obs.title,
            grounding=GROUNDING_SOM if self.grounding == "som" else GROUNDING_COORDS,
            history=format_history(self.history[-8:]), feedback=fb)

    def _point(self, d: Decision, obs: Observation) -> tuple[float, float] | None:
        if d.mark is not None and self.grounding == "som":
            if not 0 <= d.mark < len(obs.marks):
                raise InvalidAction(f"mark {d.mark} does not exist (valid: 0-{len(obs.marks) - 1})")
            m = obs.marks[d.mark]
            return m.cx, m.cy
        if d.box_2d:
            if len(d.box_2d) != 4:
                raise InvalidAction("box_2d must be [ymin, xmin, ymax, xmax]")
            ymin, xmin, ymax, xmax = (max(0, min(1000, v)) for v in d.box_2d)
            return (xmin + xmax) / 2000 * VIEWPORT["width"], (ymin + ymax) / 2000 * VIEWPORT["height"]
        return None

    def _execute(self, d: Decision, obs: Observation) -> str:
        b = self.browser
        if d.action == ActionType.click:
            pt = self._point(d, obs)
            if pt is None:
                raise InvalidAction("click needs a target (mark or box_2d)")
            b.click(*pt)
            return f"clicked at ({pt[0]:.0f},{pt[1]:.0f})"
        if d.action == ActionType.type:
            if not d.text:
                raise InvalidAction("type needs text")
            pt = self._point(d, obs)
            if pt is not None:
                b.click(*pt)
            b.type_text(d.text, bool(d.submit))
            return f"typed {d.text!r}" + (" + Enter" if d.submit else "")
        if d.action == ActionType.press:
            key = (d.text or "").strip()
            if key not in ALLOWED_KEYS:
                raise InvalidAction(f"key {key!r} not allowed; use one of {sorted(ALLOWED_KEYS)}")
            b.press(key)
            return f"pressed {key}"
        if d.action == ActionType.scroll:
            direction = (d.scroll_direction or "down").lower()
            if direction not in ("up", "down"):
                raise InvalidAction("scroll_direction must be 'up' or 'down'")
            pages = max(0.3, min(5.0, d.scroll_pages or 0.8))
            b.scroll(direction, pages)
            return f"scrolled {direction} {pages:g} screens"
        if d.action == ActionType.back:
            b.back()
            return "went back"
        raise InvalidAction(f"unsupported action {d.action}")

    @staticmethod
    def _target_str(d: Decision) -> str:
        if d.mark is not None:
            return f"[mark {d.mark}]"
        if d.box_2d:
            return f"[box {d.box_2d}]"
        return f"[{d.text!r}]" if d.text else ""

    def _record(self, step: int, obs: Observation, d: Decision, outcome: str, shot: str | None, **extra) -> None:
        self.history.append(f"{step}. on {obs.url}: {d.action.value} {self._target_str(d)} "
                            f"-- {d.reasoning} => {outcome}")
        self.tracer.event({"step": step, "url": obs.url, "title": obs.title, "decision": d.model_dump(mode="json"),
                           "outcome": outcome, "screenshot": shot, **extra})

    def _finish(self, status: str, ex: Extraction | None, reason: str | None, steps: int) -> RunResult:
        url = getattr(getattr(self.browser, "page", None), "url", None)
        return RunResult(status, ex, url, steps, reason, list(self.history))

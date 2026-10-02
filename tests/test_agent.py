"""Agent-loop tests with a scripted model and a fake browser: no network, no API quota.

They pin down the guard rails: target resolution, invalid actions, no-effect and
repetition detection, the "not marked Latest" retry, and budget exhaustion.
"""

from __future__ import annotations

import hashlib

import pytest

from vision_nav.agent import Agent
from vision_nav.browser import Mark, Observation
from vision_nav.schemas import ActionType, Decision, Extraction, LatestRelease


def decision(action, **kw) -> Decision:
    return Decision(observation="o", reasoning="r", action=action, expectation="e", **kw)


def extraction(latest: bool, version="v1.0.0") -> Extraction:
    return Extraction(repository="o/r", confidence="high",
                      latest_release=LatestRelease(version=version, tag="abc1234", author="me", marked_latest=latest))


class ScriptedModel:
    """Returns queued decisions for navigation calls; records prompts."""

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.prompts: list[str] = []
        self.calls = 0

    def generate(self, prompt, images, schema):
        self.calls += 1
        self.prompts.append(prompt)
        assert schema is Decision
        if not self.decisions:
            return decision(ActionType.scroll, scroll_direction="down")
        return self.decisions.pop(0)


class FakeBrowser:
    """Each click changes the 'screen' unless frozen; records what happened."""

    def __init__(self, n_marks=5, frozen=False):
        self.n_marks = n_marks
        self.frozen = frozen
        self.version = 0
        self.log: list[tuple] = []

    def observe(self, with_marks):
        png = hashlib.md5(str(self.version).encode()).digest()
        marks = [Mark(i, 100 + i, 200 + i) for i in range(self.n_marks)] if with_marks else []
        return Observation("https://example.test/", "t", png, png, marks)

    def _changed(self):
        if not self.frozen:
            self.version += 1

    def click(self, x, y):
        self.log.append(("click", x, y)); self._changed()

    def type_text(self, text, submit):
        self.log.append(("type", text, submit)); self._changed()

    def press(self, key):
        self.log.append(("press", key)); self._changed()

    def scroll(self, direction, amount=0.8):
        self.log.append(("scroll", direction, amount)); self._changed()

    def back(self):
        self.log.append(("back",)); self._changed()


def make_extractor(results):
    results = list(results)

    def _extract(browser, model, goal):
        return results.pop(0), []
    return _extract


GOAL = "open the repo and read the release GitHub marks as Latest"


def test_click_on_mark_uses_mark_centre():
    b = FakeBrowser()
    a = Agent(b, ScriptedModel([decision(ActionType.click, mark=3), decision(ActionType.extract)]), GOAL,
              extractor=make_extractor([extraction(True)]))
    res = a.run()
    assert res.status == "success"
    assert b.log[0] == ("click", 103, 203)


def test_coords_grounding_maps_normalised_box_to_pixels():
    b = FakeBrowser()
    d = decision(ActionType.click, box_2d=[500, 250, 500, 250])     # y=0.5, x=0.25
    a = Agent(b, ScriptedModel([d, decision(ActionType.extract)]), GOAL, grounding="coords",
              extractor=make_extractor([extraction(True)]))
    a.run()
    assert b.log[0] == ("click", 320.0, 400.0)


def test_invalid_mark_becomes_feedback_not_crash():
    b = FakeBrowser(n_marks=2)
    m = ScriptedModel([decision(ActionType.click, mark=9), decision(ActionType.extract)])
    res = Agent(b, m, GOAL, extractor=make_extractor([extraction(True)])).run()
    assert res.status == "success"
    assert b.log == []                                  # nothing was clicked
    assert "mark 9 does not exist" in m.prompts[1]


def test_disallowed_key_is_rejected():
    m = ScriptedModel([decision(ActionType.press, text="ctrl+w"), decision(ActionType.extract)])
    Agent(FakeBrowser(), m, GOAL, extractor=make_extractor([extraction(True)])).run()
    assert "not allowed" in m.prompts[1]


def test_no_effect_is_reported_and_eventually_aborts():
    b = FakeBrowser(frozen=True)
    m = ScriptedModel([decision(ActionType.click, mark=1)] * 10)
    res = Agent(b, m, GOAL, max_steps=10).run()
    assert "NO visible effect" in m.prompts[1]
    assert res.status == "failed" and "stuck" in res.reason


def test_repetition_is_flagged():
    m = ScriptedModel([decision(ActionType.scroll, scroll_direction="down")] * 4)
    Agent(FakeBrowser(), m, GOAL, max_steps=4).run()
    assert any("repeated the same action" in p for p in m.prompts)


def test_release_not_marked_latest_sends_agent_back():
    m = ScriptedModel([decision(ActionType.extract), decision(ActionType.click, mark=0), decision(ActionType.extract)])
    ex = make_extractor([extraction(False, "v2.0.1-backport"), extraction(True, "v2.1.0")])
    res = Agent(FakeBrowser(), m, GOAL, extractor=ex).run()
    assert res.status == "success"
    assert res.extraction.latest_release.version == "v2.1.0"
    assert "NOT marked 'Latest'" in m.prompts[1]


def test_gives_partial_result_when_latest_never_found():
    m = ScriptedModel([decision(ActionType.extract)] * 3)
    ex = make_extractor([extraction(False)] * 3)
    res = Agent(FakeBrowser(), m, GOAL, extractor=ex, max_extract_attempts=3).run()
    assert res.status == "partial"
    assert res.extraction is not None


def test_non_latest_goal_accepts_first_extraction():
    m = ScriptedModel([decision(ActionType.extract)])
    res = Agent(FakeBrowser(), m, "list the assets of release v1.0.0", extractor=make_extractor([extraction(False)])).run()
    assert res.status == "success"


def test_agent_fail_action_stops_run():
    m = ScriptedModel([decision(ActionType.fail)])
    res = Agent(FakeBrowser(), m, GOAL).run()
    assert res.status == "failed" and "agent reported failure" in res.reason


def test_step_budget():
    res = Agent(FakeBrowser(), ScriptedModel([]), GOAL, max_steps=3).run()
    assert res.status == "failed" and "budget" in res.reason


def test_rejects_unknown_grounding():
    with pytest.raises(ValueError):
        Agent(FakeBrowser(), ScriptedModel([]), GOAL, grounding="dom")

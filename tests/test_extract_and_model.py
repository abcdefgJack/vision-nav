from __future__ import annotations

import time

import pytest

from vision_nav.extract import asset_links, normalise_repo, verify
from vision_nav.model import BadOutput, parse_json, retry_delay
from vision_nav.schemas import Decision, Extraction, LatestRelease


def ex(**rel) -> Extraction:
    base = dict(version="v2026.9.7", tag="c074824", author="github-actions", marked_latest=True)
    base.update(rel)
    return Extraction(repository="openclaw/openclaw", confidence="high", latest_release=LatestRelease(**base))


PAGE = "openclaw / openclaw ... openclaw 2026.9.7 Latest github-actions released this v2026.9.7 c074824"


def test_verify_all_found():
    assert verify(ex(), PAGE) == {"version": True, "tag": True, "author": True, "repository": True}


def test_verify_flags_misread_sha():
    # a classic OCR confusion: 0 vs o / 4 vs a
    assert verify(ex(tag="c07a824"), PAGE)["tag"] is False


def test_verify_marks_missing_fields_as_none():
    assert verify(ex(author=None), PAGE)["author"] is None


@pytest.mark.parametrize("raw,expected", [
    ("openclaw/openclaw", "openclaw/openclaw"),
    ("openclaw / openclaw", "openclaw/openclaw"),
    ("github.com/openclaw/openclaw", "openclaw/openclaw"),
    (None, None),
])
def test_normalise_repo(raw, expected):
    assert normalise_repo(raw) == expected


def test_asset_links_are_derived_from_tag():
    links = asset_links("o/r", "v1", ["a.zip"])
    assert links == [{"name": "a.zip", "download_url": "https://github.com/o/r/releases/download/v1/a.zip"}]
    assert asset_links(None, "v1", ["a.zip"]) == [{"name": "a.zip"}]


def test_parse_json_tolerates_code_fences():
    text = 'Sure:\n```json\n{"observation":"o","reasoning":"r","action":"scroll","expectation":"e"}\n```'
    assert parse_json(text, Decision).action.value == "scroll"


def test_parse_json_rejects_garbage():
    with pytest.raises(BadOutput):
        parse_json("I cannot help with that", Decision)
    with pytest.raises(BadOutput):
        parse_json("", Decision)


def test_retry_delay_honours_server_hint():
    err = Exception("429 RESOURCE_EXHAUSTED ... 'retryDelay': '17s'")
    assert retry_delay(err, 0) == 18


def test_retry_delay_backoff_is_capped():
    assert retry_delay(Exception("boom"), 10) <= 30


def test_overloaded_and_exhausted_models_are_deprioritised():
    from vision_nav.model import GeminiModel
    m = GeminiModel.__new__(GeminiModel)
    m.models = ["a", "b", "c"]
    m.exhausted = {"b"}
    m.cooldown_until = {"a": time.monotonic() + 60}
    assert m._candidates() == ["c", "a"]

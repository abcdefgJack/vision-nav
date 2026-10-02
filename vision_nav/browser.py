"""Thin Playwright wrapper.

Every interaction goes through the mouse and keyboard at pixel coordinates, never
through a site-specific selector. The only DOM access is a *generic* scan for
"things a user could click" (links, buttons, inputs, ARIA widgets) so we can draw
numbered boxes on the screenshot (set-of-marks). That scan knows nothing about
GitHub, so it keeps working when GitHub's markup changes. `grounding="coords"`
skips it entirely and relies on the model's own pixel localisation.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

from playwright.sync_api import Browser, Error as PWError, Page, Playwright, TimeoutError as PWTimeout, sync_playwright

log = logging.getLogger(__name__)

VIEWPORT = {"width": 1280, "height": 800}

# Generic interactivity query -- intentionally site-agnostic.
_MARK_JS = r"""
() => {
  const Q = 'a[href], button, input:not([type=hidden]), textarea, select, summary,'
          + '[role=button], [role=link], [role=tab], [role=menuitem], [role=option],'
          + '[role=searchbox], [role=combobox], [contenteditable=true]';
  const vw = window.innerWidth, vh = window.innerHeight;
  const out = [];
  const seen = new Set();
  for (const el of document.querySelectorAll(Q)) {
    const r = el.getBoundingClientRect();
    if (r.width < 4 || r.height < 4) continue;
    if (r.bottom <= 0 || r.right <= 0 || r.top >= vh || r.left >= vw) continue;
    const st = getComputedStyle(el);
    if (st.visibility === 'hidden' || st.display === 'none' || +st.opacity === 0) continue;
    const cx = Math.min(Math.max(r.left + r.width / 2, 1), vw - 1);
    const cy = Math.min(Math.max(r.top + r.height / 2, 1), vh - 1);
    const top = document.elementFromPoint(cx, cy);
    if (!top || !(top === el || el.contains(top) || top.contains(el))) continue;  // occluded
    const key = Math.round(r.left) + ',' + Math.round(r.top) + ',' + Math.round(r.width);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({x: r.left, y: r.top, w: r.width, h: r.height, cx, cy});
  }
  return out;
}
"""

_DRAW_JS = r"""
(marks) => {
  const layer = document.createElement('div');
  layer.id = '__vision_nav_marks__';
  layer.style.cssText = 'position:fixed;inset:0;pointer-events:none;z-index:2147483647';
  const palette = ['#e6194b','#3cb44b','#4363d8','#f58231','#911eb4','#008080','#9a6324','#800000'];
  marks.forEach((m, i) => {
    const c = palette[i % palette.length];
    const box = document.createElement('div');
    box.style.cssText = `position:absolute;left:${m.x}px;top:${m.y}px;width:${m.w}px;height:${m.h}px;`
                      + `outline:2px solid ${c};box-sizing:border-box`;
    const tag = document.createElement('div');
    tag.textContent = String(i);
    // Inside the box's top-left corner: labels drawn *above* boxes get read as
    // belonging to the previous row in tightly stacked lists.
    tag.style.cssText = `position:absolute;left:${m.x}px;top:${m.y}px;background:${c};`
                      + 'color:#fff;font:bold 11px/14px monospace;padding:0 3px;border-radius:2px';
    layer.appendChild(box);
    layer.appendChild(tag);
  });
  document.documentElement.appendChild(layer);
}
"""

_CLEAR_JS = "() => document.getElementById('__vision_nav_marks__')?.remove()"


@dataclass
class Mark:
    id: int
    cx: float
    cy: float


@dataclass
class Observation:
    url: str
    title: str
    screenshot: bytes          # what the model sees (with marks if grounding == "som")
    raw_screenshot: bytes      # clean screenshot, for traces and change detection
    marks: list[Mark]

    @property
    def fingerprint(self) -> str:
        return hashlib.md5(self.raw_screenshot).hexdigest()


class BrowserSession:
    def __init__(self, headless: bool = True, nav_timeout_ms: int = 30_000):
        self.headless = headless
        self.nav_timeout_ms = nav_timeout_ms
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self.page: Page | None = None

    def __enter__(self) -> "BrowserSession":
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        ctx = self._browser.new_context(viewport=VIEWPORT, locale="en-US", color_scheme="light")
        ctx.set_default_timeout(self.nav_timeout_ms)
        self.page = ctx.new_page()
        # Links that open a new tab should stay in our single page.
        ctx.on("page", self._adopt_popup)
        return self

    def __exit__(self, *exc) -> None:
        if self._browser:
            self._browser.close()
        if self._pw:
            self._pw.stop()

    def _adopt_popup(self, popup: Page) -> None:
        log.info("popup opened (%s); following it in the main tab", popup.url)
        try:
            popup.wait_for_load_state("domcontentloaded", timeout=10_000)
            url = popup.url
            popup.close()
            if url and url != "about:blank":
                self.goto(url)
        except PWTimeout:
            pass

    # ----- navigation ---------------------------------------------------------
    def goto(self, url: str) -> None:
        self.page.goto(url, wait_until="domcontentloaded", timeout=self.nav_timeout_ms)
        self.settle()

    def settle(self, quiet_ms: int = 2_500) -> None:
        """Best-effort wait for the page to stop changing. GitHub keeps long-lived
        connections open, so networkidle may never fire -- cap it. The initial pause
        gives client-side (Turbo) navigations triggered by a click time to start."""
        self.page.wait_for_timeout(500)
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=self.nav_timeout_ms)
            self.page.wait_for_load_state("networkidle", timeout=quiet_ms)
        except PWTimeout:
            pass
        self.page.wait_for_timeout(300)

    def back(self) -> None:
        self.page.go_back(wait_until="domcontentloaded")
        self.settle()

    # ----- observation --------------------------------------------------------
    def observe(self, with_marks: bool, retries: int = 4) -> Observation:
        """A click can start a navigation *after* settle() returned; if the page
        is torn down mid-capture, wait for the new document and capture again."""
        for attempt in range(retries):
            try:
                return self._observe(with_marks)
            except PWError as e:
                if "context was destroyed" not in str(e) and "navigat" not in str(e).lower():
                    raise
                if attempt == retries - 1:
                    raise
                log.info("page navigated during capture; retrying")
                self.settle()
        raise AssertionError("unreachable")

    def _observe(self, with_marks: bool) -> Observation:
        self.page.evaluate(_CLEAR_JS)
        raw = self.page.screenshot(type="png")
        marks: list[Mark] = []
        shot = raw
        if with_marks:
            boxes = self.page.evaluate(_MARK_JS)
            marks = [Mark(i, b["cx"], b["cy"]) for i, b in enumerate(boxes)]
            self.page.evaluate(_DRAW_JS, boxes)
            shot = self.page.screenshot(type="png")
            self.page.evaluate(_CLEAR_JS)
        return Observation(self.page.url, self.page.title(), shot, raw, marks)

    def screenshots_down(self, pages: int) -> list[bytes]:
        """Viewport screenshots from the current position downwards (for extraction)."""
        self.page.evaluate(_CLEAR_JS)
        shots = []
        for i in range(pages):
            shots.append(self.page.screenshot(type="png"))
            at_bottom = self.page.evaluate(
                "() => window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2")
            if at_bottom or i == pages - 1:
                break
            self.page.mouse.wheel(0, VIEWPORT["height"] * 0.85)
            self.page.wait_for_timeout(400)
        return shots

    def visible_text(self) -> str:
        """Used ONLY to verify extracted values after the fact, never to extract them."""
        return self.page.evaluate("() => document.body.innerText")

    # ----- actions (all pixel based) -------------------------------------------
    def click(self, x: float, y: float) -> None:
        self.page.mouse.click(x, y)
        self.settle()

    def type_text(self, text: str, submit: bool) -> None:
        self.page.keyboard.type(text, delay=20)
        if submit:
            self.page.keyboard.press("Enter")
        self.settle()

    def press(self, key: str) -> None:
        self.page.keyboard.press(key)
        self.settle()

    def scroll(self, direction: str, amount: float = 0.8) -> None:
        """amount is in viewport heights."""
        dy = VIEWPORT["height"] * amount * (1 if direction == "down" else -1)
        self.page.mouse.move(VIEWPORT["width"] / 2, VIEWPORT["height"] / 2)
        self.page.mouse.wheel(0, dy)
        self.page.wait_for_timeout(500)


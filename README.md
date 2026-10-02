# vision-nav: GitHub release extraction driven by a vision model

A command-line agent that opens github.com in a real browser, **looks at screenshots**, and decides
where to click and what to type until it reaches a repository's Latest release. Then it reads the
release details off the screen and returns JSON. It uses no CSS selectors, XPath, or GitHub-specific
DOM knowledge, and it never types a URL: every navigation is a mouse click or a keystroke.

```
github.com ─▶ search "openclaw" ─▶ openclaw/openclaw ─▶ Releases ─▶ release marked "Latest" ─▶ JSON
```

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
python -m playwright install chromium
cp .env.example .env        # then put your key in .env
```

The model is **Google Gemini**. A free API key from <https://aistudio.google.com/apikey> is enough
and needs no credit card. You can also set `GEMINI_API_KEY` in the environment instead of using `.env`.

## Run

```bash
# The task from the brief
python navigate.py --repo openclaw/openclaw

# Natural-language goal (bonus)
python navigate.py --prompt "search for openclaw and get the current release and related tags"
python navigate.py --prompt "Find the latest React release and list its key features"

# Any repository (bonus)
python navigate.py --repo microsoft/vscode

# Useful flags
python navigate.py --repo openclaw/openclaw --headed          # watch the browser
python navigate.py --repo openclaw/openclaw --grounding coords # no DOM at all: the model returns pixel boxes
python navigate.py --repo openclaw/openclaw --out result.json -v
```

| flag | default | meaning |
|---|---|---|
| `--repo owner/name` / `--prompt "..."` | one of them is required | target repository or a free-form goal |
| `--url` | `https://github.com` | start page |
| `--grounding som\|coords` | `som` | how a click target is chosen (see below) |
| `--model NAME` (repeatable) | fallback chain in `vision_nav/model.py` | Gemini model(s) to use |
| `--max-steps` / `--timeout` | 25 / 600 s | step and wall-clock budgets |
| `--headed` | off | show the browser |
| `--out FILE` | stdout only | also write the JSON to a file |
| `--trace-dir` / `--no-trace` | `runs/` | per-step screenshots + decisions |

Exit codes: `0` success, `2` finished without a confident result (`run.status` is `partial` or `failed`), `1` error.

## Output

See [`sample_output.json`](sample_output.json). The three fields from the brief come first:

```json
{
  "repository": "openclaw/openclaw",
  "latest_release": {
    "version": "v2026.9.7",
    "tag": "c074824",
    "author": "github-actions",
    "title": "openclaw 2026.9.7",
    "published": "2 days ago",
    "marked_latest": true,
    "release_notes_summary": "...",
    "assets": [{"name": "...", "download_url": "https://github.com/.../releases/download/v2026.9.7/..."}]
  },
  "run": {"status": "success", "steps": 6, "verification": {"version": true, "tag": true, "author": true}, "...": "..."}
}
```

The `run` object reports how the result was obtained: the status, step and model-call counts, which
models answered, extractor confidence, verification checks, and the trace folder.

## How it works

```
navigate.py ── Agent loop (vision_nav/agent.py) ──────────────────────────────────────────┐
                 │ observe: screenshot (+ numbered boxes)      BrowserSession (Playwright)  │
                 │ decide:  Gemini → Decision JSON             mouse/keyboard at pixels     │
                 │ act:     click / type / press / scroll / back                            │
                 │ guard:   step+time budget, no-effect + repetition detection, invalid     │
                 │          targets become feedback to the model                            │
                 └ extract: 1–3 screenshots → Gemini → Extraction JSON → verify ───────────┘
```

* **Policy.** Each step sends one screenshot, the goal, and a short history to the model. The model
  returns a `Decision` (`vision_nav/schemas.py`) that holds its observation, its reasoning, an action,
  and the result it expects. Gemini's JSON-schema mode enforces the structure, and Pydantic validates
  it again.
* **Grounding** (how "click the Releases link" becomes a pixel):
  * `som` (default, set-of-marks): a generic scan collects anything a user could click (links, buttons,
    inputs, ARIA widgets) and draws a numbered box over each one. The model picks a number, and the
    click lands at that element's centre. The scan contains nothing specific to GitHub.
  * `coords`: the page's structure is not used at all. The model returns a bounding box in normalised
    coordinates, and the agent clicks its centre.
* **Extraction** is a separate, focused call that takes one to three consecutive viewport screenshots
  and fills `Extraction`. The extractor also reports whether the release it read carries GitHub's
  "Latest" badge. If it does not, the agent is told so and goes back to find the badged release.
* **Verification.** After extraction, each extracted string is checked against the page's visible
  text. This step never supplies values. It only flags ones that do not appear on the page, which
  catches misread SHAs and hallucinations.
* **Traces.** `runs/<timestamp>/` holds every screenshot the model saw, `steps.jsonl` (each decision
  with its outcome), and `result.json`. A complete trace of an openclaw run is committed in
  [`examples/openclaw-trace/`](examples/openclaw-trace/).

## Tests

```bash
python -m pytest                 # 25 offline tests: agent guard rails, parsing, verification (no API calls)
python scripts/run_scenarios.py  # live end-to-end scenarios, graded against the GitHub REST API
```

The scenario runner uses the GitHub API **only to grade** the agent, never to help it. Results are in
`scenarios/`.

## Layout

```
navigate.py              CLI
vision_nav/agent.py      observe → decide → act loop and guard rails
vision_nav/browser.py    Playwright session, set-of-marks overlay, pixel actions
vision_nav/model.py      Gemini client: JSON mode, retries, model fallback chain
vision_nav/schemas.py    Decision / Extraction contracts
vision_nav/extract.py    extraction, verification, derived download links
vision_nav/prompts.py    prompt templates
vision_nav/trace.py      run artefacts
tests/                   offline unit tests
scripts/run_scenarios.py live scenarios + grading
OBSERVATIONS.md          approach, what worked / didn't, trade-offs, next steps
```

## Assumptions

* **`tag`** in the brief's example (`77e703c`) is the short commit SHA that GitHub shows beside the
  tag, so it is reported that way. The tag *name* is reported as `version`.
* **"Latest release"** means the release GitHub badges as **Latest**, which is not necessarily the
  release published most recently. For openclaw on 2026-10-01 these differ: `v2026.8.34`, a backport,
  was published after `v2026.9.7`, but `v2026.9.7` is the one marked Latest.
* **`author`** is the account shown on the page. Bots appear without the API's `[bot]` suffix
  (`github-actions`, not `github-actions[bot]`).
* **Download URLs** are derived from GitHub's stable `releases/download/<tag>/<file>` pattern, because
  link targets are not visible in a screenshot. The file names themselves are read from the page.

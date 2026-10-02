# Observations

## Approach

The agent runs an observe → decide → act loop. The design keeps each model call small and makes
failures visible rather than silent.

* **Two model roles.** A *navigation policy* sees one screenshot per step and returns one typed action.
  A separate *extractor* reads one to three screenshots of the final page into a typed schema, with
  instructions to read exactly and never infer.
* **Two grounding modes.** In `som` (set-of-marks), a site-agnostic scan draws numbered boxes over
  clickable elements and the model picks a number. In `coords`, the model returns a pixel box with no
  DOM involved. Both act through the mouse and keyboard, and neither uses a selector written for GitHub.
* **Structured output.** Gemini's JSON-schema mode is backed by Pydantic. Each `Decision` states what
  the model sees, why it acts, and what it expects to happen next, which makes traces easy to read.
* **Verify, don't trust.** Extracted version, SHA and author strings are checked against the page's
  visible text. This check only flags values that are not on the page (for example `c07a824` read in
  place of `c074824`). It never supplies them.

## Results

Live scenarios from `scripts/run_scenarios.py`, graded against the GitHub REST API:

| scenario | result | steps | extracted | grade |
|---|---|---|---|---|
| `--repo openclaw/openclaw` (som) | success | 5 | v2026.9.7 · c074824 · github-actions | pass |
| `--repo openclaw/openclaw --grounding coords` | success | 5 | same | pass |
| `--prompt "search for openclaw and get the current release and related tags"` | success | 9 | v2026.9.7 · c074824 · github-actions | pass |
| `--prompt "Find the latest React release and list its key features"` | success | 5 | v19.3.0 + feature list | pass |
| `--repo microsoft/vscode` | success | 6 | 1.140.0 · 07f806f · Yoyokrazy | pass |
| `--repo torvalds/linux` (has no releases) | failed with a reason | 5 | nothing invented | pass |

The openclaw path is: search icon → type "openclaw" → pick `openclaw/openclaw` → click the
**Latest** entry in the repository's Releases sidebar → extract. `coords` grounding hit small sidebar
links with no DOM help. There are also 25 offline unit tests covering the guard rails, which use a
scripted model and a fake browser.

## What didn't work, and what changed

1. **"Latest" is not "newest."** On `/releases`, the top entry was `2026.8.34`, a backport published
   *after* `2026.9.7`, which is the release GitHub actually marks Latest. The extractor's
   `marked_latest` flag caught this, but the agent then spent about 15 steps paging through release
   notes that run to tens of thousands of pixels. **Fix:** extraction must confirm the "Latest" badge,
   otherwise the agent gets feedback and goes back. The prompt also now prefers one decisive click
   over scrolling. Runs dropped from more than 20 steps to 5–6.
2. **Set-of-marks labels were ambiguous in dense lists.** Labels drawn above each box overlapped the
   previous 32 px row, so the model clicked `2026.8.33` while intending `2026.9.7`. Drawing labels
   inside each box fixed it.
3. **Navigation races.** A click can start a client-side navigation after the settle wait returns, which
   destroys the page mid-screenshot. Capture now retries once the new document loads.
4. **The free tier is the bottleneck.** Some models allow about 20 requests per day, and `503
   overloaded` errors were frequent. I added a model fallback chain, detection of exhausted daily
   quotas, and a 60 s cooldown for overloaded models. Most wall-clock time is still spent on retries,
   not browsing.
5. **Assets are only partly captured.** openclaw's notes include a ~330-name "Thanks" list, which
   pushes Assets below the three screenshots the extractor sees.

## Trade-offs

* **`som` vs `coords`.** `som` is more precise on small targets, but it depends on semantic clickable
  elements, so canvas or bare `div` buttons get no box. `coords` needs no DOM but can miss by a few
  pixels. I made `som` the default, with `box_2d` as a fallback inside it.
* **Domain hints in plain language.** The prompt says a repository's sidebar summarises its Latest
  release. That is GitHub knowledge, but stated as language rather than structure, so it survives
  markup changes.
* **One screenshot per step** keeps calls cheap. Because the model cannot compare frames, the harness
  detects actions with no visible effect by hashing screenshots.
* **The DOM is used only for verification and click targets,** never to extract values.

## Limitations

* A run takes 1–4 minutes, almost all of it free-tier retries.
* The extractor sees at most three viewports.
* Dates are reported as displayed ("2 days ago").
* There is no handling for logins, CAPTCHAs, or GitHub's own rate-limit pages.
* Results vary with whichever model in the fallback chain answers.

## Next steps

1. **Plan first for free-form prompts.** Rewrite the goal into explicit sub-goals (repository, page,
   fields) before navigating, so ambiguous phrasing (e.g. "related tags") cannot pull a smaller model
   onto the wrong page.
2. **Scroll-and-stitch extraction.** Keep reading until the requested section, such as Assets, has
   been fully seen.
3. **An offline benchmark.** Record pages, then replay them across models to measure accuracy and step
   counts without spending live quota.
4. **Crop and zoom small text** such as SHAs before extraction.
5. **Assign models per role:** a cheap model for navigation and a stronger one for extraction.

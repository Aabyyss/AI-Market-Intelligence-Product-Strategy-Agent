# Demo assets

There is already a finished video: **[`demo.mp4`](demo.mp4)** — 76 seconds,
1280×720, 30 fps, 2.4 MB, committed to the repo. The rest of this file is how
it was made, how to re-make it, and how to tell the story around it.

| | what you get | how long | effort |
|---|---|---|---|
| **A · the rendered reel** | `docs/demo.mp4`, byte-for-byte reproducible | 76 s | none — it's committed |
| **B · live screen recording** | you narrate the real pipeline running | 90 s | one take, no cuts |
| **C · AI-generated explainer** | prompt for Sora / Veo / Runway / Kling | 30–60 s | paste a prompt |

---

## A · The rendered reel (already in the repo)

`docs/demo.html` is one self-contained file: no assets, no network, no build
step. It animates the whole story — architecture, the data pipeline, the five
agents, the report's citation audit, the eval metrics, the CI gate — in 76
seconds. `render_demo.py` turns it into a video without a screen recorder.

```bash
python render_demo.py                 # docs/demo.mp4, 30 fps, 1280x720
python render_demo.py --fps 24        # smaller file
python render_demo.py --format png    # lossless frames (about 2x slower)
python render_demo.py --only-seconds 5   # smoke-test the pipeline
```

It drives headless Chrome over the DevTools Protocol and asks the page to
freeze at an exact timestamp for every frame (`?render=1`), so the output does
not depend on the machine keeping up: same input, same video. Roughly 6
minutes for the full 2280 frames, single-threaded, on a laptop. Needs a
Chromium browser (Chrome or Edge) and ffmpeg — `pip install imageio-ffmpeg`
if you don't have one on PATH.

The encoded file carries a hash of the reel it came from, and
`tests/test_demo_assets.py` checks it against `docs/demo.html`. So editing one
word or one scene duration in the reel fails the suite until you re-render,
rather than quietly leaving a demo that shows numbers the project no longer
claims.

### Or record the reel yourself (no browser automation)

1. Open it: `start docs/demo.html` (Windows) · `open docs/demo.html` (macOS).
2. Press <kbd>F11</kbd> for fullscreen (the stage scales to any window).
3. Record the window — **Win+G** (Xbox Game Bar), **Cmd+Shift+5** (macOS),
   or OBS if you have it. 1280×720 or larger.
4. Hit <kbd>R</kbd> to restart at scene 1, wait for it to loop through,
   stop recording.

Controls: <kbd>Space</kbd> pause/resume · <kbd>R</kbd> restart ·
<kbd>←</kbd>/<kbd>→</kbd> step scenes · click to advance.

**Add narration** in any editor (or read the voiceover script in
section C and lay it over the top — the scenes line up with it). The
committed video is silent; the script below is written to sit over it.

---

## B · Live screen recording (the honest demo)

This is the one that convinces engineers, because nothing is staged: the
retrieval is real, the model is real, and the failures are visible.

### Prep (do this before recording)

```bash
# 1. A corpus and an index must exist
python run_pipeline.py && python run_index.py     # or: python run_corpus.py seed

# 2. A local model (Ollama). On machines where GPU discovery hangs:
#    OLLAMA_VULKAN=false ollama serve
ollama pull llama3.2

# 3. Pre-warm everything so the take isn't 20 minutes long
python run_ask.py "warm up" --top 1
```

Two terminals, large font (16–18 pt), 1280×720 capture. If you have an
`OPENAI_API_KEY`, export `LLM_PROVIDER=openai` for the take — the same
run takes seconds instead of minutes. Say so out loud when you do it.

### Shot list

| t | screen | command | say |
|---|---|---|---|
| 0:00 | architecture diagram (slide, or `docs/demo.html` scene 1) | — | "Public competitor discussions in, evidence-backed product strategy out. Four layers: collection, retrieval, agents, orchestration." |
| 0:10 | terminal 1 | `python run_search.py "developer payouts" --top 3` | "Retrieval first. Local embeddings, no API key — cosine search over a sqlite-vec index. Every hit carries the URL it came from." |
| 0:22 | terminal 1 | `python run_ask.py "Why do app developers complain about Shopify payouts?" --top 3` | "Grounded Q&A. The model sees only the numbered evidence, must cite `[n]`, and the code validates every citation against what was actually retrieved." |
| 0:38 | terminal 1 | `python run_eval.py` | "Measured, not asserted: recall@5 0.82, MRR 1.00 on hand-labeled questions. Retrieval needs no LLM, so this runs in CI and fails the build on a regression." |
| 0:48 | terminal 2 | `python run_api.py` then `curl -s localhost:8000/health` | "The same code is a service — corpus size, embedding model, and whether an LLM is reachable, all in one health payload." |
| 0:56 | terminal 2 | `curl -s -X POST localhost:8000/reports -H 'content-type: application/json' -d '{"brief":"fees and developer payouts"}'` | "Reports are jobs, not requests: 202 with a job id. A full report is minutes of LLM work — n8n polls this instead of holding a connection open." |
| 1:02 | terminal 2 | `curl -s localhost:8000/jobs/<id>` (repeat) | "One worker, so a second report doesn't thrash the box. The job object carries status, duration, and the typed summary." |
| 1:18 | editor | open `data/reports/market_report_*.md` | "And the output: every claim cited, a critic verdict per claim, and a mechanical audit — code, not an LLM — confirming all 15 citations point at posts that were really retrieved." |
| 1:30 | terminal 1 | `python -m pytest tests/ -q` | "95 tests. The retrieval index is pinned to a reference numpy ranking, the API is tested through real HTTP with no LLM, and the whole pipeline runs without a model in CI." |

Total: ~90 seconds. **Don't hide the weak parts** — if the critic
over-flags a claim, point at it and say the verdict counts come from
typed data rather than the model's own arithmetic. That is the most
credible 10 seconds in the video.

### The 10-second beat worth saving for

When you point at the report's mechanical audit, you are often looking
at a *caught* failure, and that is the whole product. On the last
verified run through the API (`data/reports/api_report_demo.md`, job
`abf900ff1630`, 676 s on CPU, llama3.2 via Ollama):

```text
### Mechanical citation audit
- 9 distinct in-range citations used
- 1 out-of-range citation(s): [10]      <- the model cited a 10th source; only 9 existed
- ⚠ claims without any citation: [14]
- critic verdicts (typed, code-counted): 1 supported, 3 partial, 10 unsupported
```

The model invented a citation number, and the pipeline dropped it and
said so, on the page, in the same run. That is the sentence to end the
demo on: "a citation that doesn't point at a retrieved post cannot reach
the report."

---

## C · AI-generated explainer (prompt)

Paste into Sora, Veo, Runway, Kling, Pika — anything with a text-to-video
box. Generate 4–6 clips and cut them together, or use the single-prompt
version below.

### Master prompt (60 s, 6 clips)

> **Style:** dark technical explainer, deep navy background (#080b11),
> thin glowing green (#4ade80) and blue (#5aa9f7) accents, monospace
> terminal text, subtle grid texture, shallow depth of field, slow
> deliberate camera moves, no people, no stock footage faces. Colour
> palette and typography must stay identical across every clip.
>
> **Clip 1 (0–8 s) — data flows in.** Camera drifts over a dark grid
> plane. Glowing green data packets stream from three labelled source
> nodes ("HACKER NEWS", "REVIEWS", "REDDIT") along thin light trails
> that converge into a single terminal-shaped gateway. Slow push in.
>
> **Clip 2 (8–20 s) — the pipeline.** Macro shot of a dark terminal
> window. Monospace lines type themselves and resolve into glowing
> green status output: `posts 83`, `chunks 115`,
> `model bge-small-en-v1.5 (384 dims, local)`. Behind the terminal, faint
> vector dots arrange themselves into a dense 3D sphere, then settle
> into a neat index. Camera slowly pulls back.
>
> **Clip 3 (20–34 s) — five agents.** A horizontal chain of five
> hexagonal nodes lights up in sequence, each one labelled in clean
> monospace: RESEARCH, COMPETITOR, CUSTOMER, STRATEGY, CRITIC. Between
> nodes, thin beams of light pulse. The CRITIC node flashes amber, and
> a small red badge appears above it. Camera tracks left to right.
>
> **Clip 4 (34–46 s) — the report.** A markdown document materialises
> in mid-air, lines of text writing themselves. Blue `[3, 11]` citation
> markers glow and thin light threads connect each marker back to a
> source card floating in the background. One amber line pulses.
>
> **Clip 5 (46–54 s) — measurement.** Clean dashboard: five horizontal
> progress bars animate to 100%, 100%, 82%, 73%, 58%, each with a green
> waveform glow. Camera holds steady, slight parallax.
>
> **Clip 6 (54–60 s) — orchestration.** Two calendar cards flip:
> "06:00 refresh corpus", "07:00 generate report". Then a phone-style
> notification card slides up, green checkmark, text
> "Market report ready · 5 supported · 15 citations in range". Fade to
> a dark end card with the title:
> "AI MARKET INTELLIGENCE & PRODUCT STRATEGY AGENT".
>
> **Negative prompt:** no human faces, no stock-office footage, no
> purple-pink gradient neon, no lens flares, no illegible gibberish
> text, no logos, no watermarks.
>
> **Technical:** 16:9, 24 fps, cinematic, sharp focus on text,
> consistent lighting across clips.

### Single-prompt version (30 s)

> Dark technical product explainer, 16:9, deep navy (#080b11) with
> glowing green (#4ade80) monospace accents and subtle grid texture.
> Glowing green data packets stream from labelled source nodes into a
> terminal; monospace lines type `posts 83 / chunks 115 / bge-small
> 384 dims`; five hexagonal agent nodes light up in sequence ending in
> an amber "CRITIC" node; a markdown report materialises with blue
> `[3, 11]` citation threads linking back to source cards; five metrics
> bars animate to 100/100/82/73/58 percent; a Slack notification card
> slides up reading "Market report ready". Slow deliberate camera moves,
> shallow depth of field, no faces, no stock footage, sharp legible
> monospace text.

### Voiceover script (~75 s, matches the reel in section A)

> Public competitor discussions are messy, repetitive, and full of
> contradictory opinions. Turning them into product strategy is a data
> problem — so I built the pipeline end to end.
>
> Collection first: fifteen queries against a public API, deduplicated
> and filtered by brand, stored in SQLite. It's idempotent, so it runs
> on a schedule without duplicating anything.
>
> Then retrieval. Posts are chunked, embedded locally with a small
> sentence-transformer — no API key, no cost — and indexed for cosine
> search. Every chunk remembers the post and URL it came from, because
> a claim without a source is just an opinion.
>
> On top of that, five agents. Research plans the queries. The
> competitor analyst works per brand. A customer analyst reads the
> community. Strategy ranks opportunities. And a critic fact-checks the
> draft claim by claim — supported, partial, or unsupported.
>
> The anti-hallucination defence is layered: the model only ever sees
> numbered evidence, every claim must cite it, the critic judges support,
> and then code — not a language model — verifies that each citation
> points at something that was actually retrieved.
>
> Quality is measured, not asserted. Hand-labelled questions score
> retrieval and citation quality, and because retrieval needs no LLM,
> the same eval runs in CI and fails the build on a regression.
>
> Finally, n8n owns the schedule: refresh the corpus at six, generate the
> report at seven, poll the FastAPI job until it's done, and post the
> verdicts and citation health to Slack. Python does the intelligence;
> n8n does the business workflow.
>
> The local model is small, and it shows — some claims are generic and
> the critic occasionally over-flags. That's why the numbers come from
> typed data in code rather than the model's own arithmetic, and why the
> architecture is provider-agnostic: swap the model, keep the pipeline.

---

## What to claim in an interview

Safe, specific, true:

- "Retrieval hit rate (recall@5) is 0.82 on hand-labelled questions; MRR is 1.0 — the top hit was relevant every time."
- "CI runs 95 tests plus a retrieval gate; a regression in chunking, embedding, or search fails the build."
- "Reports are async jobs because a full report is minutes of LLM work — polling beats a held-open HTTP connection and silent retries."
- "Every citation is validated against retrieved evidence, and verdict counts are code-counted, not LLM-counted."

Avoid: "it writes PRDs automatically" (on a 3B local model the prose is
generic), "deployed to production", or any accuracy figure that isn't in
`data/reports/eval_report_*.md`.

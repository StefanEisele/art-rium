# art-rium voice-rich — SEO-friendly series articles

This guide defines the voice for **rich, structured WordPress articles about a series of works**, distinct from the contemplative single-image voice in `voice.md`. Used by the article-rich writer in `services/ollama/client.py:write_rich_article`.

## Voice in one line
Confident, accessible, lightly personal essay-prose about a body of work — concrete observation grounded in the images, framed by the artist's intent and process. Written for readers who arrive via search and stay because the prose has substance.

## Perspective
- **Third-person about the work** is the default — *„Die Serie ist…"*, *„The series is…"*, *„这一系列…"*
- **First-person ("ich" / "I" / "我") is allowed** when describing the artist's process, intent, or practice — *„Ich entwickle weiterhin Arbeiten, die…"*, *„I continue to develop works that…"*. Use it where it carries information; don't sprinkle it.
- The reader is addressed in passing only — no "Folgen Sie!" / "Click here!" pushes inside the body. The closing italic footer (auto-injected) handles social CTA.

## Tone calibration
- Substantive over precious. The single-image voice is contemplative; this voice is informative.
- Warm but not chatty. Earnest but not academic.
- Concrete first, abstract second — every claim about the series should be traceable to recurring observations across the images (palette, recurring forms, materials, motion).
- Allow declarative claims about the artist's intent or process when they earn their place: *„Diese Serie untersucht…"*, *„This series explores…"*. Don't use these as filler.

## Confidence over hedging
Same rule as voice.md: direct verbs of observation. No *seems / appears / wirkt / scheint*. No precious adverbs (*etwas / fast / somewhat*).

## Allowed structural elements (NEW vs voice.md)
- **H2 headings** to organise the article into named sections. Headings are translated and provided by the orchestrator — the writer does NOT produce headings, only the prose under them.
- **H3 headings** for sub-items inside sections (e.g. individual works in "Verfügbare Werke" / "Available Works"). Also orchestrator-provided.
- **Bullet lists** ONLY in the "Technischer Ansatz" / "Technical Approach" section — 2–4 short items describing the production process (LoRAs, samplers, post-processing). Lists do not appear elsewhere.
- **Inline links** via the placeholder `[PARENT_SERIES]` — written literally where the parent series name belongs, the orchestrator substitutes an `<a>` tag. Used only in the "Teil einer größeren Praxis" / "Part of a Larger Practice" section.

## Forbidden structural elements
No headings outside the named slots. No bullet lists outside Technischer Ansatz. No bold / italic decoration in body prose. No raw URLs in the prose — links flow only through the placeholder mechanism.

## Section anatomy & word budgets

The orchestrator wraps the prose with H2 headings; you only write the prose paragraphs, the bullet items, and the placeholder text. Word counts below are for the **English** version; German is roughly the same, Chinese is character-count proportional (1 word ≈ 1.5–2 characters).

1. **Intro (60–100 w)** — opens with the series name in quotes (*„Recursive Identities"* / *"Recursive Identities"*) and a thesis sentence that states what the series investigates. One additional sentence anchors that thesis in a concrete recurring image element. No heading above this paragraph.

2. **Konzept / The Concept (100–160 w, 1–2 paragraphs)** — what fundamental question or framing the series engages. Stay grounded — every concept move should be traceable to something visible across the images. End on a claim that justifies the series existing as a series, not as separate pieces.

3. **Visuelle Sprache / Visual Language (100–160 w, 1–2 paragraphs)** — the formal vocabulary of the series: palette, recurring forms, textures, scale, light. Concrete description first, interpretation second. This section can mention parent-series colour codes (e.g. "from the ongoing Rust Orange and Dim Blue series") without using the placeholder — the placeholder is reserved for "Teil einer größeren Praxis".

4. **Technischer Ansatz / Technical Approach** — three sub-parts:
   - **Intro paragraph (40–70 w)**: name the tool stack (ComfyUI, Z-Image Turbo / Wan 2.2, custom LoRAs, etc.) and what the workflow balances (intent vs. unpredictability). One sentence sets up the bullet list.
   - **Bullet list (2–4 items, each 10–20 w)**: concrete process steps. Examples of good items: *"Custom LoRA training for consistent palette and organic textures"*, *"Iterative refinement across multiple generations"*, *"Post-processing in GIMP for final colour grading"*. No vague items like *"Using AI"*.
   - **Outro paragraph (30–60 w)**: pivot from the mechanics to what they enable — what kind of discoveries the workflow makes possible, not random variation but meaningful iteration.

5. **Teil einer größeren Praxis / Part of a Larger Practice (80–140 w, 1–2 paragraphs)** — appears ONLY when a parent series is provided. First paragraph mentions the parent series via the literal placeholder `[PARENT_SERIES]` (orchestrator substitutes the anchor). Frame the relationship concretely: shared palette, shared inquiry, what this series adds. Second paragraph (optional) connects to broader practice (recognitions, ongoing themes) without descending into self-promotion.

## Hard avoids (all languages)

Same as voice.md plus:
- AI-marketing buzzwords: *„harnessing"*, *"AI-generated masterpiece"*, *"next-generation"*, *"cutting-edge"*, *"groundbreaking"*, *„revolutionär"*, *„bahnbrechend"*. Cut.
- Hype superlatives: *atemberaubend / stunning / breathtaking / 令人窒息 / monumental / epic*. Cut.
- Marketing CTAs in body prose: *„Folgen Sie!"*, *"Don't miss…!"*, exclamation marks in general (max 0).
- Hashtags in prose. Hashtags belong on Instagram, not in the article body.
- Rhetorical questions as transitions (max 1 per article, only if it carries weight).
- "in einer Welt, in der …" / "in a world where …" / "在一个 …… 的世界里"
- Self-congratulatory framing about the work or the maker. The Lead-Creator-style framing in "Teil einer größeren Praxis" should be factual, not self-promotional ("nominated as Lead Creator" yes; "honoured to be recognised as Lead Creator" no).

## Titles (rich-article version)

- 2–6 words. Series names are usually English or German nouns; respect the artist's choice when `series_name` is provided as input — do NOT translate proper-noun series names across languages (the German article also says "Recursive Identities" if that's the series name).
- The article *title* (h1) can be different from the *series_name* — it can frame the article ("Recursive Identities — the Anatomy of a Pattern") or simply be the series name. Default: equal to `series_name` if provided.

## Per-language SEO notes

- **Excerpt** field — the WordPress post excerpt. ≤155 characters, in voice, contains the series name and one concrete detail. Used in archive views and as Yoast's meta-description fallback. No marketing language.
- **Meta description** field — the dedicated SEO snippet (Yoast). 130–155 characters, more search-active than `excerpt`: contains the focus keyphrase verbatim, names the series, surfaces what a searcher would find. Still no hype words, no exclamation, no AI-marketing buzz. Different sentence than `excerpt` — same substance, search-leaning phrasing.
- **Focus keyphrase** field — 2–4 lowercase words, the SEO target phrase a searcher would type. Localised per language (DE search ≠ EN search ≠ ZH search). Must appear verbatim in: the article `intro`, the `meta_description`, and at least one other body slot. Drawn from material/medium/theme/concept vocabulary. NEVER the bare series title alone.
- **Tags**: 3–6 lowercase, drawn from material/medium/theme vocabulary. Same set across languages (proper nouns may vary).

## Calibration sample — same imagined series ("Recursive Identities", parent "Rust Orange and Dim Blue")

**EN — Intro:**
> "Recursive Identities" is a series of generative-AI images that examines how a self holds together when it loops through endless transformation. The recurring forms — branching networks, cellular interiors, neural-looking webs — are familiar enough to read and strange enough to refuse a single name.

**EN — Konzept opening:**
> The premise is simple: identity is not a fixed object but a process that runs forwards. Each work in the series freezes one frame of that loop. What survives the loop — colour, gesture, structural rhyme — is what the series argues identity actually is.

**EN — Teil einer größeren Praxis opening:**
> "Recursive Identities" continues the inquiry that began in [PARENT_SERIES], the ongoing exploration of an iron-warm palette against a cold blue field. Where the earlier series mapped how a colour holds a room together, this one asks how it holds a self together.

(Use this calibration as the target. A reader should feel they have learned something concrete by the end of each section.)

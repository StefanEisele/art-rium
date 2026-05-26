# MODE: ESSAY

**Audience:** art critics, curators, fellow practicing artists, cultural press,
scholarly readers.

**Length:** 1,200–3,500 words across the entire piece (intro + movements).
Substantive. Do NOT pad — under-shoot before over-shoot.

**Structure:**
- Concrete opening (image / quote / material / moment / number).
- State the thesis by the end of paragraph 2.
- 3 to 5 **movements** that defend the thesis. Each movement has a short
  H2-style heading and 2–4 paragraphs of body. Headings are statements,
  not questions, not summaries. Each heading is 3–7 words.
- End on the hardest version of the claim. No conclusion paragraph.

**Reference exemplar (Goldstandard):**
> *"The title of my series is a deliberate provocation. It combines
> 'Handcrafted' — intention, labor, precision, value — with 'Slop' — mass,
> arbitrariness, trash, worthlessness. This semantic contradiction is not
> the problem of my work, but its thesis."*

**Citations:** real sources with publication dates and venue. Format inline:
*"Francesco D'Isa argued in The Philosophical Salon (December 2025) that…"*.
No fake citations. If a source name is uncertain, do NOT use it.

**SEO discipline (subordinate to voice, but enforced):**
- ONE primary keyphrase per essay. Short — **2 to 4 lowercase words**
  (e.g., *AI slop critique*, *autopoiesis as myth*, *materiality in generative art*).
  Never a sentence, never the full title.
- The `title` MUST **begin** with the exact `focus_keyphrase` (capitalised
  naturally). Title and keyphrase together — keyphrase first, then a colon
  or em-dash, then the rest.
  - GOOD title: *"Autopoiesis as Myth — On Influence in AI Art"* (keyphrase: `autopoiesis as myth`)
  - BAD title: *"Autopoiesis in der Kunst ist ein Mythos"* (keyphrase: `autopoiesis in der kunst ist ein mythos` — too long, sentence form).
- The exact `focus_keyphrase` MUST appear, verbatim:
  - in `intro[0]` (the very first paragraph)
  - in `meta_description`
  - in **at least one** `movements[].heading`
  - **3 or more times** across the full body (intro + movement bodies + closing combined).
- Localise the keyphrase per language — generate one for `en` and a
  language-natural one for `de`. They do NOT have to be word-for-word
  translations. Each language enforces its own placement independently.
- Never deform a sentence to hit the keyphrase. If a paragraph reads
  awkwardly with the keyphrase wedged in, restructure the paragraph or
  drop the occurrence — the density floor matters less than the voice.

---

## OUTPUT SHAPE

Return STRICT JSON with exactly this top-level shape:

```
{
  "en": { language block },
  "de": { language block }
}
```

Each language block contains EXACTLY these fields, in this order:

```
{
  "title":             str,              // ≤65 characters; STARTS with the focus_keyphrase verbatim; SEO-aware, voice-faithful, no clickbait
  "intro":             [str, ...],       // 2–3 opening paragraphs (concrete first, thesis by end of p2). Total 180–300 words. intro[0] MUST contain the focus_keyphrase verbatim.
  "movements": [                          // 3–5 entries; each movement ≤250 words total body (HARD CAP)
    {
      "heading": str,                     // 3–7 words; declarative; no questions. At LEAST ONE heading across all movements must contain the focus_keyphrase verbatim.
      "body":    [str, ...]               // 2 paragraphs at 80–120 words each (a third short paragraph is allowed only if total stays ≤250 words). If you have more to say, START A NEW MOVEMENT with its own H2 heading — do not extend.
    }
  ],
  "closing":           str,                // ONE paragraph (50–110 words). The hardest form of the thesis. NO summary, NO "in conclusion", NO exhortation.
  "excerpt":           str,                // ≤150 chars; voice-faithful one-sentence excerpt
  "meta_description":  str,                // 130–150 chars; Yoast SEO snippet; MUST contain the focus_keyphrase verbatim
  "focus_keyphrase":   str,                // 2–4 lowercase words; localised per language; MUST be a prefix of `title` (case-insensitive); appears verbatim in intro[0], meta_description, ≥1 movement heading, and ≥3 times total across the body
  "tags":              [str, ...],         // 3–6 short lowercase tags, in the language of the post
  "og_image_idea":     str                 // ONE sentence describing the ideal lead/social-share image
}
```

---

## VIDEO PLACEHOLDERS (only when a Video manifest is provided)

If the user message includes a "Video manifest" block listing `[VIDEO_1]`,
`[VIDEO_2]`, … each token MUST appear EXACTLY ONCE in the prose, as a
standalone string entry inside one of the paragraph arrays (`intro`
strings, a movement's `body`, or `closing`). Place each token where the
prose naturally introduces, pauses on, or extends what the video shows.
Do NOT quote the token in surrounding text ("see [VIDEO_1] below"); just
emit the bare token as its own paragraph entry. Use ALL videos. Same
placement in EN and DE. The renderer replaces each token with a YouTube
embed at that position.

---

## HARD RULES

- NO hedging verbs (*seems / appears / wirkt / scheint*).
- NO precious adverbs (*etwas / fast / somewhat / slightly*).
- NO AI-marketing buzzwords (*harnessing / cutting-edge / revolutionary /
  stunning / bahnbrechend / atemberaubend*).
- NO exclamation marks. NO hashtags. NO bold/italic markdown.
- NO embedded headings or bullet lists inside `intro`, `body`, or `closing`
  paragraphs — those are plain prose.
- NO raw URLs anywhere.
- NO self-promotion phrases (*"As a nominated Lead Creator at…"*,
  *"As a Verified Artist on Singulart…"*). Awards belong in the CV.
- NO boilerplate openings or closings (see Universal Prohibitions).
- See the universal voice guide for sentence-start variation and transition
  word rules — those apply to every paragraph here.

---

## TONE TEST (silent)

Could this paragraph appear in a press release for a venture-backed AI
startup? If yes, rewrite. Could a Kunstforum or Frieze contributor publish
this with my byline? If no, rewrite.

## SELF-AUDIT BEFORE EMITTING (silent)

Before you emit JSON, run these counts on each language block:

- For each `movements[i]`, sum the word count of `body`. The cap is
  **250 words per movement** (HARD). If any movement is >250 words, you
  MUST split it into two movements with separate H2 headings before
  emitting. Most movements should land at 180–230 words.
- Pick 10 consecutive sentences from the combined body and count
  transitions. <3 hits? Rewrite some sentences to use *aber / doch /
  deshalb / weil / indem / sodass* (DE) or *but / yet / so / because /
  while / although* (EN).
- Scan opening words sentence-by-sentence. Any three-in-a-row with the
  same opener? Rewrite one.

These checks are not optional. The validator on the receiving side will
log warnings for each violation and the post will be flagged for review.

Return ONLY the JSON object — no prose around it, no code fences, no commentary.

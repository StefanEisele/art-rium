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

**SEO discipline (subordinate to voice):** one primary long-tail keyword per
essay (e.g., *"AI slop critique"*, *"materiality in generative art"*) — never
deform a sentence to hit it.

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
  "title":             str,              // ≤65 characters; SEO-aware, voice-faithful, no clickbait
  "intro":             [str, ...],       // 2–3 opening paragraphs (concrete first, thesis by end of p2). Total 180–300 words.
  "movements": [                          // 3–5 entries
    {
      "heading": str,                     // 3–7 words; declarative; no questions
      "body":    [str, ...]               // 2–4 paragraphs; each paragraph plain prose, 80–180 words
    }
  ],
  "closing":           str,                // ONE paragraph (50–110 words). The hardest form of the thesis. NO summary, NO "in conclusion", NO exhortation.
  "excerpt":           str,                // ≤155 chars; voice-faithful one-sentence excerpt
  "meta_description":  str,                // 130–155 chars; Yoast SEO snippet; contains focus_keyphrase verbatim
  "focus_keyphrase":   str,                // 2–5 lowercase words; localised per language; appears verbatim in intro AND meta_description
  "tags":              [str, ...],         // 3–6 short lowercase tags, in the language of the post
  "og_image_idea":     str                 // ONE sentence describing the ideal lead/social-share image
}
```

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

---

## TONE TEST (silent)

Could this paragraph appear in a press release for a venture-backed AI
startup? If yes, rewrite. Could a Kunstforum or Frieze contributor publish
this with my byline? If no, rewrite.

Return ONLY the JSON object — no prose around it, no code fences, no commentary.

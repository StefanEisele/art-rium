# MODE: WORK / SERIES (New Artwork)

**Audience:** collectors, gallerists, Singulart visitors, Instagram-to-site
funnel arrivals.

**Length:** 250–600 words across the whole post. The work carries the post;
the text frames it.

**Structure:**
1. **Opening line** — ONE sentence that names the work's central tension or
   material claim. Concrete, declarative, no hedging.
2. **2–4 paragraphs** of curatorial-style description: what the work is,
   what it is made of (technically: model, post-production, output medium),
   what idea it carries.
3. **Optional 'Available Works' section** — if singulart_links are
   provided, the orchestrator renders product cards; you write a SINGLE
   12–25-word introductory sentence in `available_works_intro`.
4. **Optional 'Larger Practice' paragraph** — if parent_series is
   provided, one paragraph (60–110 words) that situates this series
   within the parent body of work, using the literal placeholder
   `[PARENT_SERIES]` where the parent name belongs (orchestrator
   substitutes a hyperlink).
5. **Metadata block** rendered by the orchestrator at the bottom (year,
   medium, dimensions, edition, status, Singulart link).

**Voice:** Curatorial. Third-person possible for the work itself
("*Entropy Egg #03* refuses…"), first-person for the maker's note
("I built this around…"). NEVER write *"stunning"*, *"breathtaking"*,
*"powerful"*, *"masterful"*. Show, don't praise.

**Avoid Singulart-as-promotion:** the Singulart link is metadata, not a
CTA in the body. Write the work seriously; let the link be a quiet
final line in the orchestrator-rendered metadata block.

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
  "title":                   str,             // 2–6 words; equals the series name when one is provided
  "opening_line":            str,             // ONE sentence; the central tension / material claim. 12–28 words.
  "body":                    [str, ...],      // 2–4 paragraphs; total 180–450 words. Plain prose, no headings, no bullets.
  "available_works_intro":   str | null,      // SINGLE 12–25-word sentence introducing the buying section. ONLY when has_singulart=true; otherwise null.
  "larger_practice":         [str, ...] | null, // 1 paragraph (60–110 words) containing the literal [PARENT_SERIES] placeholder. ONLY when has_parent_series=true; otherwise null.
  "metadata":                {
    "year":      str,                          // "2026" or "2025–2026"
    "medium":    str,                          // e.g. "AI-generated image (ComfyUI / Z-Image Turbo), post-production Photoshop, archival pigment print"
    "dimensions": str,                         // e.g. "120 × 80 cm" or "120 × 80 cm | 4096 × 2731 px" or "Digital only"
    "edition":   str,                          // e.g. "Edition of 5 + 2 AP" or "Open edition" or "Unique"
    "status":    str                           // "Available" | "Reserved" | "Sold" | "On view" — pick one based on user_notes/context
  },
  "excerpt":                 str,              // ≤155 chars; voice-faithful one-sentence excerpt naming the series and one concrete detail
  "meta_description":        str,              // 130–155 chars; Yoast SEO snippet; contains focus_keyphrase verbatim
  "focus_keyphrase":         str,              // 2–4 lowercase words; localised per language; appears verbatim in body AND meta_description. NEVER the bare series title alone.
  "tags":                    [str, ...],       // 3–6 short lowercase tags, in the language of the post
  "og_image_idea":           str               // ONE sentence describing the ideal lead/social-share image (typically the work itself, framed)
}
```

---

## HARD RULES

- NO hedging verbs (*seems / appears / wirkt / scheint*).
- NO praise-superlatives (*stunning / breathtaking / powerful / masterful /
  atemberaubend / beeindruckend*).
- NO AI-marketing buzzwords (see Universal Prohibitions).
- NO exclamation marks. NO hashtags. NO bold/italic markdown.
- NO embedded headings or bullet lists inside `body` paragraphs — plain prose.
- NO raw URLs anywhere — links flow only through the `[PARENT_SERIES]`
  placeholder when a parent series is provided.
- NO self-promotion phrases. Awards belong in the CV.
- NO Singulart-as-CTA inside the body — the buying option is metadata.

---

## METADATA-BLOCK GUIDANCE

The `metadata` object is your best inference from the images and any
`user_notes`. Use sensible defaults if a field cannot be determined:
- `year` → current year if unstated.
- `medium` → "AI-generated image (ComfyUI), digital print" if unstated.
- `dimensions` → "Digital only" if the series has no print format yet.
- `edition` → "Open edition" if unstated.
- `status` → "Available" if singulart_links are provided; "Not for sale" otherwise.

The metadata fields are the SAME across en/de — output them in English
in BOTH language blocks (the orchestrator renders a single bilingual-safe
metadata block). The orchestrator localises the labels ("Year:", "Jahr:")
but uses your verbatim VALUES.

Return ONLY the JSON object — no prose around it, no code fences, no commentary.

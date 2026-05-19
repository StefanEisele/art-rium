You are writing a single blog post about the artwork shown in the image, in three sibling languages: German (de), English (en), and 简体中文 (zh). Generate all three in one pass so the voice stays aligned — same structure, same key images observed in the artwork, same mood, idiomatic in each language (NEVER word-for-word translation).

Follow the art-rium voice guide above strictly. Every poetic move must be traceable to a concrete element actually visible in the image.

==========================================================================
VOICE RULES — non-negotiable. These are the most-violated rules in past drafts. Read them BEFORE you start writing each paragraph and check the paragraph against them BEFORE moving on.
==========================================================================

RULE 1 — NO hedging verbs.
  Replace "seems / appears / feels like / wirkt / scheint / fühlt sich an wie" with direct verbs of observation ("is / holds / rests / falls / ist / hält / liegt / fällt"). This applies to EVERY paragraph, not just the entry.
    ✗ "a silence that seems heavy to carry"      → ✓ "the silence holds"
    ✗ "the light seems to come from a source"    → ✓ "the light comes from above" (or just describe what's there)
    ✗ "scheint der Träger gebeugt"               → ✓ "der Träger ist gebeugt"
    ✗ "die Atmosphäre wirkt schwer"              → ✓ "die Atmosphäre ist schwer" (or cut — the image already shows it)
    ✗ "what it seems / was sie zu sein scheint"  → ✓ cut entirely; the phrase carries no observation

RULE 2 — NO Fazit closes. Paragraph 4 has SPECIFIC bans:
  - No sentence in p4 may begin with "It is" / "It's" / "Es ist" / "Sie ist" / "这是" / "它是" / "那是".
  - No p4 sentence may contain "the answer to" / "a symbol of" / "a metaphor for" / "what happens when" / "die Antwort auf" / "ein Symbol für" / "eine Metapher für" / "was passiert, wenn".
  - No anaphora (do not start three sentences in a row with the same subject + verb pattern).
  - No meta-commentary about looking at the image ("a moment that does not need to be explained, only seen" — forbidden).
  Instead, paragraph 4 ends on ONE concrete observation drawn from the image that has not already been named in paragraphs 1–3 — a colour shifting, a shadow's edge, a texture, a fragment of light. Two sentences max. Then stop.
    ✗ "It is an image of the journey, not of the destination. It is an image of the burden we carry…"
    ✗ "Es ist ein Bild von der Reise, nicht von dem Ziel."
    ✗ "It is a moment that does not need to be explained, but only seen. The cloud is a mirror of the silence that rules the room. It is the answer to the question of what happens when we try to carry the impossible."
    ✓ "The pause holds. It does not go empty, because the book is still open and the light keeps shifting. Muted ochre, a broken green. The colours hold the room together without closing it."
    ✓ "Die Pause hält sich. Sie wird nicht leer, weil das Buch noch offen ist und das Licht sich verschiebt. Gedämpftes Ocker, ein gebrochenes Grün. Die Farben halten den Raum zusammen, ohne ihn zu schließen."

RULE 3 — NO precious adverbs ("etwas / ein wenig / fast / beinahe / somewhat / a little / slightly"). Cut them. If the observation needs one, the noun or verb is wrong — rewrite that instead.

RULE 4 — NO interior projection beyond what's visible. If the image does not show a decision, a longing, a memory — do not write one. Stay in the room. Series/project context (paragraph 3) is the ONLY place where slightly broader framing is allowed, and even there, frame it concretely (materials, recurring forms, palette across the series), not psychologically.

RULE 5 — Each language NATIVELY. The German must read as German prose; the English as English prose; the Chinese as 中文书面语. Same observations, same structure, idiomatic phrasing in each. NEVER lift constructions across languages.
  - English-specific: do not lift German participle constructions or compound nouns. "The man is not stuffed" was a translation error from "gestopft" — write English the model would write if German didn't exist.
  - Chinese-specific: 中文 must use 中文 vocabulary throughout — no English words left untranslated (e.g. "muted" → "暗淡 / 沉静"). Check negations carefully: 关乎 ("is about") and 无关 ("is not about") are opposites; do not invert the meaning.

==========================================================================
OUTPUT SHAPE
==========================================================================

Return STRICT JSON with exactly this shape:
{
  "de": {"title": str, "body_md": str, "excerpt": str, "tags": [str, ...]},
  "en": {"title": str, "body_md": str, "excerpt": str, "tags": [str, ...]},
  "zh": {"title": str, "body_md": str, "excerpt": str, "tags": [str, ...]}
}

For each language:
  title    — 1 to 7 words, evocative, per the voice guide's "Titles" rules above. No trailing period. No quote marks.
  body_md  — the article body. Length: 260–320 words; 240 is a hard floor. Plain Markdown, paragraph breaks only (no headings, no bold/italic, no links, no bullets). Write FOUR distinct paragraphs separated by blank lines, in this order with these per-paragraph word budgets: (1) concrete entry, 40–60 words — one specific observation, not a thesis; (2) reflection/mood, 100–140 words — anchor every move in concrete detail from the image; (3) series/project context, 50–80 words — how this piece sits in a wider body of work; (4) quiet close, 20–40 words — see Rule 2 above; one or two sentences ending on a concrete observation. All four paragraphs are required.
  excerpt  — one sentence, ≤155 characters, in the article's voice. Used as the meta description; concrete imagery, no marketing language.
  tags     — 3 to 6 short lowercase tags relevant to the artwork. Single words or short phrases. Same set across languages where it makes sense (proper nouns may differ).

==========================================================================
FINAL CHECKLIST — verify before emitting JSON.
==========================================================================
1. Each body_md is 260–320 words (240 hard floor).
2. Each body_md has exactly 4 paragraphs separated by blank lines.
3. NO sentence anywhere in paragraph 4 starts with "It is" / "It's" / "Es ist" / "Sie ist" / "这是" / "它是" / "那是".
4. NO sentence anywhere in paragraph 4 contains "a decision" / "an acceptance" / "a longing" / "a memory" / "eine Entscheidung" / "eine Annahme" — these project interiority not visible in the image.

Return ONLY the JSON object — no prose around it, no code fences, no commentary.

# Iterate — VLM critic + LLM director for Z-Image Turbo

One idea in, N rounds of render → look → rewrite, one strong image out. The
VLM looks at what was actually rendered; the LLM decides what to change in
the prompt; the loop keeps only what got better.

This file is two things at once:

- **Part 1 — the cycle.** How the program runs the loop, and why it is built
  this way rather than as a plain "describe, rewrite, repeat" chain. Read by
  people, never sent to a model.
- **Part 2 — the material the models get.** Shared blocks (taste, rubric,
  operators, prompt rules, anchors) and the four role prompts that embed
  them (brief, critic, duel, editor). The program reads these sections by heading.

How the program reads Part 2: every section headed `## BLOCK: NAME`,
`## PROMPT: NAME` or `## TEMPLATE: NAME` runs to the next `## ` heading. A
`{NAME}` placeholder inside a prompt is replaced by the block of that name;
`{lowercase}` placeholders inside a template are filled by the program per
call. Same convention as `zimage-styles.md` / `zimage-enhancer.md`
(`{STYLE_BLOCK}`): the wording can be tuned here without a code change.

---

# Part 1 — The cycle

## Why not the obvious loop

The obvious loop — render, have a VLM describe it, have an LLM "improve" the
prompt, repeat — fails in six predictable ways. Each rule below answers one.

| Failure of the naive loop | What this design does instead |
|---|---|
| **Drift.** Each rewrite moves a little; after ten rounds the image is about something else. | A frozen **brief** (subject, twist, must-keeps) is written once in round 0. Every round is judged against it, never against the previous prompt. |
| **Prompt bloat.** LLMs improve by adding. A 30-word formula grows into 250 words of adjectives and loses what made it work. | The prompt is split into a locked **anchor** and an editable **core**. The editor may only touch the core, under a length budget per mode. |
| **Seed noise.** A new seed changes the image more than a better prompt does, so the loop chases luck. | **One fixed seed** for all rounds, so a change in the picture is a change caused by the prompt. A **seed sweep** at the end checks the winner holds up. |
| **Score inflation, noise and position bias.** VLMs praise; their absolute scores wander; and asked "which of two images is better", they pick by position. Measured: Gemma 3 27B chose "image 2" in *both* orders of the same pair, qwen2.5vl answered "same" in both. | The critic scores **the candidate alone** against a rubric with anchored levels and is told to be strict. A pairwise **duel** is only used for close calls, and only counts when it gives the same winner **in both orders**. |
| **Oscillation.** A→B→A: the editor undoes last round's change, then redoes it. | **Elitism**: the editor always edits the *best* core, sees the **history** of what was tried and whether it was accepted, and may not repeat a rejected move unchanged. |
| **Plateau.** Small edits stop helping; the loop polishes a dead end. | A **phase schedule** (explore → refine → polish) and a **plateau jolt** that forces a structurally different staging. |

## Roles

| Role | Model (default) | Sees | Writes |
|---|---|---|---|
| **Brief** (round 0) | editor model | the idea, optional source prompt | the frozen brief + the first core |
| **Critic** | VLM | the candidate image, the brief | observation, must-keep check, scores, defects, keeps |
| **Duel** (close calls, sweep) | VLM | two images, the brief — run twice, order swapped | the winner, or a tie |
| **Editor** | LLM | brief, anchor, best core + its report, latest report, history, phase | one diagnosis, one operator, the new core |

The critic never suggests prompt wording and the editor never judges an image
it has not been told about. A critic that writes prompts starts grading its
own suggestions; an editor that guesses at pictures invents defects.

## One run

```
idea ──► BRIEF ──► brief (frozen) + core₁
                       │
   ┌───────────────────┘
   ▼
 round i = 1..N
   prompt_i  = anchor.head ⊕ core_i ⊕ anchor.tail
   image_i   = render(prompt_i, seed S, size, LoRA stack)        ← ComfyUI
   report_i  = CRITIC(image_i, brief)                            ← VLM
   if close call:  duel = DUEL(image_i, best.image) both orders  ← VLM
   if accept(report_i, duel, best):  best ← (core_i, image_i, report_i)
   if i < N:  core_{i+1} = EDITOR(brief, anchor, best, report_i, history, phase)
   │
   ▼
 sweep: render best.prompt with 3 more seeds → DUEL each vs best, both orders
 save:  every round as one gallery series, best marked, log sidecar
```

## Accepting a candidate

Weighted total = `0.30·intent + 0.25·surreal + 0.20·signature +
0.15·composition + 0.10·craft`.

Round 1 becomes the best automatically. After that a candidate is
**eligible** only if the gates pass: `intent ≥ 5`, `craft ≥ 4`, and no
`must_keep` item is `"missing"`. A sharper picture of the wrong idea, or a
striking picture with a broken face, never becomes the best.

An eligible candidate replaces the best when:

- its total is **≥ 0.6** above the best's — clear win, no duel; or
- its total is within **±0.6** of the best's — a close call: run the DUEL
  twice, candidate first then best first. Replace only if the candidate wins
  **both** orders. Split or tie → the best stays.

Below that it is rejected without a duel. The duel costs two VLM calls, so it
runs only where a score difference cannot be trusted.

Why not compare every round: the measured position bias makes a one-order
comparison close to a coin with a known tilt, and running both orders every
round doubles the slowest step of the loop.

## Phases

| Phase | Rounds (of N) | The editor may | Purpose |
|---|---|---|---|
| explore | first 35 % | one bold operator per round; change staging, scale, camera | find the strongest version of the idea |
| refine | middle | one small operator, or CLARIFY | make it read clearly |
| polish | last 15 % (≥ 1 round) | CLARIFY or REDUCE only | fix misreadings and craft |

For N ≤ 5: explore 2, refine N−3, polish 1. For N = 1 the loop is just
brief → render → critic; the report is still saved.

**Plateau:** no accepted candidate for `max(3, round(N/5))` rounds → the next
editor call gets `plateau: true` and must use an operator it has not used in
the last four rounds and change the composition substantially, keeping every
must-keep. The jolt is allowed once per phase.

**Early stop (optional):** in polish, two rejected candidates in a row end
the run early. Off by default — the user asked for N rounds.

## Checks the program makes on every core

The prompt rules are instructions; a model can break them. Measured: the 4 B
prompt model, with these exact rules in front of it, still wrote 44 words in
a 40-word mode, a colour word into a formula core, and a core that dropped a
must-keep. So every core the editor (or the brief) returns is checked in
code before anything is rendered:

| Check | Rule |
|---|---|
| length | formula 8–40 words, prose 120–220 words |
| must-keeps | every `must_keep` noun phrase appears (case-insensitive; head noun is enough: "lighthouse" for "a lighthouse") |
| formula hygiene | no colour or capture words: rust-orange, orange, blue, blue-grey, teal, grey, copper, amber, red, gold, VHS, grain, film (as a capture word) — "rust" alone is allowed as a material |
| anchor | the core does not contain the anchor head or tail, or trigger words |
| language | no umlauts or ß outside names (catches most German leftovers, not all) |
| negation | no "no …", "without …", "not …" phrases |

A failed check sends the core back to the editor **once**, with only the
broken rule named ("Your core has 44 words; the limit is 40. Rewrite it to
fix only that."). Failing twice, the round is skipped: the best stays, and the
history records `rejected (rule: length)`. A rule violation must never cost a
render.

## Seeds

- One seed **S** for every round: the source image's seed when iterating from
  a gallery picture, otherwise random once at the start.
- Fixed seed risks a prompt that works for S only. Hence the **sweep**: the
  best prompt is rendered with three more seeds; each is scored by the
  CRITIC, and the strongest one duels the best-of-rounds image in both
  orders. The final image is the duel winner (the best stays on a split), and
  the report says how many of the four seeds passed the gates — the measure
  of whether the *prompt* is good, not just one render.

## Two prompt modes

The artist's best images came from two different prompt families, and the
loop must not turn one into the other.

- **Formula mode** — LoRA stack active. The style lives in the LoRAs and a
  short proven anchor; the core is 8–40 words: camera distance, subject, the
  twist, setting. Bloat here *removes* style, because every extra clause
  outweighs the LoRAs. Default for the signature look.
- **Prose mode** — no LoRA, or the artist chooses it. The core is one
  paragraph of 120–220 words of material physics, light and capture, in the
  order the rules below give. This is the family of "Liquid Architecture",
  "The Molten Vein", "Enlightened Unease".

The mode is chosen at the start and never changes during a run.

## Outputs

- Every round is ingested as a normal gallery image (prompt, seed, LoRAs,
  size — the usual reconstruction metadata) and put, in order, into **one
  image series** named after the idea. The series is the run's history and
  can be handed on to video/articles/Instagram like any other.
- Each image's `notes` carries `round i/N · operator · accepted|rejected · total`.
- The best image is marked in the series and opened at the end. The loop never
  sets a star rating: ratings are the artist's.
- A sidecar `storage/iterations/<run_id>.json` keeps everything: brief,
  anchor, every core, every report, every edit, timings. This is the material
  for learning which operators actually win (see below).

## Models and hardware

Measured 2026-10-08 with these prompts. Critic and duel numbers come from
the calibration below (20 critic calls and 20 duel calls per model); brief and
editor from a formula-mode round 2 on the lighthouse idea.

| Model | Role | Time | Verdict |
|---|---|---|---|
| `qwen3-vl:8b-instruct` (8.8 B, Q4_K_M, 7.5 GB on the 4060 Ti alone) | critic + duel | 8 s / image, 6 s / duel order | **Critic.** Best separation of the artist's five-star pictures (total Δ +1.33, intent Δ +2.5), stable to ±0.5, catches "the person is missing" (intent 1). Duel: not positional, but unreliable — see calibration. |
| `hf.co/unsloth/gemma-3-27b-it-GGUF:Q3_K_M` | critic + duel | 41 s / image, 23 s / duel order | Not the critic: scores compressed into 4–7 (total Δ +0.20), and in the duel it picked image A in 8 of 10 pairs — pure position. |
| same Gemma | brief + editor | 19–21 s (text) | **Editor.** Translates and keeps the must-keeps ("lighthouse"), one taste-true change ("shrank the keeper to a tiny figure at the base"), within budget, no colour words. |
| `qwen3-vl:8b-instruct` | brief + editor | 5 s | Not the editor: turned "Leuchtturm" into "tower", put "rust-orange paint" and "mineral-blue stone" into a formula core and stacked a second surreal idea on the first. |
| `qwen2.5vl:latest` (7 B) | critic | 8–18 s | Too weak: "same" for a pair in both orders, flat scores, nested a key with a typo. |
| `qwen3-vl:latest` | critic | — | **Unusable:** the *thinking* build; ignores `think: false`, writes everything into `thinking`, returns empty content. |
| `kamekichi128/qwen3-4b-instruct-2507` | brief + editor | 4–6 s | Too weak: German words in the core, dropped a must-keep, colour word in a formula core, body horror. |

Defaults (core/config.py): `ollama_iterate_critic_model = qwen3-vl:8b-instruct`,
editor = `ollama_llm_model` (Gemma). The page offers Gemma as the stricter,
slower critic.

VRAM: Gemma spreads over both cards (~8.9 GB on the 4060 Ti, ~3.9 GB on the
2070), qwen3-vl sits on the 4060 Ti alone, and neither can share the 4060 Ti
with Z-Image. A round therefore hands the card over twice:

1. **think** — editor (Gemma) writes the next core;
2. **render** — every Ollama model unloaded *and confirmed gone*
   (`wait_until_unloaded`), ComfyUI renders;
3. **think** — ComfyUI `/free` plus a 3 s settle, then critic (→ duel) on
   qwen3-vl. Two different models never share the cards: the other one is
   unloaded and confirmed gone before a load.

The confirmation is not optional. Loading Gemma a moment after qwen3-vl's
unload request crashed the runner (`GGML_ASSERT(buffer)`): Ollama's unload
returns before the VRAM is free.

Call settings that are not optional:
- **`num_ctx` = 16384, always set.** Without it Ollama sized qwen3-vl at
  34.6 GB. At 8192 it decides Gemma's 12.8 GB of weights fit the empty
  4060 Ti alone and then fails to allocate the vision projector on top
  (cudaMalloc OOM → `GGML_ASSERT(buffer)`); at 16384 the estimate no longer
  fits one card, it splits over both, and the load works.
- `think: false`, `format: json`; temperatures brief 0.6, critic 0.2,
  duel 0.1, editor 0.7 (explore) → 0.5 (refine) → 0.3 (polish).
- Images to the VLM at 768 px long edge, JPEG q85.
- **Validate every JSON reply** against its schema: coerce scores to ints
  1–10, lift a misplaced nested key, and retry once with "Reply with the
  exact keys only" when a key is missing. A malformed report must never
  become the best.

## Calibration before trusting the loop

The critic prompt is the part that decides everything, so it gets an
acceptance test against the artist's own ratings. The test set is 10 pairs,
each a five-star picture and an unrated sibling rendered from **the exact same
prompt** (only the seed differs; nothing published or in a series), with a
neutral brief written from that prompt. The script is reproducible from the
gallery: one pair per prompt family, families with both a five-star and an
unrated render.

1. **Taste test.** Score every picture alone. Pass: the five-star median
   `signature` and `surreal` are both ≥ 1.5 above the unrated median.
2. **Duel test.** Each pair in both orders. Pass: the five-star picture wins
   both orders in ≥ 6 pairs, the unrated one in ≤ 1.
3. **Stability test.** One picture scored three times. Pass: every criterion
   within ±1.

Result 2026-10-08:

| | qwen3-vl 8B | Gemma 27B |
|---|---|---|
| taste: total Δ (paired: five-star higher) | +1.33 (7/10) | +0.20 (5/10) |
| taste: intent / surreal / signature Δ | +2.5 / −0.5 / +0.5 | +0.5 / 0 / 0 |
| duel: five-star both orders / unrated both / split | 2 / 4 / 4 | 0 / 2 / 8 |
| stability spread (max−min) | ≤ 1 | ≤ 2 |
| pass taste / duel / stability | no / no / yes | no / no / yes |

How to read it. Neither model passes on taste or duel — but this is the
hardest version of the question: seed siblings of one prompt, where the
artist's pick often comes down to nuance (looked at by hand, half the pairs
are near-twins). On the clear pairs — the sibling with no person at all, the
filament without its shattered form, the weaker paint figuration — qwen3-vl
agreed with the artist in its scores. The loop compares *different prompts*
at one seed, which differ far more than seed siblings do, so the critic's
real job is closer to the clear pairs.

What follows for the design:
- Intent is the reliable signal (Δ +2.5) — which is why intent carries the
  largest weight and gates acceptance.
- The duel is the weak link: qwen3-vl gave the unrated sibling a consistent
  win in 4 of 10 pairs. A close call therefore only replaces the best on a
  win in both orders, and a split keeps the best — a wrong "accept" inside
  the ±0.6 band costs little because the scores already called it even.
- Signature/surreal taste is where the critic is weakest. First lever to try:
  the reference image ("Later, not v1"), then rubric wording. Re-run this
  calibration after any edit of BLOCK: TASTE or RUBRIC.

## Parameters

| Parameter | Default | Range | Note |
|---|---|---|---|
| iterations N | 8 | 1–30 | the user's choice |
| mode | formula | formula / prose | fixed per run |
| anchor preset | signature | see BLOCK: ANCHORS | head/tail locked |
| LoRA stack | from preset | — | not edited by the loop (v1) |
| size | 1080×1920 | Z-Image sizes | |
| seed S | source or random | — | fixed for all rounds |
| sweep seeds | 3 | 0–6 | 0 = skip |
| early stop | off | on/off | |
| critic | qwen3-vl 8B | qwen3-vl 8B / Gemma 27B | per run, on the page |

## Later, not v1

- **Candidates per round K > 1.** Render K edits per round, keep the critic's
  favourite. Multiplies the render cost by K.
- **Seed-fragility check** every 5th round: render the current best with a
  second seed; if its intent drops by ≥ 3, tell the critic "the best prompt
  is seed-fragile". The sweep at the end covers the same question once.

- **Parameter edits.** Letting the editor move LoRA strengths (±0.1 within
  0.05–0.8) or the detail amount. Powerful — the winning stacks differ only
  in those numbers — but never in the same round as a prompt edit, or no one
  can tell which change did it.
- **Operator priors.** After a few dozen runs, the sidecars show which
  operators get accepted (REDUCE is likely high). Feed the acceptance rates to
  the editor as a soft preference.
- **Reference images.** Give the critic one of the artist's five-star images
  closest to the brief as a third image ("this is the bar"). Costs context;
  try it if the taste test is borderline.
- **Human checkpoint.** Pause every k rounds for a thumbs-up/down that
  overrides the verdict.

---

# Part 2 — What the models get

## BLOCK: TASTE

THE ARTIST'S SIGNATURE — distilled from their highest-rated and published
works. This is the target look. Judge and write toward it.

The core move: ONE impossible thing, presented as calm documentary fact. The
picture looks photographed, never illustrated; the absurdity is not
explained, dramatized or decorated. It reads like "documentary evidence of
something that cannot be documented".

Recurring worlds (any one is enough; never several at once):
- The dwarfed figure: one small human — or a line, ring, grid or tower of
  identical humans — in a vast emptiness: desert, salt flat, sea of clouds,
  endless corridor, open sky. Extreme long shot; the figure is a small part
  of the frame.
- Material physics: acrylic or oil paint with mass and viscosity, pouring,
  ribboning, birthing or swallowing classical figures; a single rust-orange
  filament under gravity in a mineral-blue field; rust growing over a leaf,
  a cell, paper; kintsugi seams; translucent membranes lit from behind; veins.
- Calm body anomalies: duplicated or mirrored bodies, overlapping faces, an
  oversized head in a plain apartment, bodies stacked into a tower or tangled
  in mid-air, figures floating face-down above clouds.
- Liminal institutional space: an empty clean room, a corridor with heavy
  curtains, an apartment with one window, an operating theatre where surgeons
  tend a web of strings, a gallery with nothing on the walls.
- Architecture that behaves wrong: arms reaching out of corridor walls,
  spiked paper cities, mirrored rooms in endless regress, glass cubes
  receding to a vanishing point, a stairway made of identical people.

Palette: rust orange — iron oxide, burnt copper — against dim blue — slate,
mineral blue, overcast haze — with bone or cream white and charcoal. The warm
element is scarce: often the only warm thing in a cold field. Everything else
desaturated.

Light: either aggressively overexposed, highlights blooming into white, or
dim haze cut by one hard source — a backlight, a beam, a slit of light, a
glow from inside translucent material.

Capture: analog and documentary — film grain, VHS or CRT still, archival
photograph; telephoto compression or panoramic widescreen on the wide shots,
shallow depth of field on the macros.

Mood in the artist's own words: psychologically charged, contemplative
absurdity, quiet dread, alienation, clinical, entropic beauty.

Oddcore / weirdcore, as this artist means it: uncanny normality, wrong
scale, liminal emptiness, repetition, things that should not be there
behaving as if they belong, low-fi capture. Not internet-meme weirdcore, not
eyeball-and-creepypasta clichés, not horror.

Anti-signature — these pull an image away from the artist:
- fantasy kitsch: magic glow, sparkles, castles, creatures;
- neon cyberpunk, chrome sci-fi, glossy CGI, plastic skin;
- HDR oversaturation, rainbow palettes, teal-orange blockbuster grading;
- posed, smiling, model-like people looking at the camera;
- clutter with no focal point; two or more surreal ideas competing;
- body horror and gore: peeling, wounded or rotting skin, exposed flesh.
  Rust, veins and decay belong on materials — a leaf, a cell, paper, steel,
  a statue — never on living skin;
- cuteness, text, logos, watermarks.

## BLOCK: RUBRIC

Five criteria, each an integer 1–10. Scores are relative to the artist's
best works, not to average AI images: 7 is good, 9–10 is rare, most first
renders land at 4–6. Use the full range.

intent — does the image show the brief's subject AND its twist?
  10  every must-keep present; the twist is the first thing the eye reads
   7  subject and twist present; one must-keep weak or misplaced
   4  subject present, twist missing or misread as something else
   1  unrelated to the brief

surreal — strength of the oddcore / weirdcore effect
  10  one impossible thing rendered as photographic fact; calm, uncanny,
      you look twice
   7  clearly surreal, but slightly illustrative, explained or decorative
   4  strange decoration on an ordinary picture, or stock fantasy
   1  an ordinary photograph

signature — fit to the artist's look (BLOCK: TASTE)
  10  could hang between the artist's best works: palette discipline,
      analog capture, scale and isolation or material physics
   7  recognisably the artist's world, one element off (palette drift,
      digital sheen, too busy)
   4  generic AI aesthetic with a rust/blue tint
   1  anti-signature (kitsch, neon, glossy CGI)

composition — focal point, negative space, scale relation, use of the edges
  10  one unmistakable focal point, deliberate emptiness, scale tells the story
   7  clear but conventional
   4  focal point competes with clutter, or the subject is cut awkwardly
   1  no readable composition

craft — technical integrity of what is visible
  10  clean anatomy, hands, faces and edges; no artefacts
   7  minor flaws away from the focal point
   4  a visible defect on the focal subject (extra limb, melted face,
      garbled text, plastic skin)
   1  broken image

## BLOCK: OPERATORS

The editor's repertoire. Each names one kind of change to the core.

SCALE     change the scale relation: shrink the figure to a speck in vastness,
          make an object monumental, make a body too large for its room
REPEAT    multiply the subject into a line, ring, grid, tower or mirror regress
DOUBLE    duplicate, mirror or overlap the subject; a reflection that disagrees
MATERIAL  make one element behave as physical material: paint with viscosity,
          rust spreading, fabric draping, a filament under gravity, a
          translucent membrane
GRAVITY   suspend, float or invert bodies, liquids or objects
RELOCATE  move the scene into a liminal setting: empty corridor, salt flat,
          clean room, apartment with one window, operating theatre
MERGE     fuse body with architecture or object: arms from walls, a head set
          into a wall cavity, people forming a staircase
LIGHT     make light an object or an event: a slit, a beam, overexposed bloom,
          a glow from inside the material
CAMERA    change distance, angle or lens: extreme long shot, low angle,
          overhead, macro close-up
CAPTURE   change the capture texture: VHS still, archival film, CCTV frame,
          Polaroid
REDUCE    remove a competing element; get the image down to one idea
CLARIFY   rephrase what the model misread: concrete nouns, counts, spatial
          prepositions ("at the far end of", "three metres above")

LIGHT and CAPTURE exist in prose mode only: in formula mode the anchor owns
light and capture, and the core may not contain those words.

## BLOCK: ZIMAGE_RULES

Z-Image Turbo reads natural English and rewards concrete nouns, counts and
spatial relations. It ignores negative prompts.

Always:
- English only. Translate the artist's words ("Leuchtturmwärter" →
  "lighthouse keeper"); a German noun in the core is read as noise.
- No lists, line breaks, markdown or quotes in the core.
- Every must-keep is named in the core by its plain noun, every round. A
  must-keep that is only implied ("a tower" for "a lighthouse") gets lost.
- Content first — camera distance, subject, twist, setting — then finish:
  light, palette, capture. Early words decide what is in the picture, late
  words decide how it looks.
- Spatial language over adjectives: "a lone figure at the far end of an
  endless corridor" beats "a mysterious lonely atmospheric corridor".
- One surreal idea per image.
- Write constraints as presence. "No people" → "an empty plaza"; "no text"
  needs nothing at all. Drop every "No watermark / no logos / no text" line:
  the model ignores it and it costs attention.
- No meta-tags: masterpiece, 8k, best quality, trending, award-winning.
- Never edit, repeat or paraphrase the anchor: it is added around the core.

Formula mode (LoRA stack active):
- The core is 8–40 words, one comma-joined phrase: camera distance +
  subject + twist + setting. Example: "extremely long shot of a lone surgeon
  untangling a web of strings stretched across an empty salt flat".
- Shorter is stronger here: every extra clause outweighs the LoRAs that
  carry the style. Add words only to fix intent; prefer REDUCE.
- No palette, light or capture words in the core — the anchor has them.
  That includes every colour word and its compounds: rust, orange,
  rust-orange, copper, amber, blue, blue-grey, teal, grey, red, gold, and
  "VHS", "grain", "film". Materials are fine ("rust growing over" is a
  material; "rust-orange patches" is a colour).

Prose mode (no LoRA):
- The core is one paragraph, 120–220 words, in this order: subject and
  twist → setting and composition (camera, lens, distance) → material
  physics (mass, viscosity, surface tension, oxidation) → light (direction,
  exposure) → palette (rust orange against dim blue, scarce warmth) →
  capture (grain, VHS, archival).
- Describe physics, not feelings: "paint ribboning under its own weight,
  surface tension holding before it breaks" beats "a beautiful flowing pour".

## BLOCK: ANCHORS

The locked text the program puts around the core. `{triggers}` is the
comma-joined trigger words of the active LoRA stack.

signature (formula mode) — the anchor of the artist's published portrait
series, verbatim:
  head: "{triggers}, Psychologically charged conceptual photograph,"
  tail: "in the style of contemplative absurdities and neo-ultra panavision 70, surrealism, weirdcore/oddcore, Rust orange and dim blue, VHS filter"
  LoRA stack (from "Desert Reflections"):
    rustorangeanddimblue 0.35, art_vision_ZIG 0.20,
    zidiusArt_melancholy 0.45, art_vision_v2 0.35
  To test against it: artvis3 (Art Vision v3) 0.60 + zidiusArt_melancholy 0.30.

prose (prose mode):
  head: "{triggers}" — empty when no LoRA is active
  tail: ""

## PROMPT: BRIEF

You set up an iterative image-development run for one artist. You turn the
artist's idea into a frozen BRIEF that every later round is judged against,
and you write the first prompt core.

{TASTE}

{ZIMAGE_RULES}

PROCEDURE
1. Read the idea (any language). Name its SUBJECT in concrete nouns —
   what the picture is of.
2. Name its TWIST — the one impossible thing. If the idea already has one,
   keep it in the artist's words. If it has none, propose one that fits the
   artist's signature and the subject, and set twist_source to "proposed".
3. MUST_KEEP: up to four elements the artist explicitly asked for (counts,
   objects, places, relations), as short plain English nouns or noun phrases
   — translated, never in the original language, never listed twice. Never
   invent a must-keep; an idea of three words may have one or none.
4. FREE: what later rounds may change — staging, camera, setting, scale,
   light — so long as the subject, twist and must-keeps survive.
5. MOOD: three to six words from the artist's vocabulary.
6. CORE: the first core for the given mode, following the rules exactly. It
   names every must-keep by its noun. Make it faithful and simple; later
   rounds will push it.

OUTPUT — JSON only, exactly these keys:
{"subject": "...", "twist": "...", "twist_source": "user|proposed",
 "must_keep": ["..."], "free": ["..."], "mood": "...", "core": "..."}

## TEMPLATE: BRIEF_USER

IDEA: {idea}
MODE: {mode}
SOURCE PROMPT (the picture this run starts from, may be empty): {source_prompt}

## PROMPT: CRITIC

You are the critic in an iterative image-development loop for one artist.
You look at one rendered image and report exactly what is there and how
close it comes to the brief and to the artist's signature. You never write
or suggest prompt text — another model does that from your report.

{TASTE}

{RUBRIC}

PROCEDURE — in this order, and finish each step before the next:
1. OBSERVE the image literally, 60–100 words: what and how many,
   where, at what scale, the light, the palette, the capture texture. No
   interpretation, no praise. If something is ambiguous, say what it most
   looks like.
2. MUST-KEEP: for each brief item, "present", "weak" (there but small,
   unclear or misplaced) or "missing".
3. SCORE all five criteria against the anchors. Judge what is visible, not
   what was probably intended. Detail density is not quality.
4. DEFECTS: at most three, the most important first. Each names what is
   wrong, where, and what the viewer reads instead — in visual terms.
   ("The filament reads as a crack in the wall, not a thread hanging in
   air.") No prompt words, no fixes.
5. KEEP: at most three concrete strengths that must survive the next edit.

Be strict. Flattery breaks the loop: an inflated score makes a weak image
the best and every later round edits the wrong picture.

OUTPUT — JSON only, exactly these keys:
{"observed": "...",
 "must_keep": [{"item": "...", "status": "present|weak|missing"}],
 "scores": {"intent": 0, "surreal": 0, "signature": 0, "composition": 0, "craft": 0},
 "defects": ["..."],
 "keep": ["..."]}

## TEMPLATE: CRITIC_USER

BRIEF
subject: {subject}
twist: {twist}
must_keep: {must_keep}
mood: {mood}

The image is the candidate from round {round} of {rounds}.
{fragile_line}

## PROMPT: DUEL

You decide which of two images is the stronger realisation of one brief, in
one artist's signature. The images are labelled A and B. Their order means
nothing: you will be asked again with the order reversed, and only an
answer that holds both ways counts — so judge the pictures, not the order.

{TASTE}

PROCEDURE
1. For A, then B: one sentence on how well it shows the brief's subject AND
   twist; one sentence on how strongly it reads as the artist's work.
2. Decide in this order of importance: intent, surreal force, signature,
   composition, craft. A clear win on an earlier criterion outweighs any
   number of wins on later ones. Detail density is not quality.
3. If they are genuinely equal, say "tie". Do not invent a difference.

OUTPUT — JSON only, exactly these keys:
{"a": "...", "b": "...", "winner": "A|B|tie", "reason": "..."}

## TEMPLATE: DUEL_USER

BRIEF
subject: {subject}
twist: {twist}
must_keep: {must_keep}
mood: {mood}

The first image is A, the second image is B.

## PROMPT: EDITOR

You are the director in an iterative image-development loop for one artist.
Each round you make ONE deliberate change to the prompt core, so the next
render moves closer to the brief and to the artist's signature. The critic's
report tells you what the picture actually shows; you decide what to change.

{TASTE}

{ZIMAGE_RULES}

{OPERATORS}

PROCEDURE
1. DIAGNOSE in one sentence the biggest gap between the BEST image and the
   target: name the weakest rubric criterion (weighted: intent counts most)
   and why the picture falls short, using the critic's observation and
   defects. Address the defect that costs the most points — not simply the
   first one listed.
2. CHOOSE ONE OPERATOR from ALLOWED OPERATORS — the program has already
   applied the phase, the mode and the plateau to that list; any other
   operator is rejected.
   - explore: one bold operator that changes what the picture shows. Never
     repeat an operator that was rejected in the last three rounds unless
     you change how it is executed, and say how.
   - refine: one small operator, or CLARIFY.
   - polish: CLARIFY or REDUCE only — fix misreadings and craft, add nothing.
   - plateau: true → change the composition substantially, keeping every
     must-keep.
   If an intent must-keep is weak or missing, fix that first, whatever the
   phase: CLARIFY.
3. EDIT THE BEST CORE — not the latest candidate's core when that candidate
   was rejected. Keep every must-keep (named by its noun) and every KEEP
   strength of the best.
4. WRITE the complete new core within the mode's length and rules. Count the
   words. In formula mode, if the core is already near 40 words, your
   change must remove as much as it adds.
5. CHANGE: one short sentence saying what you changed ("replaced the stone
   tower with living people stacked shoulder on shoulder") — not the core.
6. EXPECT: one sentence on what should visibly change in the next render.
   It lets the next report tell whether your change worked.

Never touch the anchor; never put colour, light or capture words into a
formula-mode core; never stack two surreal ideas; never move toward the
anti-signature (body horror included); never turn the run into a different
idea; write English only.

OUTPUT — JSON only, exactly these keys:
{"diagnosis": "...", "operator": "SCALE|REPEAT|DOUBLE|MATERIAL|GRAVITY|RELOCATE|MERGE|LIGHT|CAMERA|CAPTURE|REDUCE|CLARIFY",
 "change": "...", "expect": "...", "core": "..."}

## TEMPLATE: EDITOR_USER

BRIEF
subject: {subject}
twist: {twist}
must_keep: {must_keep}
free: {free}
mood: {mood}

MODE: {mode}   ANCHOR (locked, for reference only): head "{anchor_head}" · tail "{anchor_tail}"
ROUND: {next_round} of {rounds}   PHASE: {phase}   PLATEAU: {plateau}
ALLOWED OPERATORS: {operators_allowed}

BEST SO FAR (round {best_round}, total {best_total})
core: {best_core}
scores: {best_scores}
keep: {best_keep}

LATEST CANDIDATE (round {round}) — {accepted_line}
core: {latest_core}
observed: {latest_observed}
must_keep: {latest_must_keep}
scores: {latest_scores}
defects: {latest_defects}
expected by the last edit: {latest_expect}

HISTORY (oldest first: round · operator · change · accepted/rejected · total)
{history}

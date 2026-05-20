# Styling MD for Z-Image Turbo Prompt Enhancer — Research Brief & Deliverable

## Research Synthesis

Before the file itself, a short summary of the empirical findings that drove every word choice. All of these come from the Z-Image Turbo Hugging Face discussion thread maintained by the Tongyi-MAI team, the official fal and deAPI prompting guides, the Fliki Z-Image vocabulary dictionary (built on r/StableDiffusion stress tests), Black Forest Labs' Flux 2 guidance, and Midjourney prompt-frequency data from Andrei Kovalev's Midlibrary and the Academy of Continuing Education's 2025 artist-usage report.

**Architecture-driven constraints.** Z-Image Turbo is a 6B-parameter S3-DiT model that fuses text and image tokens in one sequence. It runs at cfg=0, so negative prompts are silently dropped — every constraint must appear positively. Its text encoder parses **sentences, not tag soup**, exactly like Flux.1 Dev and SD 3.5 Large. The official Tongyi-MAI note recommends "long and detailed prompts," and confirms that LLM-based prompt enhancement is the intended workflow (their reference enhancer is `pe.py` on the model's HF Space). Attention quality is strongest in the first ~75 tokens and degrades well before the 512-token ceiling, so the highest-signal descriptors must lead each block.

**The "plastic default" trap.** The single most replicated finding on r/StableDiffusion is that Z-Image Turbo's prior collapses to glossy stock-photo beauty unless you name a *camera body, focal length, and film stock or lighting modifier*. Emotional adjectives ("realistic," "moody," "melancholic") underperform; equipment language ("Hasselblad 110mm f/2," "Cinestill 800T halation," "Kodak Portra 400 tones") overperforms by a wide margin. Texture nouns ("visible pores," "silver grain," "patina," "paint pooling") are similarly load-bearing — without them, surfaces render as plastic.

**Artist anchors that actually carry signal.** Cross-referencing the Midjourney prompt-frequency tables with Midlibrary's per-artist visual fingerprints, the following names reliably steer modern diffusion models without overfitting: Roger Deakins (22,297 prompts; bleach-bypass, Blade Runner 2049 amber/teal), Caravaggio (39k; chiaroscuro/tenebrism), Irving Penn (still-life isolation), Hiroshi Sugimoto (minimal long-exposure), Bernd & Hilla Becher (New Topographics typology), Edward Burtynsky (industrial-aerial earth tones), Andrei Tarkovsky (Stalker draped contemplation), Andrew Wyeth (muted tempera interiors), Anselm Kiefer (lead/ash/straw material gravity), Wolfgang Tillmans (intimate still life), Alberto Seveso (ink-in-water figuration), Iris van Herpen (fluid-couture sculpture), Daniel Arsham (eroded geological figures). Names with weak or noisy footprints in current models (Refik Anadol, Beeple, Joey L., Erik Almas, Platon, Kara Rosenlund) were dropped or replaced. Cinematographer + film title pairs ("Roger Deakins, Blade Runner 2049") consistently outperform a name alone because the title locks in the visual fingerprint.

**Palette vocabulary.** The user's signature "rust orange + dim blue" maps cleanly onto well-trained color tokens. The most-replicable triggers, in descending order of reliability: *burnt sienna, copper, oxidized iron, rust patina, terracotta* on the warm side; *dim teal, slate blue, oxidized navy, payne's grey, dusk blue, cold steel* on the cool side. The pair "burnt sienna and teal" is widely seen in design palettes and works as a strong split-complementary trigger. Adding one anchor neutral ("bone white," "cream," "ash grey," "concrete grey") prevents the model from oversaturating.

**Structure for the enhancer.** Each block below is a prose-friendly *menu of high-signal phrases* rather than a finished sentence, so the Qwen3-4B enhancer can weave them naturally into the scene-specific prompt it builds. The blocks lead with composition/lens (most decisive for Z-Image), then lighting, palette, texture, and finish with anchors. Each is ~110–140 tokens. Total file: ~1,650 tokens.

---

## `styling.md` — Deliverable

```markdown
# Z-Image Turbo Styling Library — Stefan Eisele Signature Set

Six rotating style families, plus one optional aerial overlay. The enhancer
selects one Style per variant and weaves its descriptors into the natural-
language prompt as a *closing aesthetic clause*. Place the Style descriptors
AFTER the subject and action — Z-Image Turbo weights early tokens for
content and late tokens for finish. Use full sentences with commas, never
tags. The unifying palette across all six is rust orange and dim blue;
each Style biases that palette differently.

---

## Style A — Wire-Wrapped Tenebrism
*Cinematic warehouse portrait of a figure tangled in glowing copper cable.*

Medium close-up on a 85mm f/1.4 portrait lens, shallow depth of field,
volumetric god rays cutting through atmospheric haze, hard key from a
single high-side source with deep negative fill, Rembrandt triangle on
the cheek, faint amber rim along the cables. Palette of oxidized copper
and burnt orange wire-glow against dim teal shadow and slate concrete,
gauzy translucent fabric, tactile skin pores, fine dust motes, Cinestill
800T tungsten halation around the wire highlights. Mood: melancholic,
industrial, hushed. Anchors: Roger Deakins cinematography for Blade
Runner 2049, Caravaggio tenebrism, Joel-Peter Witkin staging.

---

## Style B — Sleeping Object Macro
*Hyperreal still-life where an everyday object reveals a sleeping face.*

Top-down or three-quarter macro on a 100mm f/2.8 macro lens, extreme
shallow depth of field with the focal plane on a single feature, single
soft raking key from camera-left, deep negative fill, no bounce. Object
rests on dark grainy sand or unbleached linen against a dim navy void.
Palette of bruised plum, oxblood, and burnt umber subject against
payne's-grey background, with one rust-orange catchlight. Visible
condensation droplets, surface fuzz, micro-pores, subsurface scattering
on translucent flesh. Mood: intimate, dreamlike, reverent. Anchors:
Irving Penn still life, Wolfgang Tillmans intimate object studies,
Hiroshi Sugimoto stillness.

---

## Style C — Marbled Bust Dissolve
*Profile portrait whose head dissolves into a swirling acrylic paint pour.*

Studio profile or three-quarter view on a 50mm f/1.8, medium-format
Hasselblad rendering, soft north-window key with large white bounce
fill, shadowless even wrap. Subject's marble-pale skin transitions into
fluid acrylic marbling along the crown and jawline; sometimes a
double-exposure composition where two faces share one form. Palette of
rust orange, deep cobalt, bone white, and cream cream-on-cream, on a
neutral warm-grey seamless backdrop. Glossy paint viscosity, fine
veining, hyperreal pore texture against liquid abstraction. Mood:
classical, fluid, uncanny. Anchors: classical Greco-Roman bust
photography, Alberto Seveso ink-in-water, Daniel Arsham eroded figures.

---

## Style D — Sediment Block Macro
*Painted concrete and cardboard cubes shedding paper like geological strata.*

Frontal or slight overhead on a 60mm macro lens, f/8 for full sharpness
across the block face, soft diffused north light, gentle wrap shadows,
no specular highlights. Cubes coated in peeling layers of slate blue
and rust orange industrial paint, with book pages, ledger paper, and
printed text emerging from cracks like sediment. Palette of oxidized
iron, dim teal, faded cream paper, charcoal grime. Texture: weathered
patina, flaking enamel, paint drips, paper fiber edges, hairline
fractures, dust accumulation. Often arranged in a tight grid or
brutalist stack. Mood: archaeological, entropic, quiet. Anchors: Bernd
and Hilla Becher industrial typology, Edward Burtynsky surface
abstraction, Anselm Kiefer material gravity.

---

## Style E — Draped Sheet Tableau
*Wide cinematic interior of anonymous figures inside vast white fabric.*

Cinematic wide shot on a 35mm f/2 lens, 2.39:1 aspect framing, eye-level
or low static camera, centered or rule-of-thirds composition, deep
negative space. Diffused window light from one side, soft falloff,
luminous white-on-white tonal compression with one accent of rust-orange
fabric. Massive draped sheets form portals, cocoons, and geometric
folds inside a concrete industrial space. Palette of bone white, cream,
ash grey, and a single burnt sienna fabric note; dim blue shadow in
the deepest folds. Texture: heavy cotton weave, soft creasing,
particulate haze. Anonymous silhouetted figures, contemplative slow
performance. Mood: meditative, sacral, Tarkovskian. Anchors: Andrei
Tarkovsky's Stalker interiors, Bill Viola video tableaux, Andrew Wyeth
muted tempera.

---

## Style F — Paint-Splash Figuration
*Acrylic pour explosion from which a bird, hand, or body emerges.*

Frozen-motion studio shot on a 24-70mm zoom at 1/2000s, high-speed
strobe with hard light, large white seamless backdrop, slight top-down
angle. Dynamic gestural splashes and ribbons of liquid acrylic in rust
orange and dim teal coalesce into the form of a kingfisher, a pair of
hands, or a running figure; oil-paint thickness with photographic
edge realism. Palette of pure rust orange, deep teal, payne's grey,
and bone white. Texture: glossy paint viscosity, droplets in mid-air,
fine marbling, satin sheen, hyperreal feather or skin detail emerging
from the abstract pour. Mood: kinetic, ecstatic, dissolving. Anchors:
Alberto Seveso ink figuration, Iris van Herpen fluid couture, James
Nares gestural sweep.

---

## Style G (optional overlay) — Rust Coast Aerial
*Drone view of rust-colored earth meeting dim blue water.*

High-altitude drone aerial on a 24mm equivalent, near-orthographic
top-down or 30-degree oblique, midday flat light or overcast diffusion
to flatten shadow, mild atmospheric haze. Composition uses painterly
color blocking: rust-orange oxidized coastline, salt flats, or rusted
shipwreck hull meeting payne's-grey ocean. Palette of iron oxide, ochre
sediment, dim teal water, white salt crust. Texture: aerial
abstraction, sediment plumes, corrosion striations, tidal feathering.
Mood: geological, sublime, detached. Anchors: Edward Burtynsky
Extraction/Abstraction series, New Topographics movement, Caspar David
Friedrich contemplative scale.

---

## Rotation Notes for the Enhancer

- Cycle A → B → C → D → E → F deterministically across the four variants
  per generation, advancing the cycle offset each call so all six styles
  surface evenly across batches. Use G only when the subject is a
  landscape or environment with no human figure.
- Inside each variant, render the Style descriptors as flowing prose,
  not bullets. Drop the section header. Preserve the lens-light-palette-
  texture-anchor ordering so the highest-signal tokens land first.
- Never mix two Styles in the same prompt — Z-Image Turbo collapses
  contradictory style cues into uncanny output.
- Always include at least one concrete texture noun (pores, patina,
  grain, droplet, weave, viscosity) — this is the single biggest lever
  against the model's plastic default.
- Keep one anchor name per prompt; two is the maximum. A cinematographer
  paired with a film title (e.g., "Roger Deakins, Blade Runner 2049")
  carries more signal than a name alone.
```

---

## Notes on Specific Choices

A few decisions worth flagging so they can be revised if testing diverges from expectation.

The original brief proposed Refik Anadol, Beeple, Joey L., Erik Almas, Platon, and Kara Rosenlund as anchors. I dropped all six. Anadol and Beeple are not strongly disambiguated in Midjourney/Flux footprints — they nudge toward generic "data viz" or "everydays" aesthetics rather than the specific marble-and-paint dissolve you want. The four photographers (Joey L., Almas, Platon, Rosenlund) are too thinly represented in training caption corpora to function as steering tokens; Caravaggio, Penn, Tillmans, Tarkovsky, Wyeth, the Bechers, Burtynsky, Sugimoto, Seveso, van Herpen, Arsham, and Kiefer all sit in the high-frequency tier and have legible, distinct fingerprints in Midlibrary's per-artist comparisons.

I added one anchor the brief didn't list — James Nares for Style F — because his single-sweep gestural paintings are a closer visual analog to the splash-becomes-figure aesthetic than Iris van Herpen alone, and Nares carries decent diffusion signal via his appearance in numerous art-curation captions.

Joel-Peter Witkin was added to Style A as a third anchor because the wire-wrapped-figure-in-warehouse aesthetic in your references reads closer to his staged tableaux than to a pure Deakins cinematography reference. Drop him if testing produces results that feel too macabre.

The "Cinestill 800T tungsten halation" cue in Style A is the single most reliable trigger I found for the warm-amber-glow-on-cool-shadow look in the wire-wrapped references — it consistently pulls Z-Image, Flux, and SD 3.5 toward that exact halation aesthetic.

Total file weight is approximately 1,650 tokens by tiktoken cl100k count, leaving headroom for the separate Z-Image Turbo system prompt within a reasonable Qwen3-4B context.
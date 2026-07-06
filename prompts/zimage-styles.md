# Z-Image Turbo Styling Library — Stefan Eisele Signature Set

Four style families, each built from a favorite artist's moodboard
(technique, composition, framing — never their subject matter) and then
run through one shared material identity: a rust-orange and dim-blue
palette with a viscous, iridescent liquid-paint pour dissolving or
birthing the figure. The enhancer selects one Style per variant and
weaves its descriptors into the natural-language prompt as a *closing
aesthetic clause*. Place the Style descriptors AFTER the subject and
action — Z-Image Turbo weights early tokens for content and late tokens
for finish. Use full sentences with commas, never tags.

---

## Style A — Ember-Horizon Levitation
*Gravity-defying figures suspended against a night skyline lit by a
distant fire, a ribbon of liquid paint trailing from their silhouette.*

Cinematic wide shot on a 35mm lens, low static camera at rooftop or
vantage-point height, deep negative space of open sky above a suburban
or coastal horizon, contre-jour composition with the subject backlit
against a glowing bloom of distant firelight. The figure floats
mid-gesture, weightless, a viscous ribbon of paint coiling from its
trailing edge like a slipstream, an iridescent oil-slick sheen catching
the ember glow. Palette of deep midnight blue sky and slate cityscape
against rust-orange and amber fire-bloom on the horizon, filmic 35mm
grain, soft halation around the light source. Texture: atmospheric
haze, fine grain, glossy paint viscosity at the figure's edges, dusty
rooftop texture underfoot. Mood: nostalgic, uncanny, suspended,
cinematic dread. Anchors: 1980s Amblin-era suburban night
cinematography, Alberto Seveso ink figuration.

---

## Style B — Pastel Grotesque Close-Up
*Uncanny beauty close-up in flat studio pastel, the feature dissolving
at its edge into a paint pour.*

Tight beauty-macro on a 100mm f/2.8 lens, shallow depth of field, flat
high-key studio strobe with no visible shadow, symmetrical centered
framing against a seamless pastel backdrop. A single feature — a mouth,
an eye, a gloved hand — fills the frame, glossy and hyperreal, then
dissolves at its edge into a thick pour of rust-orange and dim-blue
liquid paint, the two states meeting on one uncanny surface. A thin
iridescent sheen glazes the paint boundary. Palette of soft
lavender-pink or mint studio backdrop punctuated by the rust-orange and
dim-blue pour, otherwise desaturated. Texture: glossy skin macro-pore
detail, satin paint viscosity, crisp specular highlights, no grain.
Mood: uncanny, glossy, unsettling-cute, editorial. Anchors: Miles
Aldridge pastel high-fashion surrealism, Alberto Seveso ink figuration.

---

## Style C — Mirrored Pilgrimage
*Small robed or uniformed figures dwarfed by symmetrical architecture, a
paint vortex rising at the vanishing point.*

Wide environmental shot on a 24mm lens, deep focus, eye-level or slight
low angle, strict bilateral or kaleidoscopic mirror symmetry down a
corridor, mountain pass, or road, the figure small against the
vastness. Natural overcast or blue-hour light, desaturated cool
grey-blue grade. At the vanishing point a vortex of rust-orange and
dim-blue liquid paint spirals like a portal, coiling ribbons catching a
faint iridescent sheen against the muted architecture. Palette of muted
slate-blue and sage-grey environment against the one saturated
rust-orange/dim-blue paint event. Texture: fine documentary grain, cool
atmospheric haze, glossy paint viscosity concentrated at the vortex
only. Mood: monastic, dwarfing, ritual, quietly surreal. Anchors:
Andrei Tarkovsky vast symmetrical interiors, Alberto Seveso ink-in-water.

---

## Style D — Chrome Flash Ritual
*Harsh on-camera flash on reflective, liquid-slicked figures in a dark
crowd, paint dripping like chrome.*

Direct on-camera flash photography on a 28mm lens, black crushed
background, high-contrast documentary snapshot framing, crowded or
claustrophobic composition with flash-blown highlights and hard shadow
falloff. Skin and fabric are slicked in a viscous, mirror-bright liquid
that reads as both wet latex and molten paint, dripping in rust-orange
and dim-blue rivulets with a chrome-iridescent sheen. Palette of
crushed black against rust-orange and dim-blue flash-lit highlights, no
midtones. Texture: harsh flash grain, wet specular sheen, dripping
paint viscosity, mirror-shard fragmentation. Mood: dystopian, ecstatic,
cultish, nocturnal. Anchors: underground rave flash documentary
photography, Alberto Seveso ink figuration.

---

## Rotation Notes for the Enhancer

- Cycle A → B → C → D deterministically across the variants per
  generation, advancing the cycle offset each call so all four styles
  surface evenly across batches. All four are equally weighted defaults —
  there is no optional/overflow tier.
- Inside each variant, render the Style descriptors as flowing prose,
  not bullets. Drop the section header. Preserve the lens-light-palette-
  texture-anchor ordering so the highest-signal tokens land first.
- Never mix two Styles in the same prompt — Z-Image Turbo collapses
  contradictory style cues into uncanny output.
- Always include at least one concrete texture noun (pores, grain,
  droplet, viscosity, sheen) — this is the single biggest lever against
  the model's plastic default.
- Keep one anchor name per prompt; two is the maximum. A named
  photographic reference paired with a concrete descriptor carries more
  signal than a name alone.
- The rust-orange/dim-blue palette and the liquid-paint pour/dissolve
  motif are non-negotiable across all four — they are what makes the
  four styles read as one signature set rather than four unrelated
  moodboards. Vary everything else (lens, framing, subject scale,
  lighting hardness) to keep the styles visually distinct.

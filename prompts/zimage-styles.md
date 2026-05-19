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

- Cycle A → B → C → D → E → F deterministically across the variants
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

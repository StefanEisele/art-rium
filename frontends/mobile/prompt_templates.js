// ═══════════════════════════════════════════════════════════════════════════
// STYLE TEMPLATES (personal style archive — Z-Image Turbo)
// ═══════════════════════════════════════════════════════════════════════════
const PROMPT_TEMPLATES = [
  {
    name: "Oil Diffusion Portrait",
    prompt: `Iron oxide spreading through a mineral blue field — rust as corrosion, not as color. The orange enters the way oxidation spreads across aged steel: slow, irreversible, following the grain of the surface rather than any designed path. The blue is not a background — it is a pressurized environment, dense as deep water, a field the subject exists within rather than in front of. Heavy and lightless. Where the two zones collide, forms erode: surface memory dissolves, structure gives way to chemical process. Portrait format. The image should read like a cross-section of something that has weathered for decades — material fatigue rendered in a single frame. Archival photographic quality, heavy analog grain, no digital smoothing. No watermark, no text overlays, sharp focus, professional quality.`,
  },
  {
    name: "Recursive Identities",
    prompt: `Raw industrial pigment — acrylic as a physical substance with mass and viscosity, not decoration. Paint ribboning under its own weight, surface tension holding the pour together as a continuous filament before it breaks. Burnt copper oxide bleeding into slate mineral blue, cream white settling into the low points. The freeze-frame captures not motion but physics — the exact moment before surface tension fails and the ribbon separates. Forms dissolve not in motion but in material fatigue — like a surface stressed past its structural threshold. Dramatic raking light from upper left, revealing texture across every plane. Canon 5D Mark IV with 85mm f/1.4 lens, shallow depth of field, foreground detail sharp enough to show material grain. Hyperrealistic photography of an impossible material state, tactile surface rendering. No watermark, no logos, no text overlays, sharp focus, professional quality.`,
  },
  {
    name: "Geometric Portal",
    prompt: `Heavy structural geometry draped in fabric that carries weight and memory — not decorative, not pristine. The textile shows the marks of load: pulled tight over edges, pooling where gravity wins, the specific drape of something substantial that has been used. The structural anchor points show iron oxide corrosion — rust bleeding down into the fabric from the places where stress is concentrated, orange staining the white material below the load point. Warm amber light reads as thermal rather than atmospheric — heat from within, not ambience. Wide-angle environmental shot. Shot on Sony A7R V with 35mm f/2 lens. Hyperrealistic photography emphasizing material weight and the evidence of endured stress. No watermark, no text overlays, sharp focus, professional quality.`,
  },
  {
    name: "Paint as Connection",
    prompt: `Paint as matter in gravitational relationship: a single filament of burnt copper oxide pigment falling under its own weight, viscous enough to hold as a continuous thread before the drop forms. The moment of surface contact — where the falling substance meets what receives it. Extreme close-up, macro detail revealing pigment suspended in medium, the grain of the material itself. Mineral blue atmosphere in the background — dim, vertical, pressurized. The image reads as evidence of a physical process rather than a designed composition. Hasselblad medium format with 80mm macro lens, Kodak Portra 400, film grain reinforcing the material weight. No watermark, sharp focus, professional quality.`,
  },
  {
    name: "Rust Thread / Filament",
    prompt: `A single filament of iron oxide — rust orange under the physics of gravity, thin enough to be transparent at the edges, thick enough to hold its form as a continuous thread. It connects two states: what is above and what is below, the source and what receives it. The surrounding field is mineral blue — water, haze, or atmosphere indistinguishable from each other in density and tone. The filament is the only warm element in a cold field. It reads as both intrusion and tether. Where the thread meets a surface, the contact point is the center of the composition. Photorealistic, natural light, no studio artifice. The image should feel like documentary evidence of something that cannot be documented. Analog grain, no digital smoothing. No watermark, no text overlays, sharp focus, professional quality.`,
  },
  {
    name: "Submerged Field",
    prompt: `The mineral blue of deep water as total environment — not background but pressure, density, immersion. Rust orange enters from below, rising as a thermal plume or dissolving upward the way iron oxide behaves when sediment is disturbed: branching, biological in shape, driven by convection rather than intention. The air-water boundary is visible above — a threshold, a membrane between states. Light refracts from the surface in caustic patterns, making the blue field luminous at the edges while remaining heavy at depth. The orange substance rises toward the boundary without reaching it. Underwater photographic quality — the specific color desaturation of water at depth, bubbles as material evidence, sediment as texture at the base. No watermark, no text overlays, sharp focus, professional quality.`,
  },
  {
    name: "Membrane + Echo Figure",
    prompt: `A translucent barrier as material subject — industrial plastic sheeting, gauze, or architectural membrane with its own physical presence: creases, condensation, the slight milkiness of a surface that has been under stress. A figure seen through the membrane, with ghost versions visible at the edges — the same form repeated at lower opacity, slightly offset, as if the transparency of the barrier multiplies rather than clarifies. The structural suspension points of the membrane show iron oxide rust — corrosion at the load-bearing anchor, bleeding downward into the white material below. The environment behind and around everything: dim mineral blue haze. The rust orange exists only at the structural failure point — specific, located, earned by time. Photorealistic, dim diffused interior light. Leica M10 with 35mm lens, Kodak Tri-X pushed to 1600, heavy grain. No watermark, no text overlays, sharp focus, professional quality.`,
  },
];

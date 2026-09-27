# UI icon styles — the looks scripts/ui_icons.py paints the hub's icons in

Each `## name` section below is one style: its text is appended verbatim to
the motif ("A cinema film reel, " + style), so it has to read as the second
half of that sentence. Everything above the first section is documentation
and never reaches the model.

The icons sit in a 46 px tile on a near-black card, so every style asks for
the same three things, in its own words:

- **A bold, simple silhouette.** Anything that only reads at 400 px is noise
  at 46 px. Three-quarter view gives the object a volume that a flat front
  view does not.
- **Brand colour as an accent, not a fill.** Rust orange and dim blue are
  the UI's own accents (shared.css); an icon soaked in them fights the
  buttons around it.
- **A plain mid-grey backdrop with no shadow.** The picture is cut out with
  BRIA RMBG afterwards; a cast shadow gets cut out *with* the object and
  shows up as a grey smudge on the dark card. Mid-grey (not light grey, as
  in the kids app this came from) so both dark and pale objects stand apart.

Lessons carried over from the pipeline this was ported from (17_kids_app):
say what the object IS, never what it isn't — the model draws "no face" as a
face; and no size words ("small", "tiny") — they make Z-Image draw the object
tiny and fill the frame with something else. The one exception is the shadow
line, which does hold up in its emphatic form.

## objekt
rendered as a precise industrial-design object in matte graphite and satin brushed aluminium, crisp bevelled edges, a few small parts finished in rust-orange enamel and dim steel-blue enamel. Premium studio product photograph, three-quarter view, one bold and simple silhouette that reads at a glance, centered and large in the frame, soft even studio light with a gentle rim light along the edges. Plain, flat, uniform mid-grey studio background, no gradient, no texture, nothing else in the picture. The object floats with absolutely NO cast shadow, NO drop shadow and NO reflection on the background. No text, no logo, no watermark.

## guss
sculpted from matte charcoal stone, partly coated in a thick glossy pour of rust-orange and dim-blue liquid paint that runs over its edges and drips, with a thin iridescent oil-slick sheen on the paint. Fine-art studio photograph, three-quarter view, one bold and simple silhouette that reads at a glance, centered and large in the frame, soft even studio light. Plain, flat, uniform mid-grey studio background, no gradient, no texture, nothing else in the picture. The object floats with absolutely NO cast shadow, NO drop shadow and NO reflection on the background. No text, no logo, no watermark.

## keramik
made as a matte unglazed porcelain object in warm off-white, smooth precise forms, with a few surfaces glazed in deep rust-orange and dim blue. Museum object photograph, three-quarter view, one bold and simple silhouette that reads at a glance, centered and large in the frame, soft even gallery light. Plain, flat, uniform mid-grey studio background, no gradient, no texture, nothing else in the picture. The object floats with absolutely NO cast shadow, NO drop shadow and NO reflection on the background. No text, no logo, no watermark.

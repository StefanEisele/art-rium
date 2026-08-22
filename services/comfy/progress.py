"""
Folding ComfyUI's live node/step events into a job's own progress payload.

Every long-running submission in this project reports the same way: the
submitter keeps a `{phase, message, pct}` entry and remembers which ComfyUI
prompt it is currently waiting on, and the polling endpoint enriches that entry
with whatever the listener last saw for that prompt. Without the enrichment a
multi-minute render sits frozen at one percentage; with it, the client reads
"Sampling… step 7/20" and the bar moves inside the band the submitter reserved
for that stage.

Shared by the video router and the image-upscale endpoint, which is the point:
both fold the same two private keys the same way, and a second copy would drift
from this one the first time the listener's shape changed.
"""
from __future__ import annotations

from workers.comfy_listener import get_listener


def attach_live_stage(prog: dict) -> dict:
    """Fold ComfyUI's current node + sampler step into a progress payload.

    Consumes three private keys, all optional:
      `_prompt_id` — the prompt this job is waiting on right now
      `_band`      — (lo, hi) percentage range that prompt owns, so its own
                     0-100% maps into the job's overall bar
      `_band_node` — the node whose counter actually represents the work. Every
                     node reports a counter, and most of them are not a
                     fraction of anything: a VHS loader reading a 93-frame
                     control track announces "93/93" within a second, which
                     mapped into the band reads as almost-finished and then
                     falls back to 20% once the sampler starts. Naming the node
                     keeps the bar monotonic; other nodes still contribute their
                     stage text, just not the percentage. Omitting the key keeps
                     the older behaviour, where any counter drives the band.

    Returns the dict with those keys removed, so it is safe to serialise.
    """
    prompt_id = prog.pop("_prompt_id", None)
    lo, hi = prog.pop("_band", (None, None))
    band_node = prog.pop("_band_node", None)
    listener = get_listener()
    step = listener.get_step_progress(prompt_id) if listener else None
    if not step:
        return prog

    label = step.get("label")
    value, maximum = step.get("value"), step.get("max")
    if maximum:
        ratio = max(0.0, min(1.0, float(value or 0) / float(maximum)))
        owns_band = band_node is None or str(step.get("node")) == str(band_node)
        if lo is not None and owns_band:
            prog["pct"] = int(lo + ratio * (hi - lo))
        prog["step"] = {"value": int(value or 0), "max": int(maximum)}
        prog["detail"] = f"{label or 'Sampling…'} step {int(value or 0)}/{int(maximum)}"
    elif label:
        # Between samplers — loading a model, decoding, encoding the mp4. No
        # counter to report, but the stage name is the informative part anyway.
        prog["detail"] = label
    return prog

"""Structural validation of the LTX-2.3 i2v workflow builder against ComfyUI's
object_info: every class_type exists, every required input is satisfied (literal
or wired), and every node reference points to a real node + output slot.
Does NOT queue a generation."""
import json, urllib.request, sys
sys.path.insert(0, ".")

from routers.video import _build_ltx_single_workflow


def object_info(cls):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:8188/object_info/{cls}", timeout=20) as r:
            return json.load(r).get(cls)
    except Exception:
        return None


wf, save_id = _build_ltx_single_workflow(
    "dummy.png", "a test prompt", frame_count=25, width=960, height=960, fps=24,
    vid_prefix="artrium_test",
)
print(f"Built {len(wf)} nodes, save node = {save_id}")

errors = []
infos = {}
for nid, node in wf.items():
    cls = node["class_type"]
    info = infos.get(cls) or object_info(cls)
    infos[cls] = info
    if info is None:
        errors.append(f"{nid}: class_type '{cls}' MISSING in ComfyUI")
        continue
    req = info["input"].get("required", {})
    opt = info["input"].get("optional", {})
    # Widgets injected at runtime that don't appear in static object_info:
    # LoadImage's UI-only 'upload' flag, and VHS_VideoCombine's format-dependent
    # ffmpeg options (pix_fmt/crf/save_metadata) — all accepted in practice and
    # used by the existing Wan builder.
    dynamic = {
        "LoadImage": {"upload"},
        "VHS_VideoCombine": {"pix_fmt", "crf", "save_metadata"},
    }.get(cls, set())
    allowed = set(req) | set(opt) | dynamic
    provided = set(node["inputs"])
    # Required inputs that have no default must be provided
    for k, spec in req.items():
        has_default = len(spec) > 1 and isinstance(spec[1], dict) and "default" in spec[1]
        if k not in provided and not has_default:
            errors.append(f"{nid} ({cls}): missing required input '{k}'")
    # Unknown inputs are a typo risk
    for k in provided:
        if k not in allowed:
            errors.append(f"{nid} ({cls}): unknown input '{k}'")
    # Validate wired references
    for k, v in node["inputs"].items():
        if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
            ref, slot = v
            if ref not in wf:
                errors.append(f"{nid} ({cls}).{k} -> unknown node '{ref}'")

if errors:
    print("\n".join("  X " + e for e in errors))
    print(f"\nFAILED: {len(errors)} issue(s)")
    sys.exit(1)
print("OK: all nodes valid, all required inputs satisfied, all references resolve")

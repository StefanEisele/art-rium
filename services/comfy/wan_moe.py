"""Where Wan 2.2's two experts hand over — and why it is not at half the steps.

Wan 2.2's 14B model is a mixture of two experts. The **high-noise** expert runs
first and decides composition and, crucially, *motion*: how far anything in the
frame actually travels over the clip. The **low-noise** expert takes the result
and refines texture and detail. Both are separate 14 GB files, and a render
loads both.

The handover point is not a free parameter. Wan's own inference code switches
experts at the diffusion timestep where signal and noise are 1:1, and the model
card fixes that timestep per task:

    i2v   0.900
    t2v   0.875

A sampler step is not a timestep, though. Where the boundary falls among N steps
depends on N, on the sigma shift, and on the scheduler — non-linearly. Splitting
at ``steps // 2`` is therefore only ever right by accident, and it is wrong in a
direction that costs motion: it hands the high-noise expert *more* of the
schedule than the model was trained to give it, and the high-noise expert under
a full-strength distill LoRA is exactly what produces the slow-motion look.

Measured against ComfyUI 0.33.1's own `simple_scheduler` at shift 5.0, boundary
0.900 — what the split should be, against what a 50/50 split would do:

     steps    correct    50/50
       4       1 / 3     2 / 2
       6       2 / 4     3 / 3
       8       2 / 6     4 / 4
      10       3 / 7     5 / 5
      12       4 / 8     6 / 6
      16       5 / 11    8 / 8
      20       7 / 13   10 / 10

So the correction gets *cheaper*, not more expensive, as steps go up: the extra
steps land on the low-noise expert, which is the one that adds detail.

`moe_split_step` mirrors ComfyUI's flow-model sigma math exactly rather than
asking a custom node for it, because the graph builder in routers/video.py
submits API-format workflows and there is no such node installed. It agrees with
stduhpf/ComfyUI-WanMoeKSampler, which is where the technique comes from:

    switching_step = steps
    for (i, t) in enumerate(timesteps[1:]):
        if t < boundary:
            switching_step = i
            break

Reading that carefully matters: it is the *end* sigma of a step that decides.
The high-noise expert keeps a step only while the step still lands above the
boundary — as soon as the next sigma would drop below it, the low-noise expert
takes over.
"""
from __future__ import annotations

# The scheduler this module's math is written for. Exported so the graph
# builder feeds the KSampler the same one — a builder that switched to "beta"
# while this file still assumed "simple" would compute a split for a curve the
# sampler is not walking, and nothing would fail loudly.
SCHEDULER = "simple"

# Wan's own SNR-1:1 timesteps, from the model card.
I2V_BOUNDARY = 0.900
T2V_BOUNDARY = 0.875

# ComfyUI builds a flow model's sigma table over this many discrete timesteps
# (comfy/model_sampling.py::ModelSamplingDiscreteFlow.set_parameters).
_TIMESTEPS = 1000


def time_snr_shift(shift: float, t: float) -> float:
    """Sigma at normalised timestep `t` under a sigma shift.

    Verbatim from comfy/model_sampling.py — ModelSamplingSD3's `shift` bends the
    schedule toward the noisy end, which is why the same step count crosses the
    boundary at a different place for shift 5 than for shift 8.
    """
    if shift == 1.0:
        return t
    return shift * t / (1 + (shift - 1) * t)


def flow_sigmas(steps: int, shift: float) -> list[float]:
    """The sigma schedule a Wan render actually walks, ending at 0.0.

    Mirrors comfy/samplers.py::simple_scheduler over
    ModelSamplingDiscreteFlow's sigma table, including its integer truncation —
    which is not cosmetic. `int(x * 1000 / steps)` is what makes 6 steps sample
    t = 0.834 rather than a clean 0.8333, and reproducing it is the difference
    between predicting the sampler's split and guessing near it.
    """
    steps = max(1, int(steps))
    stride = _TIMESTEPS / steps
    sigmas = [
        time_snr_shift(shift, (_TIMESTEPS - int(x * stride)) / _TIMESTEPS)
        for x in range(steps)
    ]
    return sigmas + [0.0]


def moe_split_step(steps: int, shift: float, boundary: float = I2V_BOUNDARY) -> int:
    """How many of `steps` belong to the high-noise expert.

    The return value is meant to be used directly as the high sampler's
    `end_at_step` and the low sampler's `start_at_step`, so it is clamped into
    [1, steps-1]: both experts always get real work. The clamp only ever bites
    below 4 steps (at shift 5 the boundary is already crossed by the first step
    at steps <= 2), and a 0-step KSamplerAdvanced is not a graph worth building.
    """
    sigmas = flow_sigmas(steps, shift)
    split = steps
    for i, sigma in enumerate(sigmas[1:]):
        if sigma < boundary:
            split = i
            break
    return max(1, min(split, max(1, steps - 1)))

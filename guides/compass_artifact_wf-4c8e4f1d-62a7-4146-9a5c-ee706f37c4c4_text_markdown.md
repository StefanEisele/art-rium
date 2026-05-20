# Building a Local Qwen-Based Prompt Enhancer for Z-Image Turbo in ComfyUI
### Technical Research & Recommendation Report — May 2026

This report consolidates official documentation, the Tongyi-MAI reference prompt enhancer (`pe.py`), community testing on r/StableDiffusion, the Fliki prompting guide, the `zit_enhancer` Ollama project, and the purpose-built `Qwen3-4B-Z-Image-Engineer` finetune. It is targeted at Stefan's setup: ComfyUI on an RTX 4060 Ti (16 GB) with a secondary RTX 2070 (8 GB), feeding Z-Image Turbo through ControlNet Depth and Tile Upscaling.

---

## 1. How Z-Image Turbo Actually Behaves (Why Prompts Are Different)

Z-Image Turbo is a 6 B-parameter, Apache-2.0 model from Alibaba's Tongyi-MAI lab, released late November 2025. It uses a **Scalable Single-Stream Diffusion Transformer (S3-DiT)** that concatenates text, semantic, and VAE tokens into one sequence. Four architectural facts dictate how you must write prompts:

1. **`guidance_scale = 0.0` at inference.** Classifier-free guidance is disabled in the Turbo distillation. **Negative prompts are silently ignored.** Every constraint must be phrased as a positive presence ("clean seamless backdrop" — not "no clutter"). The Tongyi team confirmed this in the HF discussion and stated they are "actively working on it".
2. **8 sampling steps, fixed.** The reference pipeline uses `num_inference_steps=9` which yields 8 DiT forwards. More steps do not help meaningfully; fewer break coherence.
3. **Text attention cap ≈ 512 tokens, with quality drift beginning much earlier.** The Tongyi team has explicitly stated that the online demo caps at 512 tokens because of this. Community testing shows attention starts to drift past **~75–100 effective tokens**; concept density matters more than raw length.
4. **Natural-language prose, not tag soup.** Z-Image was trained on narrative captions. Comma-separated SDXL/Midjourney "spell books" (`masterpiece, 8k, trending on artstation`) are essentially no-ops here — they should be replaced by concrete photography vocabulary (lens, film stock, lighting direction).

### Z-Image Turbo's "plastic default"
Out of the box the model gravitates to glossy beauty-stock aesthetics, especially for portraits, and toward young Asian/Han Chinese female subjects (a documented training-data bias). Words like "realistic", "average", or "not a model" do almost nothing by themselves; the model snaps into documentary mode only when you name **equipment, film stock, or non-idealised features** (e.g., "point-and-shoot film camera, Kodak Portra 400, slight asymmetry, visible pores").

### Other behavioural notes
- **In-image text** is a flagship strength. Always wrap exact rendered text in **straight double quotes** (`"FUTURE STACK 2026"`). This is also enforced in the official `pe.py` template.
- The model has **low intra-prompt variation** — community users report that long detailed prompts produce nearly identical renders across seeds. The bracket-syntax trick (`{A|B|C}` in ComfyUI multi-prompt nodes) is the standard workaround.
- It is **bilingual EN/CN**, but for Stefan's English-only requirement this is irrelevant.
- Long Z-Image Turbo prompts (~200–250 words / ~300–400 tokens) consistently outperform short ones, provided concepts don't contradict.

---

## 2. The Official Tongyi Prompt-Enhancer Template (`pe.py`)

Tongyi-MAI ships a reference enhancer prompt (originally in Chinese) at `huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo/blob/main/pe.py`. Translated and condensed, the official five-step workflow is:

1. **Lock immutable core elements** from the user prompt: subject, count, action, state, named IPs, colors, and any literal text. These are non-negotiable and must be preserved verbatim.
2. **Generative reasoning when needed.** If the user asks "what would X look like?" or "design Y" or "show how to solve Z", first invent a concrete, visualisable answer, then describe that answer rather than the meta-question.
3. **Inject professional aesthetic detail**: explicit composition, lighting direction and mood, material/texture, color palette, and a layered sense of spatial depth.
4. **Handle text with surgical precision.** Transcribe any rendered text exactly, wrap it in English double quotes, and for posters/menus/UI specify font, typography, and layout. Same rule applies to signage, screens, and any text the enhancer itself adds (charts, captions).
5. **Stay objective and concrete.** No metaphors, no emotional rhetoric, **no meta-tags such as "8K", "masterpiece", or drawing instructions**.

Output rule (verbatim): **"Output strictly only the final modified prompt, nothing else."**

This is the canonical behavioural spec, and the recommended system prompt below maps each Tongyi rule to a token-efficient English directive.

---

## 3. Qwen Model Selection on Stefan's Hardware

### VRAM accounting (Ollama, Q4_K_M baseline, ~4 k context)

| Model | Weights | + KV cache (≤4k ctx) | Real Ollama footprint |
|---|---|---|---|
| Qwen3-4B-Instruct-2507 Q4_K_M | ~2.5 GB | ~0.4 GB | **~3.0 GB** |
| Qwen3-4B-Instruct-2507 Q5_K_M | ~2.9 GB | ~0.4 GB | ~3.4 GB |
| Qwen3-4B-Instruct-2507 Q8_0 | ~4.3 GB | ~0.4 GB | ~4.8 GB |
| Qwen3-8B Q4_K_M | ~5.2 GB | ~0.7 GB | ~6.0 GB |
| Qwen3-14B Q4_K_M | ~9.0 GB | ~1.0 GB | ~10.0 GB |
| Qwen3-30B-A3B-Instruct-2507 Q4_K_M (MoE) | ~18 GB | ~1.5 GB | ~19 GB |

Z-Image Turbo BF16 weights are ~12 GB, plus VAE, text encoder, ControlNet Depth (~1.5 GB), and Tile Upscale tiles. On a 16 GB 4060 Ti this is already tight. Co-residing an LLM on the same GPU realistically means **≤4 GB of headroom**, which immediately excludes anything above Qwen3-4B at Q5_K_M.

### Why Qwen3-4B-Instruct-2507 is the sweet spot

- **Quality:** Qwen3-4B is the model the broader community has converged on for image-prompt enhancement work. The `BennyDaBall/Qwen3-4B-Z-Image-Engineer-V2` finetune (a Z-Image Turbo-specific LoRA-merge with 4 k+ monthly downloads) uses exactly this base. It's good enough to write 200–250-word cinematographic paragraphs, follows multi-rule system prompts reliably, and crucially **does not over-think** a fundamentally creative task — unlike the Qwen3 "Thinking" variants, which waste tokens on reasoning chains a prompt enhancer doesn't need.
- **Speed:** On a 4060 Ti or 2070, a 4B Q4–Q5 model produces ~40–60 tokens/sec, so a 250-word enhanced prompt arrives in 4–6 seconds. That's compatible with iterative image generation in ComfyUI.
- **VRAM:** ~3.0–3.4 GB leaves room for everything else.
- **Recency:** The `-2507` (July 2025) refresh significantly improved instruction following over the original Qwen3-4B; multi-turn context retention is now strong, useful if the ComfyUI Ollama node feeds back refinements.
- **Bigger isn't better here.** Qwen3-8B at Q4 gains marginal quality on this task but doubles the VRAM and halves the speed. The 30B-A3B MoE used by `zit_enhancer` is excellent quality but only viable on Stefan's 4060 Ti if ComfyUI is *not* loaded simultaneously — defeating the purpose.

### Dual-GPU placement strategy (recommended)

Stefan's RTX 2070 (8 GB) is the obvious home for Ollama. Set Ollama to bind to GPU 1:

```
# In ComfyUI startup environment
set CUDA_VISIBLE_DEVICES=0           # 4060 Ti for ComfyUI

# In separate Ollama process / system env
set CUDA_VISIBLE_DEVICES=1           # 2070 for Qwen
set OLLAMA_KEEP_ALIVE=30m            # keep model warm between gens
```

With Qwen on the 2070, you can comfortably run **Qwen3-4B-Instruct-2507 at Q6_K (≈3.3 GB) or even Q8_0 (≈4.8 GB)** — Q8 is essentially full-quality and still leaves 3 GB free on the 2070. If Stefan ever drops the dual-GPU setup, fall back to Qwen3-4B Q4_K_M on the 4060 Ti.

### Recommended pulls (in order of preference)

```bash
# Primary recommendation — generic base, fully controlled by your system prompt
ollama pull qwen3:4b-instruct-2507-q5_K_M

# If on the 2070 alone (more headroom, near-FP16 quality)
ollama pull qwen3:4b-instruct-2507-q8_0

# Optional: purpose-built Z-Image finetune (pre-baked behaviour, less steerable)
ollama pull hf.co/BennyDaBall/qwen3-4b-Z-Image-Engineer:Q5_K_M
```

The community `Z-Image-Engineer` finetune is excellent but opinionated — it always produces a 200–250-word paragraph and embeds its own lens-spec biases. For a **styling MD file approach where Stefan wants control**, the base Qwen3-4B-Instruct-2507 plus a custom system prompt is the more flexible foundation.

### Ollama Modelfile parameters

Lower temperature than chat-default is correct for this task — you want disciplined rewriting, not creativity drift:

```
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.05
PARAMETER num_ctx 4096
PARAMETER num_predict 512
PARAMETER stop "</prompt>"
PARAMETER stop "\n\nUser:"
```

---

## 4. System-Prompt Design Principles

From analysing the official Tongyi `pe.py`, the BennyDaBall finetune system prompt, and the `zit_enhancer` templates, the patterns that consistently produce clean output:

1. **Open with a role, not a task.** "You are an X" outperforms "Your job is to Y" for instruction-tuned Qwen models. Keep it to one line.
2. **State the output contract first, in CAPS.** `OUTPUT: only the final prompt. No preamble, no markdown, no quotes around the result, no explanations.` Place this near the top *and* repeat it at the end — this is the single highest-leverage trick to suppress Qwen's tendency to add "Here is your enhanced prompt:".
3. **Use a numbered procedure, not paragraphs.** Qwen-Instruct follows numbered steps far more reliably than prose. Cap it at 5–7 steps.
4. **Be explicit about what to *preserve verbatim*.** Subject count, named entities, literal text in double quotes, and any provided lens/aperture specs. Loss of user-specified detail is the most common enhancer failure mode.
5. **Codify the no-go list.** Forbid: `negative prompts`, `meta tags ("8K", "masterpiece", "trending on...")`, `tag-soup commas with no syntax`, `metaphor and emotional rhetoric`, `multiple style families in one prompt`.
6. **Define handling for sparse input.** Without this, "a cat" produces either a 5-word echo or a 800-word hallucination. Specify a target word range (200–250 words is the community consensus) and instruct the model to **invent plausible concrete details that don't contradict the user**.
7. **Force English.** Add a single line: `Always write the output in English, regardless of input language.` This is critical because Qwen3 is heavily bilingual and will sometimes mirror Chinese input.
8. **Forbid line breaks and markdown inside the output.** Z-Image Turbo wants flowing prose; ComfyUI's downstream conditioning node also handles a single paragraph more cleanly than multi-line text.

### Token-efficient instruction patterns

- `Do X.` and `Never Y.` — short imperatives beat conditional explanations.
- Inline examples in the form `bad: "no clutter" → good: "clean seamless backdrop"` teach the rule and the rewrite simultaneously in ~12 tokens.
- One-shot examples (a single 4-line `INPUT → OUTPUT` pair) outperform either zero-shot or multi-shot for 4B-class models on this task, at a cost of ~150 tokens.

---

## 5. Recommended System-Prompt Template (Drop-In for Ollama)

The following template encodes every rule above in ~430 tokens. It is intentionally **style-agnostic** because Stefan plans to append a separate styling MD file. Variables in `{curly braces}` are intentional extension points for that file.

```text
You are a prompt engineer for Z-Image Turbo, a diffusion model that
ignores negative prompts and rewards concrete, sensory English prose.

OUTPUT CONTRACT (read this twice):
- Reply with the final prompt ONLY.
- One single flowing English paragraph, 120–220 words.
- No preamble, no explanation, no markdown, no quotes around the result,
  no trailing notes. If you cannot enhance, return the user input verbatim.

PROCEDURE
1. PRESERVE verbatim: subject + count, named people/IPs/brands, exact
   colors, exact in-image text, and any lens/aperture/film-stock the user
   already named. Never replace these. Never translate them.
2. REASON if the input is a question or a design brief ("what would X
   look like?", "design a Y"): silently invent a single concrete visual
   answer, then describe that answer — not the question.
3. EXPAND with: composition and camera angle, lens or capture medium,
   lighting direction + color temperature + time of day, materials and
   surface texture, color palette, spatial depth and background.
   Add 3–5 strong visual concepts maximum. Stop before contradiction.
4. RENDER TEXT: any words that must appear in the image go inside
   straight double quotes. Specify font weight, layout, and placement.
5. CONSTRAINTS AS PRESENCE: Z-Image ignores negation. Rewrite every
   "no X / without X / not Y" as a positive substitute.
   bad: "no clutter"   → good: "clean seamless backdrop"
   bad: "not a model"  → good: "ordinary everyday face, slight asymmetry"

FORBIDDEN
- Negative prompts of any kind.
- Meta-tags: "8K", "masterpiece", "best quality", "trending on...",
  "award-winning", "hyperrealistic" as a standalone word.
- Tag-soup ("woman, red dress, park, sunny, 4k").
- Mixing more than one style family (no "photoreal anime oil painting").
- Metaphor, emotion words, or instructions to the model.
- Line breaks, lists, markdown, emojis, Chinese characters.

LANGUAGE: Always write the output in English, even if the user input is
in German, Chinese, or any other language. Translate proper nouns only
when an English equivalent is clearly more recognisable.

SPARSE INPUT HANDLING: If the user gives fewer than 6 words, invent
plausible concrete details (setting, lighting, lens, palette) that do
not contradict the input. Aim for 150 words.

STYLE LAYER: {styling_md_will_be_appended_here}

EXAMPLE
INPUT: a cat
OUTPUT: A lean tortoiseshell cat perched on a sun-warmed terracotta
windowsill in a quiet Lisbon apartment, late-afternoon light slanting
through gauzy linen curtains and casting long honey-gold stripes across
its fur, half-closed amber eyes tracking a moth outside, individual
whiskers catching the rim light, peeling cream-painted wood frame and a
small chipped ceramic pot of rosemary beside its paw, shallow depth of
field with a soft creamy background of out-of-focus rooftop tiles, shot
with a 50mm f/1.8 prime, gentle film grain, Kodak Portra 400 tones,
warm muted palette of terracotta, sage, and cream, intimate documentary
mood.

Now enhance the next user input. Reply with the prompt only.
```

This template directly mirrors the five-step procedure of Tongyi's `pe.py`, with the failure-mode prevention layer (FORBIDDEN block) drawn from community testing, and the English-only and sparse-input clauses added for Stefan's specific use case.

---

## 6. Practical Tuning Notes for Stefan's Pipeline

- **ComfyUI integration.** The most maintained option is the `ComfyUI-Ollama` custom node. Point it at `qwen3:4b-instruct-2507-q5_K_M`, paste the system prompt above into the node's system field, set `keep_alive` to at least 10 minutes to avoid reload latency, and route its output into your standard CLIP text-encode node.
- **Z-Image Turbo sampler defaults to keep:** `cfg = 0.0`, `steps = 9` (= 8 DiT forwards), bf16, no negative prompt. Anything else is wasted compute.
- **Pair with bracket variety for batch work.** Because the model has low seed variance on long prompts, generate sets with `{A|B|C}` syntax via the existing ComfyUI multi-prompt nodes — that's the community-standard way to get a "photoshoot" out of one enhanced prompt.
- **For ControlNet Depth + Tile Upscaling**, the enhancer's job is unchanged — just ensure spatial/compositional terms ("centred", "three-quarter framing") stay in the output because they meaningfully steer Z-Image even at cfg=0.
- **Future-proofing:** Tongyi indicated they are working on enabling negative prompts in a future Z-Image (non-Turbo) variant. The full `Z-Image` base model already supports them. If Stefan migrates to the base model later, the FORBIDDEN block in the system prompt should be relaxed accordingly.

---

## 7. Summary Recommendation

- **Model:** `qwen3:4b-instruct-2507`
- **Quantization:** `Q5_K_M` if co-resident with ComfyUI on the 4060 Ti; `Q8_0` if isolated on the RTX 2070
- **Placement:** RTX 2070 via `CUDA_VISIBLE_DEVICES`, with `OLLAMA_KEEP_ALIVE=30m`
- **System prompt:** Use the template in section 5 — it encodes the Tongyi `pe.py` procedure plus community failure-mode prevention in ~430 tokens, leaves a clean injection point for a styling MD file, and produces English-only single-paragraph output ready to feed straight into a CLIP text-encode node.
- **Expected behaviour:** 150–220 word cinematographic prose prompts, no negatives, no meta-tags, in-image text in double quotes, 4–6 seconds per enhancement on the 2070.
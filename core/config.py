from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Server ──────────────────────────────────────────────────────────────
    port: int = 8000

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://art_rium:changeme@localhost:5432/art_rium"

    # ── ComfyUI ──────────────────────────────────────────────────────────────
    comfyui_host: str = "127.0.0.1:8188"
    comfyui_output_dir: Path = Path("E:/00_comfy/output")
    # Where ComfyUI lives, and what relaunches it. The dashboard's "ComfyUI neu
    # starten" button kills whatever holds comfyui_host's port and then runs
    # this script — which is the same one start-remote.bat calls, so the
    # restarted process carries the exact flags the box needs (--cuda-device 0
    # above all; scripts/start-comfy.bat says why).
    comfyui_dir: Path = Path("E:/00_comfy")
    comfyui_start_script: Path = Path(__file__).parent.parent / "scripts" / "start-comfy.bat"
    # Route the Wan 2.2 experts' attention through SageAttention (INT8 QK,
    # FP8 PV) instead of PyTorch SDPA. Scoped to the two Wan builders on
    # purpose — it is an approximation, and every other model in this project
    # was calibrated without it. Needs `sageattention` in ComfyUI's venv and
    # KJNodes' PathchSageAttentionKJ; set false to fall back to SDPA without a
    # code change. See routers/video.py::_attention_backend for the numbers.
    wan_sage_attention: bool = True

    # ── Segmentation (SAM 3) ─────────────────────────────────────────────────
    # The segmenter runs in ComfyUI's interpreter, not art-rium's. That venv
    # already carries torch+cu128 and a transformers new enough to ship SAM 3
    # natively, so nothing has to be installed and no custom node has to be
    # added to a render stack whose behaviour is measured. art-rium stays
    # torch-free; see scripts/sam3_segment.py.
    sam3_python: Path = Path("E:/00_comfy/venv/Scripts/python.exe")
    sam3_model_dir: Path = Path("E:/00_comfy/models/sam3/sam3-hf")
    sam3_device: str = "cuda"

    # ── Embedding training (kohya sd-scripts) ────────────────────────────────
    # Textual-inversion embeddings trained from gallery pictures
    # (services/embedding/). sd-scripts pins its own transformers/diffusers, so
    # it lives in its own venv — installing it into ComfyUI's would move the
    # render stack's packages under it. Checkpoint = the one the AnimateLCM
    # graph renders with, so an embedding learns *that* model's vocabulary.
    sd_scripts_dir: Path = Path("E:/04_sd-scripts")
    sd_scripts_python: Path = Path("E:/04_sd-scripts/venv/Scripts/python.exe")
    embedding_base_checkpoint: Path = Path(
        "E:/00_comfy/models/checkpoints/sd_15/juggernaut_reborn.safetensors"
    )
    # Numbered the way nvidia-smi numbers the cards (the trainer is launched
    # with CUDA_DEVICE_ORDER=PCI_BUS_ID): 1 is the 4060 Ti. CUDA's own default
    # order is "fastest first", which happens to agree today and is exactly
    # the kind of agreement that stops holding after a driver update.
    embedding_train_gpu: str = "1"

    # ── Storage (managed, ingested files) ────────────────────────────────────
    storage_dir: Path = Path(__file__).parent.parent / "storage"

    # ── Auth ─────────────────────────────────────────────────────────────────
    api_key: str = ""

    # ── WordPress ────────────────────────────────────────────────────────────
    wp_base_url: str = ""          # e.g. https://yourdomain.de
    wp_username: str = ""
    wp_app_password: str = ""      # WP Application Password (not account pw)
    wp_default_language: str = "en"  # Polylang language code for media uploads
    # Media-upload encoding. AVIF is ~40-60% smaller than JPEG at comparable
    # quality and the live host has Imagick+libavif (sub-sizes generated in
    # AVIF too). Flip to "jpeg" to fall back without code change.
    wp_upload_format: Literal["jpeg", "avif"] = "avif"
    wp_avif_quality: int = 65   # 0-100; 65 is the AVIF sweet spot
    wp_avif_speed:   int = 6    # 0=slowest/smallest, 10=fastest/largest

    # ── Ollama (local VLM for image analysis) ────────────────────────────────
    ollama_host: str = "http://localhost:11434"
    ollama_vlm_model:    str = "qwen2.5vl:latest"   # vision; alt-text + media metadata
    ollama_llm_model:    str = "qwen3.6:27b"        # vision; multilingual article writer (think:false required)
    ollama_titler_model: str = "qwen2.5vl:3b"       # vision; lightweight title brainstorming
    # Z-Image Turbo prompt enhancer — text-only, community mirror on the Ollama hub.
    # Pulls with: ollama pull kamekichi128/qwen3-4b-instruct-2507
    ollama_prompt_model: str = "kamekichi128/qwen3-4b-instruct-2507:latest"
    vlm_analysis_max_edge: int = 512          # downscale before sending to VLM

    # ── Instagram (optional) ─────────────────────────────────────────────────
    instagram_user_id: str = ""
    instagram_access_token: str = ""
    image_share_token: str = ""
    instagram_graph_api_base: str = "https://graph.facebook.com/v18.0"
    # Meta's `is_ai_generated` self-disclosure ("AI info" label). Applied per
    # post at container creation; this is only what a NEW post starts with, and
    # every post carries its own flag from then on. Everything this tool
    # publishes is generated, so it defaults to on.
    instagram_ai_label_default: bool = True

    # ── Public URL (needed for Instagram to fetch images) ────────────────────
    public_base_url: str = ""  # e.g. https://xyz.trycloudflare.com

    # ── Outpost (Pi posting service for cloud-scheduled posts) ───────────────
    outpost_base_url: str = ""        # e.g. https://ig.stefaneisele.com
    outpost_shared_secret: str = ""   # X-Outpost-Key

    # ── MiniMax H3 cloud video API (services/video_api/) ─────────────────────
    # Paid, per-second billing. The budget ledger in services/video_api/budget.py
    # is what stops a runaway from becoming a runaway bill; these are its knobs.
    minimax_api_key: str = ""
    minimax_api_base: str = "https://api.minimax.io"
    # MiniMax allows 2 concurrent tasks on pay-as-you-go; more just fails.
    video_api_max_concurrent: int = 2
    # Defaults for a brand-new month. An existing period carries its own
    # settings forward instead, so editing the limit in the UI sticks.
    video_api_default_limit_eur: float = 20.00
    video_api_warn_threshold_pct: int = 80
    video_api_usd_eur_rate: float = 1.08     # USD per EUR; editable in the UI
    # Applied to reservations only, never to the displayed price. Covers rate
    # drift and rounding between reserving and being billed.
    video_api_safety_factor: float = 1.10

    # ── ffmpeg (needed for Reel video generation) ─────────────────────────────
    ffmpeg_path: str = "ffmpeg"  # override if ffmpeg is not on PATH

    # ── Artist (used in WordPress rich-article footers) ──────────────────────
    artist_website_url: str = ""    # e.g. https://www.stefaneisele.com
    artist_instagram_url: str = ""  # e.g. https://www.instagram.com/stefaneiseleart/

    # ── YouTube (used by the Articles tool to embed videos via wp:embed) ─────
    # OAuth Desktop-app credentials from Google Cloud Console:
    #   APIs & Services → Credentials → OAuth 2.0 Client IDs (Desktop app)
    # Refresh token is obtained once by running `python scripts/youtube_auth.py`
    # and survives indefinitely as long as the OAuth consent screen is set to
    # "In production" (NOT "Testing" — testing tokens expire after 7 days).
    youtube_client_id:        str = ""
    youtube_client_secret:    str = ""
    youtube_refresh_token:    str = ""
    youtube_privacy_default:  str = "public"   # "public" | "unlisted" | "private"

    @field_validator("ollama_host", mode="after")
    @classmethod
    def _normalize_ollama_host(cls, v: str) -> str:
        # Tolerate bare hosts like "127.0.0.1" (Ollama's own OLLAMA_HOST env var
        # is often set this way and pydantic-settings picks it up). Prepend
        # http:// and the default port if missing.
        if not v.startswith(("http://", "https://")):
            v = "http://" + v
        if urlparse(v).port is None:
            v = v.rstrip("/") + ":11434"
        # OLLAMA_HOST is a *server bind* address; people set it to 0.0.0.0 / ::
        # to expose Ollama on the LAN. As a *client* target those are
        # unconnectable (on Windows: WinError 10049 "address invalid in this
        # context"), surfacing as httpx "All connection attempts failed".
        # Rewrite any wildcard bind address to loopback for our own calls.
        parsed = urlparse(v)
        if parsed.hostname in ("0.0.0.0", "::", "[::]", "0"):
            v = v.replace(parsed.hostname, "127.0.0.1", 1)
        return v

    @property
    def images_dir(self) -> Path:
        return self.storage_dir / "images"

    @property
    def previews_dir(self) -> Path:
        """Cached AVIF previews (services/image/preview.py).

        Derived and disposable: every file here can be re-rendered from the
        image it belongs to, so this directory needs no backup and can be
        deleted wholesale to reclaim space.
        """
        return self.storage_dir / "previews"

    @property
    def shop_prep_dir(self) -> Path:
        return self.storage_dir / "shop_prep"

    @property
    def reels_dir(self) -> Path:
        return self.storage_dir / "reels"

    @property
    def videos_dir(self) -> Path:
        return self.storage_dir / "videos"

    @property
    def improv_dir(self) -> Path:
        """Raw user-uploaded improvisation recordings (iPhone MP4s) before muxing."""
        return self.storage_dir / "improv"

    @property
    def songs_dir(self) -> Path:
        """Generated audio (ACE-Step 1.5 Turbo MP3s + optional waveform PNGs)."""
        return self.storage_dir / "songs"

    @property
    def control_dir(self) -> Path:
        """Uploaded control tracks for the VACE structure-video workflow —
        Blender depth passes, object-ID mask renders, and ordinary footage that
        DepthAnything turns into depth in-graph.

        These are assets, not job inputs: the same turntable render gets run at
        a dozen strengths while the look is being found, so it is uploaded once
        and referenced by id.
        """
        return self.storage_dir / "control"

    @property
    def embeddings_dir(self) -> Path:
        """ComfyUI's embeddings folder. Trained embeddings are written straight
        into it (under `artrium/`), because that is the only place a render can
        name them from."""
        return self.comfyui_dir / "models" / "embeddings"

    @property
    def embedding_train_dir(self) -> Path:
        """Per-training working files: the prepared dataset, the dataset and
        prompt configs, the trainer's log. Disposable once a training is done;
        the embedding itself lives in `embeddings_dir`."""
        return self.storage_dir / "embeddings"

    @property
    def sam3_script(self) -> Path:
        """The segmentation worker. Not a setting: it is part of this repo and
        travels with it, unlike the interpreter that runs it."""
        return Path(__file__).parent.parent / "scripts" / "sam3_segment.py"


settings = Settings()

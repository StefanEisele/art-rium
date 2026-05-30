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

    # ── Public URL (needed for Instagram to fetch images) ────────────────────
    public_base_url: str = ""  # e.g. https://xyz.trycloudflare.com

    # ── Outpost (Pi posting service for cloud-scheduled posts) ───────────────
    outpost_base_url: str = ""        # e.g. https://ig.stefaneisele.com
    outpost_shared_secret: str = ""   # X-Outpost-Key

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
        return v

    @property
    def images_dir(self) -> Path:
        return self.storage_dir / "images"

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


settings = Settings()

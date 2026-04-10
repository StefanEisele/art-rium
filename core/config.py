from pathlib import Path
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

    # ── Instagram (optional) ─────────────────────────────────────────────────
    instagram_user_id: str = ""
    instagram_access_token: str = ""
    image_share_token: str = ""
    instagram_graph_api_base: str = "https://graph.facebook.com/v18.0"

    @property
    def images_dir(self) -> Path:
        return self.storage_dir / "images"

    @property
    def shop_prep_dir(self) -> Path:
        return self.storage_dir / "shop_prep"


settings = Settings()

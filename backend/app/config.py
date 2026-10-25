"""Settings from environment / .env (see .env.example)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    demo_mode: bool = True
    admin_api_key: str = ""
    database_url: str = f"sqlite:///{(REPO_ROOT / 'data' / 'codmav.db').as_posix()}"
    media_keys: str = ""
    media_active_key_id: str = ""
    media_dir: Path = REPO_ROOT / "data" / "media"
    secrets_dir: Path = REPO_ROOT / "data" / "secrets"
    media_ttl_days: int = 30
    log_ttl_days: int = 365
    model_bundle: str = ""
    bundles_root: Path = REPO_ROOT / "models" / "bundles"
    config_dir: Path = REPO_ROOT / "config"
    max_document_bytes: int = 10 * 1024 * 1024
    max_video_bytes: int = 50 * 1024 * 1024
    log_level: str = "INFO"
    run_jobs_inline: bool = False   # tests: run pipeline synchronously
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    def validate_security(self) -> None:
        if not self.demo_mode and not self.admin_api_key:
            raise RuntimeError("DEMO_MODE=0 requires ADMIN_API_KEY to be set")


@lru_cache
def get_settings() -> Settings:
    return Settings()

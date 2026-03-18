from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    gemini_api_key: str | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    tavily_api_key: str | None = Field(default=None, validation_alias="TAVILY_API_KEY")
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        validation_alias="GEMINI_MODEL",
    )
    gemini_embedding_model: str = Field(
        default="gemini-embedding-001",
        validation_alias="GEMINI_EMBEDDING_MODEL",
    )
    app_data_dir: Path = Field(default=Path("data"), validation_alias="APP_DATA_DIR")
    allowed_domains_raw: str = Field(
        default="esilv.fr,www.esilv.fr,devinci.fr,www.devinci.fr",
        validation_alias="ESILV_ALLOWED_DOMAINS",
    )
    site_cache_ttl_hours: int = Field(default=24, validation_alias="SITE_CACHE_TTL_HOURS")
    max_search_hits: int = Field(default=5, validation_alias="MAX_SEARCH_HITS")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def allowed_domains(self) -> tuple[str, ...]:
        domains = [item.strip().lower() for item in self.allowed_domains_raw.split(",")]
        return tuple(item for item in domains if item)

    @property
    def conversations_dir(self) -> Path:
        return self.app_data_dir / "conversations"

    @property
    def site_cache_dir(self) -> Path:
        return self.app_data_dir / "site_cache"

    @property
    def uploads_dir(self) -> Path:
        return self.app_data_dir / "uploads"

    @property
    def upload_files_dir(self) -> Path:
        return self.uploads_dir / "files"

    @property
    def upload_registry_path(self) -> Path:
        return self.uploads_dir / "registry.json"

    @property
    def vector_store_dir(self) -> Path:
        return self.app_data_dir / "vector_store"

    @property
    def docling_artifacts_dir(self) -> Path:
        return self.app_data_dir / "docling_artifacts"

    @property
    def agent_settings_path(self) -> Path:
        return self.app_data_dir / "agent_settings.json"

    def ensure_directories(self) -> None:
        self.app_data_dir.mkdir(parents=True, exist_ok=True)
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        self.site_cache_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.upload_files_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.docling_artifacts_dir.mkdir(parents=True, exist_ok=True)

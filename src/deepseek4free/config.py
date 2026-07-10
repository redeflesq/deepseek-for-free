"""Centralized runtime configuration for deepseek4free.

Raньше конфигурация была размазана по os.getenv() вызовам в dsk/api.py,
dsk/server.py, dsk/bypass.py и example.py, каждый со своим дефолтом и без
валидации. Теперь это единый источник правды: каждая настройка объявлена
один раз, с типом и дефолтом, pydantic валидирует её при первом реальном
обращении, а не молча падает где-то в глубине обработчика запроса.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- DeepSeek client ---------------------------------------------------
    deepseek_auth_token: str = Field(default="", description="Bearer token for chat.deepseek.com")

    # Runtime data (cookies.json) lives outside the installed package on
    # purpose - `pip install .` may put the package in site-packages, which
    # is the wrong place for mutable state that needs to survive restarts
    # and be volume-mounted in Docker. Old layout mounted ./dsk:/app/dsk for
    # the same reason; this replaces that with an explicit data directory.
    data_dir: Path = Field(default=Path("data"), validation_alias="DEEPSEEK_DATA_DIR")

    # --- Cloudflare bypass service (browser-driven cookie refresher) -------
    docker_mode: bool = Field(default=False, validation_alias="DOCKERMODE")
    server_port: int = Field(default=8000, validation_alias="SERVER_PORT")
    server_ready_timeout: int = Field(default=30, validation_alias="SERVER_READY_TIMEOUT")

    # --- Chat FastAPI server -------------------------------------------------
    fastapi_server_port: int = Field(default=8018, validation_alias="FASTAPI_SERVER_PORT")
    max_upload_bytes: int = Field(default=50 * 1024 * 1024, validation_alias="MAX_UPLOAD_BYTES")

    # --- Ollama-compatible API (separate process/port, see server/ollama_compat/) ---
    # Default 11434 matches real Ollama's default port on purpose, so clients like
    # Continue.dev work against this server with apiBase=http://localhost:11434
    # without any reconfiguration.
    ollama_compat_port: int = Field(default=11434, validation_alias="OLLAMA_COMPAT_PORT")
    enable_ollama_api: bool = Field(default=True, validation_alias="ENABLE_OLLAMA_API")

    # --- Logging -------------------------------------------------------------
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    @property
    def cookies_path(self) -> Path:
        return self.data_dir / "cookies.json"


@lru_cache
def get_settings() -> Settings:
    """Returns the process-wide Settings singleton.

    lru_cache (а не модульная глобальная переменная) выбран специально: он
    даёт ленивую инициализацию (значения читаются при первом реальном
    обращении, а не при импорте модуля - важно для тестов, которые
    monkeypatch'ят переменные окружения до первого вызова) и остаётся честным
    насчёт того, что это кэш, а не константа - get_settings.cache_clear()
    явно сбрасывает его в тестах, если нужно поменять окружение между кейсами.
    """
    return Settings()

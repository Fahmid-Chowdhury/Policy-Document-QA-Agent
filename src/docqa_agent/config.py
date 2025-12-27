from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_env: str = "dev"
    log_level: str = "INFO"
    index_dir: str = "./.index"
    collection_name: str = "docqa_chunks"


def load_config() -> AppConfig:
    # Typed config object; easy to pass around.
    return AppConfig()
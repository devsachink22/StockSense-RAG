"""Centralized configuration for StockSense-RAG."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    # Still attempt to load environment variables from default locations.
    load_dotenv()


@dataclass(slots=True)
class Settings:
    """Strongly-typed settings container."""

    news_api_key: str | None
    openai_api_key: str | None
    embedding_provider: str
    vector_db_dir: Path
    data_dir: Path
    models_dir: Path
    model_path: Path
    default_llm_model: str
    default_embedding_model: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""

    data_dir = PROJECT_ROOT / "data"
    models_dir = PROJECT_ROOT / "models"
    vector_db_relative = os.getenv("VECTOR_DB_DIR", "vectordb/chroma")
    vector_db_dir = PROJECT_ROOT / vector_db_relative

    return Settings(
        news_api_key=os.getenv("NEWS_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "local").lower(),
        vector_db_dir=vector_db_dir,
        data_dir=data_dir,
        models_dir=models_dir,
        model_path=models_dir / "price_model.pkl",
        default_llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        default_embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    )

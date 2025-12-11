"""Embedding providers for StockSense-RAG."""
from __future__ import annotations

import logging
from typing import Iterable

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

from src.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Dispatch embeddings to OpenAI or a local SentenceTransformer."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider = self.settings.embedding_provider
        self.model_name = self.settings.default_embedding_model
        self._client = None
        self._local_model = None

        if self.provider == "openai":
            if not self.settings.openai_api_key:
                msg = "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai"
                raise ValueError(msg)
            if OpenAI is None:
                raise ImportError("openai package is not installed")
            self._client = OpenAI(api_key=self.settings.openai_api_key)
            self.model_name = self.model_name or "text-embedding-3-small"
        else:
            self._load_local_model()

    def _load_local_model(self) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        logger.info("Loading sentence-transformer model %s", self.model_name)
        self._local_model = SentenceTransformer(self.model_name)

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed a list of texts into dense vectors."""

        clean_texts = [text or "" for text in texts]
        if not clean_texts:
            return []

        if self.provider == "openai" and self._client is not None:
            response = self._client.embeddings.create(model=self.model_name, input=clean_texts)
            embeddings = [data.embedding for data in response.data]
            return embeddings

        if self._local_model is None:
            self._load_local_model()

        embeddings = self._local_model.encode(clean_texts, normalize_embeddings=True)
        return [vector.tolist() for vector in embeddings]

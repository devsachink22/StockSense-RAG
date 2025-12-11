"""ChromaDB vector store helpers."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb

from src.config import get_settings

logger = logging.getLogger(__name__)


class ChromaStore:
    """Thin wrapper around a persistent ChromaDB collection."""

    COLLECTION_NAME = "stock_news"

    def __init__(self, persist_directory: str | Path | None = None) -> None:
        settings = get_settings()
        directory = Path(persist_directory or settings.vector_db_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(directory))
        self.collection = self.client.get_or_create_collection(name=self.COLLECTION_NAME)

    def upsert_news_embeddings(
        self,
        news_items: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        """Insert or update embedded news documents."""

        if not news_items or not embeddings:
            logger.warning("No news items or embeddings provided for upsert.")
            return

        if len(news_items) != len(embeddings):
            raise ValueError("news_items and embeddings must be the same length")

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for item, embedding in zip(news_items, embeddings, strict=True):
            doc_text = item.get("short_summary") or item.get("clean_text") or item.get("headline") or ""
            ids.append(item.get("id") or f"{item.get('ticker', 'TICKER')}_{item.get('published_at', '')}")
            documents.append(doc_text)
            metadatas.append(
                {
                    "ticker": item.get("ticker"),
                    "published_at": item.get("published_at"),
                    "source": item.get("source"),
                    "url": item.get("url"),
                    "headline": item.get("headline"),
                }
            )

        self.collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        logger.info("Upserted %d news embeddings into Chroma", len(ids))

    def query_news(
        self,
        ticker: str,
        query: str,
        start_date: str | None = None,
        end_date: str | None = None,
        top_k: int = 10,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Query the news collection for relevant items."""

        if not query:
            raise ValueError("Query text is required")

        where = {"ticker": ticker.upper()}
        query_kwargs: dict[str, Any] = {"where": where, "n_results": top_k * 2}
        if query_embedding is not None:
            query_kwargs["query_embeddings"] = [query_embedding]
        else:
            query_kwargs["query_texts"] = [query]
        results = self.collection.query(**query_kwargs)
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        filtered: list[dict[str, Any]] = []
        for idx, doc_id in enumerate(ids):
            metadata = metadatas[idx]
            published = metadata.get("published_at")
            if start_date and published and published < start_date:
                continue
            if end_date and published and published > end_date:
                continue
            filtered.append(
                {
                    "id": doc_id,
                    "text": documents[idx],
                    "score": distances[idx] if distances else None,
                    "metadata": metadata,
                }
            )
            if len(filtered) >= top_k:
                break

        return filtered

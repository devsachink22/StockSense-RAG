"""RAG pipeline composition."""
from __future__ import annotations

import logging
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional
    OpenAI = None  # type: ignore

from src.config import get_settings
from src.embeddings.embedding_service import EmbeddingService
from src.vectordb.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Coordinates retrieval from Chroma and answer generation via an LLM."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: ChromaStore,
        llm_client: Any | None = None,
        llm_model: str | None = None,
    ) -> None:
        settings = get_settings()
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_model = llm_model or settings.default_llm_model
        self.llm_client = llm_client

        if self.llm_client is None and settings.openai_api_key and OpenAI is not None:
            self.llm_client = OpenAI(api_key=settings.openai_api_key)

    def retrieve_context(
        self,
        ticker: str,
        query: str,
        start_date: str | None = None,
        end_date: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the top news records for the user query."""

        query_vectors = self.embedding_service.embed_texts([query])
        vector = query_vectors[0] if query_vectors else None
        results = self.vector_store.query_news(
            ticker=ticker,
            query=query,
            start_date=start_date,
            end_date=end_date,
            top_k=top_k,
            query_embedding=vector,
        )
        return results

    def generate_answer(
        self,
        user_query: str,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        top_k: int = 5,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Produce a natural language answer plus the supporting context."""

        contexts = self.retrieve_context(ticker, user_query, start_date, end_date, top_k)
        if not contexts:
            return ("No relevant news articles were found for the requested window.", [])

        prompt = self._build_prompt(user_query, ticker, start_date, end_date, contexts)

        if self.llm_client is None:
            logger.warning("LLM client unavailable, returning heuristic summary.")
            bullets = [f"- {ctx['metadata'].get('headline')}: {ctx['text']}" for ctx in contexts]
            answer = (
                f"Key developments for {ticker.upper()}:\n" + "\n".join(bullets[:top_k]) +
                "\n(Note: Generated without an LLM because no API key was provided.)"
            )
            return answer, contexts

        response = self.llm_client.chat.completions.create(  # type: ignore[attr-defined]
            model=self.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial analyst who explains equity moves using factual news."
                        "Always cite concrete events and avoid speculation."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        answer = response.choices[0].message.content  # type: ignore[index]
        return answer, contexts

    @staticmethod
    def _build_prompt(
        user_query: str,
        ticker: str,
        start_date: str | None,
        end_date: str | None,
        contexts: list[dict[str, Any]],
    ) -> str:
        """Compose the LLM prompt containing retrieved context."""

        window_desc = f"from {start_date} to {end_date}" if start_date and end_date else "for the requested period"
        context_snippets = []
        for idx, ctx in enumerate(contexts, start=1):
            meta = ctx.get("metadata", {})
            snippet = (
                f"Article {idx}: {meta.get('headline')} (source: {meta.get('source')}, date: {meta.get('published_at')})\n"
                f"Summary: {ctx.get('text')}"
            )
            context_snippets.append(snippet)

        prompt = (
            f"User Question: {user_query}\n"
            f"Ticker: {ticker.upper()} {window_desc}.\n"
            "Use the following news summaries to craft a concise answer:\n"
            + "\n\n".join(context_snippets)
        )
        return prompt

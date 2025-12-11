"""FastAPI server exposing StockSense-RAG capabilities."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.embeddings.embedding_service import EmbeddingService
from src.ingestion.news_client import NewsClient
from src.preprocessing.text_cleaner import preprocess_news_items
from src.rag.rag_pipeline import RAGPipeline
from src.vectordb.chroma_store import ChromaStore
from src.ml.train_price_model import train_model
from src.ml.predict_price import predict_next_move
from src.config import get_settings

logger = logging.getLogger(__name__)

app = FastAPI(title="StockSense-RAG", version="0.1.0")
settings = get_settings()
print("OPENAI:", bool(settings.openai_api_key))
print("NEWS:", bool(settings.news_api_key))


news_client = NewsClient()
embedding_service = EmbeddingService()
vector_store = ChromaStore()
rag_pipeline = RAGPipeline(embedding_service=embedding_service, vector_store=vector_store)


class IngestRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"])
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")


class TrainRequest(IngestRequest):
    pass


class AskRequest(IngestRequest):
    question: str = Field(..., description="Natural language question about the ticker")


@app.post("/ingest-news")
async def ingest_news(request: IngestRequest) -> dict[str, Any]:
    """Fetch, preprocess, embed, and store news."""

    news_items = news_client.fetch_news_for_ticker(request.ticker, request.start_date, request.end_date)
    if not news_items:
        raise HTTPException(status_code=404, detail="No news items retrieved.")

    processed = preprocess_news_items(news_items)
    texts = [item.get("short_summary") or item.get("clean_text") or "" for item in processed]
    embeddings = embedding_service.embed_texts(texts)
    vector_store.upsert_news_embeddings(processed, embeddings)

    return {"inserted": len(processed)}


@app.post("/train-model")
async def train_endpoint(request: TrainRequest) -> dict[str, Any]:
    """Trigger model training for a ticker."""

    metrics = train_model(request.ticker, request.start_date, request.end_date)
    return metrics


@app.post("/ask")
async def ask_endpoint(request: AskRequest) -> dict[str, Any]:
    """Answer user questions with RAG + price outlook."""

    answer, contexts = rag_pipeline.generate_answer(
        user_query=request.question,
        ticker=request.ticker,
        start_date=request.start_date,
        end_date=request.end_date,
    )
    try:
        prediction = predict_next_move(request.ticker, request.end_date)
    except Exception as exc:  # model might be missing
        logger.warning("Prediction unavailable: %s", exc)
        prediction = {"error": str(exc)}

    used_articles = [
        {"headline": ctx["metadata"].get("headline"), "url": ctx["metadata"].get("url")}
        for ctx in contexts
    ]

    return {
        "rag_answer": answer,
        "price_prediction": prediction,
        "used_articles": used_articles,
    }


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


def run() -> None:
    """Launch the FastAPI app via uvicorn."""

    import uvicorn

    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()

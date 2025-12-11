"""Typer-based CLI for StockSense-RAG."""
from __future__ import annotations

import json
import logging
from typing import Any

import typer

from src.api import server as api_server
from src.embeddings.embedding_service import EmbeddingService
from src.ingestion.news_client import NewsClient
from src.ml.predict_price import predict_next_move
from src.ml.train_price_model import train_model
from src.preprocessing.text_cleaner import preprocess_news_items
from src.rag.rag_pipeline import RAGPipeline
from src.vectordb.chroma_store import ChromaStore

app = typer.Typer(name="stocksense", help="Interact with the StockSense-RAG toolkit.")

logging.basicConfig(level=logging.INFO)


def _build_rag_pipeline() -> RAGPipeline:
    embedding_service = EmbeddingService()
    vector_store = ChromaStore()
    return RAGPipeline(embedding_service, vector_store)


def _ingest_flow(ticker: str, start: str, end: str) -> int:
    news_client = NewsClient()
    news_items = news_client.fetch_news_for_ticker(ticker, start, end)
    processed = preprocess_news_items(news_items)
    embedding_service = EmbeddingService()
    embeddings = embedding_service.embed_texts(
        [item.get("short_summary") or item.get("clean_text") or "" for item in processed]
    )
    ChromaStore().upsert_news_embeddings(processed, embeddings)
    return len(processed)


@app.command("ingest-news")
def ingest_news_cmd(ticker: str = typer.Option(...), start: str = typer.Option(...), end: str = typer.Option(...)) -> None:
    """Ingest news for a ticker and date range."""

    count = _ingest_flow(ticker, start, end)
    typer.echo(f"Inserted {count} articles into the vector store.")


@app.command("train-model")
def train_model_cmd(ticker: str = typer.Option(...), start: str = typer.Option(...), end: str = typer.Option(...)) -> None:
    """Train the price prediction model."""

    metrics = train_model(ticker, start, end)
    typer.echo(json.dumps(metrics, indent=2))


@app.command("ask")
def ask_cmd(
    ticker: str = typer.Option(...),
    start: str = typer.Option(...),
    end: str = typer.Option(...),
    question: str = typer.Option(...),
) -> None:
    """Run the RAG pipeline and display the answer plus prediction."""

    pipeline = _build_rag_pipeline()
    answer, contexts = pipeline.generate_answer(question, ticker, start, end)
    typer.echo("RAG Answer:\n" + answer)
    try:
        prediction = predict_next_move(ticker, end)
        typer.echo("\nPrice Prediction:\n" + json.dumps(prediction, indent=2))
    except Exception as exc:
        typer.echo(f"Prediction unavailable: {exc}")

    typer.echo("\nTop articles:")
    for ctx in contexts:
        headline = ctx["metadata"].get("headline")
        url = ctx["metadata"].get("url")
        typer.echo(f"- {headline} ({url})")


@app.command("serve-api")
def serve_api_cmd() -> None:
    """Start the FastAPI server."""

    api_server.run()


if __name__ == "__main__":
    app()

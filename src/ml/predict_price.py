"""Inference helpers for the trained price model."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import joblib
import numpy as np

from src.config import get_settings
from src.ingestion.news_client import NewsClient
from src.ingestion.price_client import PriceClient
from src.ml.features import build_feature_table
from src.preprocessing.text_cleaner import preprocess_news_items

logger = logging.getLogger(__name__)


def _load_artifact() -> dict[str, Any]:
    settings = get_settings()
    model_path = settings.model_path
    if not model_path.exists():
        raise FileNotFoundError(
            "Trained model not found. Please run the training routine first."
        )
    return joblib.load(model_path)


def predict_next_move(ticker: str, as_of_date: str | None = None) -> dict[str, Any]:
    """Predict the probability of a positive next-day move."""

    settings = get_settings()
    as_of_dt = datetime.fromisoformat(as_of_date) if as_of_date else datetime.utcnow()
    lookback_start = (as_of_dt - timedelta(days=60)).strftime("%Y-%m-%d")
    as_of_str = as_of_dt.strftime("%Y-%m-%d")

    price_client = PriceClient()
    news_client = NewsClient()

    price_df = price_client.fetch_price_history(ticker, lookback_start, as_of_str)
    news_items = news_client.fetch_news_for_ticker(ticker, lookback_start, as_of_str)
    processed_news = preprocess_news_items(news_items)

    feature_table = build_feature_table(price_df, processed_news)
    latest = feature_table.tail(1)
    if latest.empty:
        raise ValueError("Not enough data to generate prediction.")

    artifact = _load_artifact()
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    features = latest[feature_columns].fillna(0.0).to_numpy(dtype=np.float32)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0, 1]
    else:
        proba = float(model.predict(features)[0])

    direction = "up" if proba >= 0.5 else "down"
    return {
        "ticker": ticker.upper(),
        "as_of_date": as_of_str,
        "up_move_probability": float(proba),
        "predicted_direction": direction,
    }

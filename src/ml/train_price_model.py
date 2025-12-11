"""Training script for the price prediction model."""
from __future__ import annotations

import argparse
import logging
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.config import get_settings
from src.ingestion.news_client import NewsClient
from src.ingestion.price_client import PriceClient
from src.ml.features import build_feature_table
from src.preprocessing.text_cleaner import preprocess_news_items

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

FEATURE_COLUMNS = [
    "return_1d",
    "ma_3",
    "ma_7",
    "ma_14",
    "volatility_5",
    "volatility_10",
    "volume_ma_5",
    "rsi_14",
    "macd",
    "macd_signal",
    "news_count",
    "news_count_last_3d",
    "news_count_last_7d",
]


def train_model(ticker: str, start_date: str, end_date: str) -> dict[str, Any]:
    """Train the RandomForest classifier and persist the artifact."""

    settings = get_settings()
    price_client = PriceClient()
    news_client = NewsClient()

    price_df = price_client.fetch_price_history(ticker, start_date, end_date)
    raw_news = news_client.fetch_news_for_ticker(ticker, start_date, end_date)
    processed_news = preprocess_news_items(raw_news)

    feature_table = build_feature_table(price_df, processed_news)
    if feature_table.empty:
        raise ValueError("Insufficient data to train the model.")

    X = feature_table[FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=np.float32)
    y = feature_table["target_direction"].to_numpy(dtype=np.int8)

    if len(feature_table) < 20:
        logger.warning("Dataset is small; consider a longer training window.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    model = RandomForestClassifier(
    n_estimators=400,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=False)

    artifact = {"model": model, "feature_columns": FEATURE_COLUMNS}
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, settings.model_path)
    logger.info("Saved trained model to %s", settings.model_path)

    return {"accuracy": accuracy, "classification_report": report}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train price prediction model")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g., AAPL")
    parser.add_argument("--start", required=True, help="Training window start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="Training window end date (YYYY-MM-DD)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metrics = train_model(args.ticker, args.start, args.end)
    logger.info("Training complete. Accuracy: %.3f", metrics["accuracy"])
    print(metrics["classification_report"])  # noqa: T201


if __name__ == "__main__":
    main()

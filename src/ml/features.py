"""Feature engineering utilities."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_price_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """Derive technical indicators from OHLCV data."""

    df = price_df.copy()
    if "Date" not in df.columns:
        df = df.reset_index().rename(columns={"index": "Date"})

    df["return_1d"] = df["Close"].pct_change()
    df["ma_3"] = df["Close"].rolling(window=3, min_periods=1).mean()
    df["ma_7"] = df["Close"].rolling(window=7, min_periods=1).mean()
    df["ma_14"] = df["Close"].rolling(window=14, min_periods=1).mean()
    df["volatility_5"] = df["return_1d"].rolling(window=5, min_periods=1).std().fillna(0.0)
    df["volume_ma_5"] = df["Volume"].rolling(window=5, min_periods=1).mean()

    df["target_return"] = df["Close"].shift(-1) / df["Close"] - 1.0
    df["target_direction"] = (df["target_return"] > 0.002).astype(int)
    # RSI 14
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = exp1 - exp2
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # 10-day volatility
    df["volatility_10"] = df["return_1d"].rolling(10).std().fillna(0.0)

    return df


def news_items_to_frame(news_items: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate news counts per day."""

    rows: list[dict[str, Any]] = []
    for item in news_items:
        published = item.get("published_at", "")
        date_key = published[:10]
        if not date_key:
            continue
        rows.append({"Date": date_key, "news_count": 1})

    if not rows:
        return pd.DataFrame(columns=["Date", "news_count"])

    df = pd.DataFrame(rows)
    agg = df.groupby("Date").sum().reset_index()
    agg["Date"] = pd.to_datetime(agg["Date"])
    agg = agg.sort_values("Date")
    agg["news_count_last_3d"] = agg["news_count"].rolling(window=3, min_periods=1).sum()
    agg["news_count_last_7d"] = agg["news_count"].rolling(window=7, min_periods=1).sum()
    return agg


def build_feature_table(price_df: pd.DataFrame, news_items: list[dict[str, Any]]) -> pd.DataFrame:
    """Combine price features with aggregated news features."""

    prices = compute_price_features(price_df)
    prices["Date"] = pd.to_datetime(prices["Date"])
    news_df = news_items_to_frame(news_items)

    if news_df.empty:
        logger.warning("No news data available for feature enrichment.")
        news_df = pd.DataFrame(
            {
                "Date": prices["Date"],
                "news_count": np.zeros(len(prices)),
                "news_count_last_3d": np.zeros(len(prices)),
                "news_count_last_7d": np.zeros(len(prices)),
            }
        )

    # ✅ FIX: Flatten MultiIndex columns before merging
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = [
            "_".join([str(c) for c in col if c != ""])
            for col in prices.columns
        ]

    # ✅ Ensure Date is clean for merge
    prices["Date"] = pd.to_datetime(prices["Date"])
    news_df["Date"] = pd.to_datetime(news_df["Date"])

    # ✅ Now safe to merge
    features = prices.merge(news_df, on="Date", how="left")

    features[["news_count", "news_count_last_3d", "news_count_last_7d"]] = features[
        ["news_count", "news_count_last_3d", "news_count_last_7d"]
    ].fillna(0.0)

    features = features.dropna(subset=["target_return", "target_direction"])
    return features.sort_values("Date")

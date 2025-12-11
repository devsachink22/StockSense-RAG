"""Price ingestion utilities using yfinance."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import get_settings

logger = logging.getLogger(__name__)


class PriceClient:
    """Fetches and caches OHLCV history for equities."""

    def __init__(self, data_dir: Path | None = None) -> None:
        settings = get_settings()
        self.prices_dir = data_dir or (settings.data_dir / "prices")
        self.prices_dir.mkdir(parents=True, exist_ok=True)

    def fetch_price_history(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Retrieve price history and persist it as CSV."""

        logger.info("Downloading price history for %s", ticker)
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            msg = f"No price data returned for {ticker} between {start_date} and {end_date}."
            logger.error(msg)
            raise ValueError(msg)

        sanitized_start = start_date.replace("-", "")
        sanitized_end = end_date.replace("-", "")
        file_path = self.prices_dir / f"{ticker.upper()}_{sanitized_start}_{sanitized_end}.csv"
        df.to_csv(file_path)
        logger.info("Saved price history to %s", file_path)
        return df.reset_index(names="Date")

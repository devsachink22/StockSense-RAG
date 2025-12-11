"""Utilities for collecting equity-related news."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests

from src.config import get_settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NewsClient:
    """Lightweight client to fetch or mock news for a given ticker."""

    NEWS_API_URL = "https://newsapi.org/v2/everything"

    def __init__(self, data_dir: Path | None = None, api_key: str | None = None) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.news_api_key
        self.raw_news_dir = data_dir or (settings.data_dir / "raw_news")
        self.raw_news_dir.mkdir(parents=True, exist_ok=True)

    def fetch_news_for_ticker(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch financial news for a ticker over a date range."""

        logger.info("Fetching news for %s between %s and %s", ticker, start_date, end_date)
        news_items: list[dict[str, Any]] = []

        if self.api_key:
            params = {
                "q": ticker,
                "from": start_date,
                "to": end_date,
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": min(limit, 100),
                "apiKey": self.api_key,
            }
            try:
                response = requests.get(self.NEWS_API_URL, params=params, timeout=30)
                response.raise_for_status()
                payload = response.json()
                articles = payload.get("articles", [])
                for idx, article in enumerate(articles):
                    news_items.append(
                        {
                            "id": f"{ticker}_{article.get('publishedAt', '')}_{idx}",
                            "ticker": ticker.upper(),
                            "headline": article.get("title") or "",
                            "summary_or_description": article.get("description") or "",
                            "published_at": article.get("publishedAt") or start_date,
                            "source": article.get("source", {}).get("name", "unknown"),
                            "url": article.get("url"),
                            "raw_text": article.get("content") or article.get("description") or "",
                            "ingest_window": {"start_date": start_date, "end_date": end_date},
                        }
                    )
            except requests.RequestException as exc:
                logger.warning("News API call failed (%s). Falling back to mock data.", exc)

        if not news_items:
            news_items = self._generate_mock_news(ticker, start_date, end_date, limit)

        self._persist_news(ticker, start_date, end_date, news_items)
        return news_items

    def _generate_mock_news(
        self, ticker: str, start_date: str, end_date: str, limit: int
    ) -> list[dict[str, Any]]:
        """Create deterministic placeholder stories when an API key is unavailable."""

        logger.info("Generating mock news for %s", ticker)
        mock_items: list[dict[str, Any]] = []
        start_dt = datetime.fromisoformat(start_date)
        for idx in range(min(limit, 5)):
            published = start_dt.strftime("%Y-%m-%dT08:00:00Z")
            mock_items.append(
                {
                    "id": f"mock_{ticker}_{idx}",
                    "ticker": ticker.upper(),
                    "headline": f"{ticker.upper()} mock news headline #{idx + 1}",
                    "summary_or_description": (
                        f"Synthetic summary for {ticker.upper()} describing market sentiment "
                        f"and corporate developments during the requested window."
                    ),
                    "published_at": published,
                    "source": "mock-feed",
                    "url": f"https://example.com/{ticker.lower()}/mock/{idx}",
                    "raw_text": (
                        f"Placeholder article {idx + 1} for {ticker.upper()} spanning from {start_date} "
                        f"to {end_date}. Useful for demo runs without API credentials."
                    ),
                    "ingest_window": {"start_date": start_date, "end_date": end_date},
                }
            )
            start_dt += timedelta(days=1)
        return mock_items

    def _persist_news(
        self, ticker: str, start_date: str, end_date: str, news_items: list[dict[str, Any]]
    ) -> None:
        """Save raw news payload to disk for traceability."""

        sanitized_start = start_date.replace("-", "")
        sanitized_end = end_date.replace("-", "")
        file_path = self.raw_news_dir / f"{ticker.upper()}_{sanitized_start}_{sanitized_end}.json"
        with file_path.open("w", encoding="utf-8") as fp:
            json.dump(news_items, fp, ensure_ascii=False, indent=2)
        logger.info("Persisted %d news items to %s", len(news_items), file_path)

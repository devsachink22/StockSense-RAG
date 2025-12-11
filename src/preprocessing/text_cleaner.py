"""Text preprocessing helpers for news ingestion."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from src.config import get_settings

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Remove HTML artifacts and normalize whitespace."""

    soup = BeautifulSoup(text or "", "html.parser")
    stripped = soup.get_text(" ")
    ascii_text = stripped.encode("ascii", errors="ignore").decode("ascii")
    normalized = re.sub(r"\s+", " ", ascii_text).strip()
    return normalized


def summarize_text(text: str, sentence_count: int = 2) -> str:
    """Return a naive summary by keeping the first N sentences."""

    if not text:
        return ""
    # Split on '.', '!' or '?' boundaries.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    summary = " ".join(sentences[:sentence_count]).strip()
    return summary or text[:280]


def preprocess_news_items(news_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Augment raw news items with cleaned text and heuristics."""

    settings = get_settings()
    processed_dir = settings.data_dir / "processed_news"
    processed_dir.mkdir(parents=True, exist_ok=True)

    processed: list[dict[str, Any]] = []
    for item in news_items:
        base_text = item.get("raw_text") or item.get("summary_or_description") or item.get("headline") or ""
        clean = clean_text(base_text)
        summary = summarize_text(clean if clean else item.get("summary_or_description", ""))
        enriched = {**item, "clean_text": clean, "short_summary": summary}
        processed.append(enriched)

    if processed:
        ticker = processed[0].get("ticker", "UNKNOWN")
        ingest_window = processed[0].get("ingest_window", {})
        start_date = str(ingest_window.get("start_date", "na")).replace("-", "")
        end_date = str(ingest_window.get("end_date", "na")).replace("-", "")
        file_path = processed_dir / f"{ticker}_{start_date}_{end_date}.json"
        with file_path.open("w", encoding="utf-8") as fp:
            json.dump(processed, fp, ensure_ascii=False, indent=2)
        logger.info("Persisted processed news to %s", file_path)
    else:
        logger.warning("No news items supplied for preprocessing.")

    return processed

# StockSense-RAG

StockSense-RAG is an end-to-end Retrieval-Augmented Generation (RAG) system for equity analysis. It ingests stock-specific news, embeds articles into a ChromaDB vector store, answers natural language questions with contextual retrieval, and pairs the narrative output with a machine learning-based price outlook.

## Architecture Overview

```
[News APIs/Scrapers] --> [Ingestion] --> [Preprocessing] --> [Embeddings] --> [ChromaDB]
                                                           |                     |
                                               [Price Data via yfinance]          |
                                                           |                     |
                                                        [Features] --> [RF Model]
                                                           |                     |
                                               [RAG Pipeline / FastAPI / CLI] <---+
```

- **Ingestion** pulls news headlines (real API or mock) and OHLCV prices (yfinance).
- **Preprocessing** cleans and summarizes news for downstream embedding.
- **Embeddings + Vector DB** store semantic representations for low-latency retrieval.
- **RAG Pipeline** retrieves relevant news, builds prompts, and queries an LLM (OpenAI or a heuristic fallback).
- **ML Module** engineers technical and news-derived features, trains a RandomForest classifier, and predicts short-term direction.
- **Interfaces** include a Typer CLI and a FastAPI server for automation or integration.

## Getting Started

1. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your NEWS_API_KEY / OPENAI_API_KEY if available
   ```
4. **Populate the vector store (optional demo)**
   ```bash
   python -m src.cli.main ingest-news --ticker AAPL --start 2025-11-12 --end 2025-12-10
   ```

## CLI Usage

All commands are namespaced under `stocksense`:

```bash
python -m src.cli.main ingest-news --ticker AAPL --start 2025-11-12 --end 2025-12-10
python -m src.cli.main train-model --ticker AAPL --start 2025-11-12 --end 2025-12-10
python -m src.cli.main ask --ticker AAPL --start 2025-11-12 --end 2025-12-10 --question "Why was the stock volatile?"
python -m src.cli.main serve-api
```

## FastAPI Endpoints

Start the server (either via CLI `stocksense serve-api` or `uvicorn src.api.server:app`). Then call:

```bash
curl -X POST http://localhost:8000/ingest-news \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "start_date": "2025-11-12", "end_date": "2025-12-10"}'

curl -X POST http://localhost:8000/train-model \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "start_date": "2025-11-12", "end_date": "2025-12-10"}'

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "start_date": "2025-11-12", "end_date": "2025-12-10", "question": "What drove price moves?"}'
```

Responses include the natural language explanation (`rag_answer`), price outlook (`price_prediction`), and supporting articles.

## Notes

- Without API keys, the system falls back to deterministic mock news and a rule-based RAG summary so you can exercise the full pipeline.
- OpenAI is optional; configure `EMBEDDING_PROVIDER=local` for sentence-transformers and rely on the heuristic RAG answer.
- The RandomForest classifier is intentionally simple and intended as a baseline—extend `src/ml/features.py` for richer features or sentiment signals.

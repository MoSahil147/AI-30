# RAG Pipeline — From Baseline to Production API (Week1)

A production-grade Retrieval Augmented Generation system built over 5 days,
with measurable improvements at each stage.

## Results

| Day | What I Built | Faithfulness | Context Recall |
|-----|-------------|--------------|----------------|
| 1 | Baseline RAG | 0.81 | N/A |
| 2 | + Hybrid Search (BM25 + Vector) | 0.96 | 0.80 |
| 3 | + Cross-Encoder Reranking | 1.00 | 0.80 |
| 4 | + Query Expansion | 1.00 | 0.93 |
| 5 | + FastAPI Production API | 1.00 | 0.93 |

## Tech Stack
- LangChain + ChromaDB — document storage and retrieval
- BM25 + Vector Search + RRF — hybrid retrieval
- CrossEncoder (ms-marco-MiniLM) — reranking
- Groq LLaMA 3.3 70B — generation
- RAGAS — evaluation framework
- FastAPI + Uvicorn — production API

## How It Works
1. PDF is chunked and embedded into ChromaDB
2. Query is expanded into 4 variations using LLM
3. Each variation searches via BM25 + vector (hybrid)
4. Results merged using Reciprocal Rank Fusion (RRF)
5. Cross-encoder reranks top 10 → picks best 3
6. LLM generates answer from top 3 chunks

## API Usage
Start the server:
```bash
uv run python -m uvicorn api:app --reload
```

Ask a question:
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Your question here"}'
```

## Key Learnings
- Hybrid search fixed vocabulary mismatch problem
- Reranking improved faithfulness to 1.0 (zero hallucination)
- Query expansion fixed retrieval failures for complex questions
- Ground truth quality matters as much as retrieval quality
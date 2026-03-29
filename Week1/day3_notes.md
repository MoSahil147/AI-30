This is the Plan
Question → Hybrid Search → top 10 chunks → Reranker → top 3 chunks → LLM

**The problem with retrieval alone:**

Both vector search and BM25 work by comparing the question and document **separately** — they convert each to a representation, then measure similarity.
```
Vector search:
question → [0.21, -0.54, 0.88...]
chunk    → [0.23, -0.51, 0.85...]
similarity = how close these vectors are
```

The question and chunk never "see" each other directly. The model doesn't know HOW the chunk answers the question — just that they're similar topics.

This is called a **bi-encoder** — two separate encodings.

---

**What a reranker does differently:**

A reranker uses a **cross-encoder** — it reads the question AND the chunk TOGETHER in one pass:
```
Cross-encoder input:
"[QUESTION] How many accounts were in the dataset? 
 [CHUNK] The dataset contains 12,355 accounts of which 
 8,267 are genuine and 4,088 are fake..."

Output: relevance score = 0.97 ← very relevant
```

Because it reads both together, it understands:
- Does this chunk directly answer this question?
- Is the answer actually in this chunk?
- How completely does it answer it?

This is far more accurate than comparing vectors separately.

---

**The tradeoff:**

| | Bi-encoder | Cross-encoder |
|--|--|--|
| Speed | Fast — pre-compute vectors | Slow — must process every pair |
| Accuracy | Good | Much better |
| Use case | Retrieval — scan 157 chunks | Reranking — score top 10 only |

**This is why we use BOTH:**
```
Bi-encoder (fast) → find top 10 candidates from 157 chunks
Cross-encoder (accurate) → rerank those 10, pick best 3

cross-encoder/ms-marco-MiniLM-L-6-v2
This model was trained specifically to score question-document relevance. It outputs a score for each pair — higher = more relevant.
It runs locally on your machine — free, no API needed.

Sending 5 chunks increases the risk of noisy, irrelevant content confusing the LLM. By reranking and keeping only the top 3 most relevant chunks, we improve the signal-to-noise ratio — the LLM has less to read and everything it reads is highly relevant.

Reranking: fetch 10 → score each → keep top 3
Why top 3 not 5:
- Less noise = LLM less confused
- "Lost in the middle" problem - LLMs ignore middle chunks
- Higher signal-to-noise ratio = better answers
Cross-encoder reads question + chunk TOGETHER
Bi-encoder reads them SEPARATELY
Cross-encoder more accurate but slower - use only on top 10

"A cross-encoder reads the question and chunk together, so it can directly judge whether the chunk answers the question — not just whether they're topically similar. A bi-encoder misses cases where a chunk is on the same topic but doesn't actually answer the question."


Chunk size selection:
- Research papers: 500-800
- Legal/long docs: 800-1200
- FAQ/support: 200-300
- Always experiment with RAGAS to find best size
- chunk_overlap = 10% of chunk_size rule of thumb
- Day 4: will benchmark 3 chunk sizes

"We keep chunk_size and chunk_overlap identical because we're changing only ONE variable — the reranking — so any score improvement can only be attributed to reranking, not chunk size changes."

Day 3 progress - CrossEncoder loaded
rerank() function: hybrid gets 10 → reranker scores each → returns top 3
CrossEncoder reads [question, chunk] TOGETHER → more accurate than vectors

In short 

Vector search = compares question and chunk SEPARATELY
Cross-encoder = reads question and chunk TOGETHER = understands direct answerability

Why wrap in a class?
- RetrievalQA only accepts BaseRetriever objects
- Plain functions are rejected
- Inheriting BaseRetriever = getting the "license"
- _get_relevant_documents = the one method LangChain calls


qa_chain.invoke() = chef cooking a meal
                  = gives you the FOOD (answer)
                  = doesn't show you the INGREDIENTS (chunks)

hybrid_search + rerank = going back to the kitchen
                       = getting the same ingredients
                       = showing them to RAGAS for inspection


qa_chain.invoke() returns answer only — not the chunks used
RAGAS needs contexts (chunks) separately
So we call hybrid_search + rerank again to get chunks for RAGAS
Same chunks the LLM used — just retrieved again for evaluation

"I built a RAG pipeline over 3 days. Day 1 baseline: faithfulness 0.81. Day 2 I added hybrid search — BM25 plus vector with RRF — faithfulness jumped to 0.96. Day 3 I added cross-encoder reranking — faithfulness hit 1.0, meaning zero hallucination on my test set. I measured everything with RAGAS on a ground truth test set I created manually."


## clear your conecpt

Q1 — 7/10 ⚠️
Context recall 0.80 means for 80% of questions the retriever found ALL chunks needed to answer correctly.
But your guess about which question scored 0.0 is wrong. Look at your scores again:
context_recall: [1.0, 1.0, 1.0, 1.0, 0.0]
The scores match your test questions in order:
Q1: "What F1 score..."          → 1.0 ✅
Q2: "How many total accounts..."  → 1.0 ✅
Q3: "How many genuine accounts..." → 1.0 ✅
Q4: "What API was used..."        → 1.0 ✅
Q5: "What deep learning models..." → 0.0 ❌
So Q5 "What deep learning models were used?" scored 0.0 — the retriever completely failed to find the right chunks for this question. That's your Day 4 fix target.

Q2 — 10/10 ✅
Perfect. Bi-encoder = separate encodings compared by similarity. Cross-encoder = reads both together = direct answerability scoring.


Q3 — 6/10 ⚠️
You said "because of hybrid search and cross-encoder reranker" — but that explains why faithfulness improved, not why context_recall stayed at 0.80.
The real answer

"Faithfulness hit 1.0 because reranking ensured only highly relevant chunks reached the LLM — so answers were grounded. Context_recall stayed at 0.80 because Q5 still had retrieval failure — neither hybrid search nor reranking could find the right chunks for 'What deep learning models were used?' — that question needs a different fix."

Day 3 scores:
faithfulness = 1.0 → reranking ensures only relevant chunks reach LLM
context_recall = 0.80 → Q5 "What deep learning models" still failing
Q5 scored 0.0 → retriever not finding right chunks for this question
Day 4 mission: fix Q5 retrieval failure

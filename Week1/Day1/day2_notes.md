Solution to halucination 

baic problme our retireververs used vector search only, whih finds similar chunks but it missed exact keyword matches! So if we use some specify some key terms like F1 score, it wont be able to find the right chunks!

Solution is Hybrid search
vector Search: finds by meaning
BM25 search: finds by EXACT Keywords (Google also used the same before neural search,, its too good at fidning docs conating specific words)
combine both, crazyyy

Day 2 Core Concept:
Vector search = finds by meaning (semantic similarity)
BM25 = finds by exact keywords (term frequency)
Hybrid = combine both using RRF (Reciprocal Rank Fusion)
RRF = merges two ranked lists into one better ranked list

"A chunk that scores high in both vector search and BM25 is relevant both semantically AND by exact keywords — two independent signals agreeing means higher confidence it's the right chunk."

Score = 1/(rank_in_vector + 60) + 1/(rank_in_BM25 + 60)
The 60 is a constant that prevents top-ranked results from dominating too much.

BM25 works directly on text — it tokenizes words and counts frequencies. No embeddings needed. That's actually why it's faster than vector search — no neural network involved, just word counting.

Why we are keepg the chunk size same?
Scientific principle: change ONE variable at a time.
Day 2 variable: retrieval method (vector → hybrid)
Everything else: identical to Day 1
This is how you prove what actually caused improvement.

np.argsort sorts from lowest to highest by default. But we want highest score first — best chunks at the top.
np.argsort() → sorts lowest to highest
[::-1]       → reverses to highest to lowest
Always need [::-1] when you want top scores first

Lambda function in python is used to short the get score function! we only needed one time so used one time

Block 4 summary

"Block 4 implements hybrid search. For each query, vector search finds the top 10 semantically similar chunks and BM25 finds the top 10 keyword-matching chunks. Each chunk gets an RRF score based on its rank in each system — 1 divided by rank plus 60. If a chunk appears in both systems its scores are added together (READ AGAIN). Finally all chunks are sorted by combined score and the top k are returned."

When we wrte a function and dont call it 
No garbage value. No error. No memory wasted. The function just sits in memory waiting to be called — like a recipe written in a cookbook that nobody has cooked yet.


"HybridRetriever inherits from BaseRetriever — like a child inheriting traits from a parent. We get all of LangChain's retriever infrastructure for free, and only override the one method that actually fetches documents — replacing it with our hybrid search."

Field() = pydantic's way of defining class attributes
Type hints = tell you what goes in and comes out of functions ex:  ->List[Documents]
Control variables = change ONE thing at a time to prove causation

On Day 1, pure vector search gave me faithfulness of 0.81. I diagnosed the problem, vocabulary mismatch causing wrong chunks to be retrieved. I implemented hybrid search combining BM25 and vector search using RRF. Day 2 faithfulness jumped to 0.96 — a 15% improvement. I measured this with RAGAS on a ground truth test set.


Day 2 Final Scores:
- faithfulness: 0.9600 (up from 0.8135 — +14.65%)
- context_recall: 0.8000 (new metric, unlocked with ground truth)

What caused improvement:
- Hybrid search (BM25 + vector + RRF)
- BM25 fixed vocabulary mismatch problem
- Exact keywords like "Tweepy", "12355", "F1 score" now found correctly

Day 3 target:
- Push context_recall from 0.80 toward 0.95+
- Add reranking pass to improve chunk quality further
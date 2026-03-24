Q1 — 8/10 ✅
Correct — query expansion rewrites the original question multiple ways to increase chances of matching the right chunks. One addition: we use an LLM to generate the variations, not just synonyms.
Q2 — 10/10 ✅
Perfect. faithfulness 1.0, context_recall 0.80.
Q3 — 10/10 ✅ 🎯
This is exactly right and this is the KEY insight for today:
Abstract (page 1)     → CNN/LSTM/GCN mentioned briefly
Page 6 (methodology)  → explained in detail
Conclusion            → mentioned again

This is WHY Q5 scores 0.0 — the answer is scattered across 3 different sections. A single chunk retrieval finds one mention but misses the others. The retriever needs to find chunks from ALL three locations to fully answer the question.
Query expansion fixes this by searching with multiple phrasings — increasing the chance of hitting all 3 locations.
(Now read the above again!)


Query expansion:
- Generate 3 variations of original question using LLM
- Search with all 3 variations
- Merge results using RRF (same formula as Day 2)
- Chunk appearing in multiple searches = higher confidence
- Then reranker picks final top 3

Original question
      ↓
LLM generates 3 query variations
      ↓
Search with each variation (vector + BM25)
      ↓
RRF merge all results
      ↓
Reranker picks top 3
      ↓
Groq answers

separation of concerns in engineering. Two LLMs, two jobs, no interference.

Query expansion risk: LLM generates similar queries = no benefit
Fix: prompt LLM to generate DIFFERENT angles, not just synonyms
Separation of concerns: one LLM call for expansion, one for answering

"Day 3 improved HOW we rank chunks. Day 4 improves HOW we find chunks — by searching with multiple query variations instead of just one."

f stands for formatted string. It lets you embed Python variables or expressions directly inside a string using {}.


Good thinking but the real reason is slightly different:
"We keep the original because it's the most precise version of what the user wants. The 3 variations cast a wider net. Together they maximize both precision AND coverage."
If we only used variations, we might drift too far from the original intent.


"Eat off empty spaces" = remove whitespace from start and end
"Eat up next lines" = split by newline character \n into a list
Perfect. Type it exactly as you'd say it in an interview — just more formally:

.strip() removes leading and trailing whitespace
.split("\n") splits string into list at every newline


"A chunk appearing in all 4 query variations accumulates 4 RRF scores (4 × 1/(rank+60)), so its combined score is much higher than a chunk appearing in only 1 variation — meaning chunks relevant from multiple angles float to the top."
That's the whole power of query expansion. More angles = more chances to accumulate score = better chunks win.

hybrid_search_single = the engine (does one search)
expanded_hybrid_search = the strategy (calls the engine 4 times, one per query variation, then merges)

You can't merge results from 4 queries without a function that handles just 1 query. Single responsibility — each function does one job cleanly.

Bug 1: lambda: missing the parameter x — lambda needs lambda x: to know what variable to operate on
Bug 2: outer loop variable chunk shadowed the inner loop variable chunk — Python got confused which one to use. Outer should be content
Bug 3: page_comtent → page_content — one wrong letter, Python can't find the attribute

Double retrieval problem:
- qa_chain.invoke() runs retrieval internally
- We run it again separately for RAGAS contexts
- Same result both times = correct but wasteful
- Fix: cache retrieval results (Day 6)

Ground truth quality matters as much as retrieval quality.
Bad ground truth = RAGAS penalizes correct retrievals.
Always write ground truth using language FROM the document,
not your own summary of it.
Lesson: RAGAS is only as good as your ground truth.
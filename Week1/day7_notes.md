Current behavior:
User asks "What is the capital of France?"
→ RAG searches fake accounts paper
→ LLM gets confused chunks
→ Returns garbage or hallucination

Correct behavior:
User asks "What is the capital of France?"
→ System detects out of scope
→ Returns "This question is outside my knowledge base"

Adding Guardrials


Without guardrail: fast but returns garbage for out-of-scope questions
With guardrail:    slower but always returns sensible responses

Guardrail = extra LLM call to check if question is in scope
Tradeoff: +latency +cost vs better user experience
Worth it in production - wrong answers damage trust

We check for `NO` because:
- If question is out of scope → return early, save time
- If question is in scope → continue to full pipeline
- `.upper()` ensures we catch `no`, `No`, `NO` all the same way

.upper() = defensive programming
Normalize text before comparing
"no" == "No" == "NO" after .upper()
Check for NO not YES = fail safe
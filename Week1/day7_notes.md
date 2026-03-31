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

Q1 — API key ✅
Correct — Railway has an environment variables section where you set GROQ_API_KEY. Railway injects it at runtime, same as your .env file.
Q2 — PDF ⚠️
Not quite. We can't "store in chunks" — the PDF needs to be in the repository so Railway can access it when building. We'll commit the PDF to GitHub and Railway will have it.
Q3 — Run command ✅
Close — we create a file called Procfile that tells Railway exactly what command to run:
web: uvicorn api:app --host 0.0.0.0 --port $PORT
Notice --host 0.0.0.0 and --port $PORT — two differences from local. Why?

0.0.0.0 = accept connections from anywhere (not just localhost)
$PORT = Railway assigns the port, we don't hardcode 8000

## Proctflie is like my manager, he will do the thing, mein nahi chalunga backend!

Procfile = startup instructions for cloud server
Without it → server doesn't know how to start your app
web: = process type (web server)
$PORT = cloud assigns port, we don't hardcode
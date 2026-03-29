How to reduce latency in RAGs

1. Cache frequent queries
If 100 users ask the same question, don't run the full pipeline 100 times. Store the answer the first time, return it instantly after. Redis is the standard tool for this.
2. Reduce chunk retrieval
Fetching k=10 and reranking is slower than k=3 directly. Find the minimum k that maintains quality — every extra chunk costs time.
3. Smaller embedding model
all-MiniLM-L6-v2 is already small. But if latency is critical, you can go even smaller at the cost of slight quality drop.
4. Skip reranking for simple queries
Reranking adds 200-500ms. For simple factual questions, hybrid search alone is often good enough. Rerank only when needed.
5. Async processing
Run vector search and BM25 search simultaneously instead of one after the other. Cuts search time roughly in half.
6. Faster LLM
Groq is already fast. But smaller models like Haiku or Haiku-equivalent answer in 1-2 seconds vs 5-8 seconds for 70B models.
The honest priority order:
Caching → biggest impact
Async search → medium impact  
Smaller k → small impact
Smaller model → quality tradeoff


Toady we will make a API of our RAG
API=Application Programming Interface
JSON = JavaScript Object Notation

FastAPI = the framework that lets you define your API endpoints — what URLs exist, what they accept, what they return. It's the blueprint of your factory.
Uvicorn = the web server that actually runs FastAPI — it listens for incoming HTTP requests and passes them to FastAPI. It's the electricity that powers the factory.
Request comes in
      ↓
Uvicorn receives it (the server)
      ↓
FastAPI handles it (the framework)
      ↓
Your RAG pipeline runs
      ↓
Response goes back
FastAPI = framework, defines endpoints and logic
Uvicorn = web server, receives and handles HTTP requests
Together: Uvicorn runs FastAPI, FastAPI runs your code

Concept 1 — Endpoints
An endpoint is a URL your API exposes:
GET  /health  → "is the server alive?"
POST /ask     → "here's my question, give me an answer"

/health → GET because we're just asking "are you alive?" — no data needed
/ask → POST because we're sending a question and expecting processing

GET  = asking a waiter "what's on the menu?"
POST = telling the waiter "I'll have the pasta"

GET  = fetch only, no body, safe to repeat
POST = send data + trigger action, has request body
/health → GET  (just checking status)
/ask    → POST (sending question, getting answer)

Concept 2 — Request and Response
Every API call has two parts:
Request  → what the user SENDS  (the question)
Response → what you SEND BACK   (the answer)

In FastAPI both are defined as Python classes using Pydantic:

class AskRequest(BaseModel):
    question: str        ## user sends this

class AskResponse(BaseModel):
    answer: str          ## you send this back
    sources: list        ## which chunks were used

Concept 3 — Decorators

FastAPI uses decorators to define endpoints:

@app.get("/health")      ## this URL + GET method
def health_check():
    return {"status": "ok"}

@app.post("/ask")        ## this URL + POST method  
def ask_question(request: AskRequest):
    return {"answer": "..."}

The `@app.get` and `@app.post` are decorators — they tell FastAPI "when someone hits this URL, run this function."

Summary 
FastAPI concepts:
- Endpoint = a URL your API exposes
- GET = retrieve data, no body needed (fetch data)
- POST = send data, has a request body (sends data)
- Decorator (@app.get, @app.post) = connects URL to function
- Pydantic models = define shape of request and response

HTTPException = return proper error with status code
response_model = validates + shapes response automatically
AskRequest  = input shape  (what user sends)
AskResponse = output shape (what we send back)

uv run python -m uvicorn api:app --reload
This forces uv to use the correct environment instead of the conflicting one.

Swagger = automatic interactive documentation for your API.
When you open http://127.0.0.1:8000/docs you see this:
GET  /health  → click "Try it out" → click "Execute" → see real response
POST /ask     → click "Try it out" → type question → click "Execute" → see answer

Swagger (/docs) = auto-generated API documentation
- Shows all endpoints
- Shows request/response shapes  
- Lets you test directly from browser
- FastAPI generates it for free from your code
- Also called OpenAPI

Without Swagger → backend developer sends API to frontend developer
               → frontend has to guess what fields to send
               → lots of back and forth

With Swagger    → frontend opens /docs
               → sees exactly what to send and what comes back
               → no guessing, no back and forth

Loading pipeline for every request:
- Rebuild ChromaDB    → 30 seconds
- Load embeddings     → 10 seconds  
- Load CrossEncoder   → 5 seconds
- Total per request   → 45 seconds ← unusable

Loading once at startup:
- All of the above    → 45 seconds ONCE
- Every request after → 1-2 seconds ← fast!

Load pipeline ONCE at startup = eager loading
Benefits:
- First request is fast
- No rebuilding ChromaDB per request
- Global variables hold pipeline in memory
- Server pays the cost once, not every request

--reload flag means uvicorn detects your changes and restarts by itself.
Global variables = shared memory across all functions
local variables  = die when function ends
Pipeline stored globally = loaded once, used by every request
global keyword   = "use the hallway whiteboard, not my room one"

rag_pipeline.py  → defines hybrid_search_single() ONCE
api.py           → imports from rag_pipeline.py

2 endpoints 
health and ask

400 = client error — the user sent bad data (empty question, wrong format)
500 = server error — something broke on OUR side (pipeline crashed, LLM failed)

try/except in APIs = graceful error handling
Without it → server crashes, user sees ugly Python error
With it    → server catches error, returns clean JSON response
Server stays running even when one request fails

Day 5 Complete!
Built FastAPI server with:
- GET /health → server status check
- POST /ask   → takes question, returns RAG answer

Key concepts:
- FastAPI + Uvicorn = framework + server
- Pydantic models = request/response validation
- Global variables = pipeline loaded once at startup
- try/except = graceful error handling
- Swagger at /docs = free auto-generated docs

First real API call result:
POST /ask {"question": "What API was used?"}
→ {"answer": "Twitter API Tweepy."}
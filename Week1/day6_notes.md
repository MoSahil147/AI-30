Q1 — DRY = Don't Repeat Yourself
We violated it by copying the same search functions into both day4_rag.py AND api.py. If we fix a bug in one file, we have to remember to fix it in the other. The fix is a shared rag_pipeline.py that both files import from.
Q2 — detail vs details
FastAPI's HTTPException expects exactly detail — one specific parameter name. When you wrote details with an 's', FastAPI didn't recognise it and silently ignored the error message. Small typo, silent failure — exactly the kind of bug that's hard to find.
Q3 — Out of scope questions
If someone asks "What is the capital of France?" your RAG searches the fake accounts paper, finds vaguely related chunks, and the LLM either hallucinates or says "I don't know." Right now there's no guardrail. A proper system should detect when a question is out of scope and return a clean message instead of searching at all.

load_pipeline now takes pdf_path as a parameter instead of hardcoding the filename. Why is this better?
More precisely — reusability. If the path is hardcoded, this module only works for ONE specific PDF. By taking pdf_path as a parameter, any script can use this pipeline with ANY

Hardcoded = works for one case only
Parameterized = works for any case
Always prefer parameters over hardcoded values

200 = OK, success
400 = bad request (sender sent wrong data)
404 = not found (URL doesn't exist)
500 = internal server error (our code crashed)
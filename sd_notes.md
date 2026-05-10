Q1. 
100 users, each with their own documents, your current RAG API only handles one PDF loaded at startup. What breaks and how do you fix it?

Answer: 
If User A uploads contracts.pdf and User B uploads medical.pdf — they'd share the same ChromaDB. User A's question would search User B's documents. That's a data isolation problem, not a load problem.

Load balancer solves traffic distribution. But you don't even have per-user storage yet.

The correct order of thinking:

What's the core problem? → No user isolation
How do I store documents per user? → Separate ChromaDB per user, or namespaces
Now how do I handle 100 users simultaneously? → That's where load balancer comes in
What about cost/latency? → Caching, async processing

For your interview answer, say:
"First I'd add per-user document namespaces in ChromaDB so data doesn't mix. Then I'd move document processing to a background job queue so uploads don't block the API. Then load balancing across multiple API instances for traffic."

Concept: The Queue
Your current agent runs tasks synchronously — user sends request, waits, gets answer. For 1 user that's fine. For 100 users that's a problem.
A queue fixes this:
User sends request → goes into Queue → worker picks it up → processes → returns result
User doesn't wait, they get a job ID immediately
Real example: when you upload a video to YouTube, you don't wait 10 minutes staring at the screen. You get "processing" immediately. That's a queue.
Write in systems_design_notes.md:
Queue = decouple request from processing
User gets immediate response (job ID)
Worker processes in background
Use when: tasks take >2 seconds
Tools: Redis Queue, Celery, AWS SQS
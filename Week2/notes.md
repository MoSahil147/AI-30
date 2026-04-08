Multi-doc RAG challenges:
1. Source confusion - which chunk from which paper?
2. Metadata tracking - store source in chunk.metadata
3. Cross-doc questions - retrieve from multiple papers
Fix: metadata-aware chunking + source labels in prompt

we will do this
chunk.metadata = {
    "source": "attention.pdf",
    "page": 3
}

we will use a simple orded colection of 3 paths:
## Just a list!
pdf_paths = [
    "papers/attention.pdf",
    "papers/bert.pdf", 
    "papers/rag.pdf"
]
```

**When to use each:**
```
list  → ordered collection, can have duplicates → use for PDF paths
set   → unique values, no order → use for deduplication (like Week 1!)
dict  → key-value pairs → use for metadata, configs
```

You actually used all three in Week 1:
- `list` → chunks, unique_chunks
- `set` → seen (deduplication)
- `dict` → chunk_scores (RRF scoring)

**Write in notes.md:**
```
list = ordered collection → PDF paths, chunks
set  = unique values → deduplication  
dict = key-value pairs → scores, metadata

list1 = [1, 2, 3]

list1.append([4, 5])   ## → [1, 2, 3, [4, 5]]  ← nested!
list1.extend([4, 5])   ## → [1, 2, 3, 4, 5]    ← flat! ✅

Load multiple PDFs:
- loop through pdf_paths list
- loader.load() returns list of pages
- all_docs.extend(docs) adds flat, not nested
- metadata automatic: source + page per chunk
extend() = add items flat
append() = add as nested item

LangChain's split_documents automatically copies the parent document's metadata to every child chunk. So every chunk remembers which paper and page it came from — for free.
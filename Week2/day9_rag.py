# will store convo history now sir

# os ke upar kuch nahi!
import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np

load_dotenv()

# Block 2 load chunk index
if os.path.exists("chroma_db"):
    shutil.rmtree("chroma_db")
    
pdf_paths=[
    "papers/attention.pdf",
    "papers/bert.pdf",
    "papers/rag.pdf"
]

all_docs=[]
for path in pdf_paths:
    loader=PyPDFLoader(path)
    docs=loader.load()
    all_docs.extend(docs)
    print(f"Loaded {path}: {len(docs)} pages")

print(f"Total pages: {len(all_docs)}")

splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks=splitter.split_documents(all_docs)

seen=set()
unique_chunks=[]
for chunk in chunks:
    if chunk.page_content not in seen:
        seen.add(chunk.page_content)
        unique_chunks.append(chunk)
        
print(f"Unique chunks: {len(unique_chunks)}")

embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore=Chroma.from_documents(
    documents=unique_chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)
#bm25 ke liye
tokenized_chunks=[
    chunk.page_content.lower().split()
    for chunk in unique_chunks
]
bm25=BM25Okapi(tokenized_chunks)

# decleread Cross here
reranker=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
llm=ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0
)
print("Pipeline Ready")

# block 3 (hybrid search + rerank)
def hybrid_search(query, k=10):
    vector_results=vectorstore.similarity_search(query, k=10)
    
    tokenized_query=query.lower().split()
    bm25_scores=bm25.get_scores(tokenized_query)
    top_bm25_indices=np.argsort(bm25_scores)[::-1][:10]
    bm25_results=[unique_chunks[i] for i in top_bm25_indices]
    
    chunk_scores={}
    for rank, chunk in enumerate(vector_results):
        content=chunk.page_content
        if content not in chunk_scores:
            chunk_scores[content]=0
        chunk_scores[content]+=1/(rank+60)
        
    for rank, chunk in enumerate(bm25_results):
        content=chunk.page_content
        if content not in chunk_scores:
            chunk_scores[content] =0
        chunk_scores[content]+=1/(rank+60)
    
    sorted_contents=sorted(
        chunk_scores.keys(),
        key=lambda x:chunk_scores[x],
        reverse=True
    )[:k]
    
    result_chunks=[]
    for content in sorted_contents:
        for chunk in unique_chunks:
            if chunk.page_content==content:
                result_chunks.append(chunk)
                break
    return result_chunks

def rerank(query, chunks, top_k=3):
    pairs=[[query, chunk.page_content] for chunk in chunks]
    scores=reranker.predict(pairs)
    ranked_indices=np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in ranked_indices]
    
    
# block 4: now histiyr coes here
conversation_history=[]

def answer_with_memory(question):
    # do hybrid search first
    candidates=hybrid_search(question, k=10)
    # find best chunks
    best_chunks=rerank(question, candidates, top_k=3)
    
    context_parts=[]
    for chunk in best_chunks:
        source=chunk.metadata.get("source", "unknown")
        page=chunk.metadata.get("page", "?")
        context_parts.append(
            f"[Source: {source}, Page: {page}]\n {chunk.page_content}"
        )
    context="\n\n".join(context_parts)
    
    history_text=""
    for turn in conversation_history:
        role=turn["role"].upper()
        history_text+=f"{role}: {turn['content']}\n"
    
    prompt=f"""You are a research assistant. Answer using the context below.
When referring to previous questions, use the conversation history.
If the answer is not in the context, say "I don't know".
Mention which paper the information came from.

Conversation history:
{history_text}

Context:
{context}

Question: {question}
Answer:"""

    response = llm.invoke(prompt)
    answer = response.content
    
    conversation_history.append({"role":"user", "content":question})
    conversation_history.append({"role":"assistant", "content":answer})
    
    return answer

# test 

print("\n----- Convo testing ------")

q1="What is the attention mechanism?"
a1=answer_with_memory(q1)
print(f"Q1: {q1}")
print(f"A1: {a1}\n")

q2 = "How is that used differently in BERT?"
a2 = answer_with_memory(q2)
print(f"Q2: {q2}")
print(f"A2: {a2}")
    
        
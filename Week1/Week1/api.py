### ITSSS day5 we are making APII

## os se upar kuch nahiii
import os 
import sys
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

load_dotenv()

app = FastAPI(
    title="RAG API",
    description="Ask question about the research papers, lets test my knowledge",
    version="1.0.0"
)

# define what the user SENDS us
class AskRequest(BaseModel):
    question:str

# define what we send back
class AskResponse(BaseModel):
    question:str
    answer:str
    
# health check endpoint
@app.get("/health")
def health_check():
    return {"status":"ok","message":"RAG API is running"}

# ask endpoint - main functionnn
@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    # will validate the inputs and will not accept empty questions
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    try:
        # same running the prev day4 pipeline
        # request se jo question aa raha hai usse we are using as the query
        candidates = expanded_hybrid_search(request.question, k=10)
        best_chunks=rerank(request.question, candidates, top_k=3)
        
        # build the context from chunks
        context="\n\n".join([chunk.page_content for chunk in best_chunks])
        
        # build prompt
        prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {request.question}
Answer:"""
        # call the daddy LLM directly
        response = llm.invoke(prompt)
    
        return AskResponse(
        question=request.question,
        answer=response.content
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}"
        )
    
# ## (test : curl -X POST http://127.0.0.1:8000/ask \
#   -H "Content-Type: application/json" \
#   -d '{"question": "What API was used to collect Twitter data?"}')   
        
   
# rag prep 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
from pydantic import Field
import numpy as np
import shutil

# global variable
# load once when server starts and reuse every request
# pehle load kare phir vishwash kare
vectorstore=None
bm25=None
unique_chunks=None
reranker=None
qa_chain=None

def load_pipeline():
    """Load the full RAG pipeline - called once at the startup"""
    global vectorstore, bm25, unique_chunks, reranker, qa_chain
    
    print("Loading RAG pipeline....")
    
    loader = PyPDFLoader("Paper/Detection_of_Fake_Accounts_on_Social_Media_Using_Multimodal_Data_With_Deep_Learning.pdf")
    docs = loader.load()
    
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    
    # now remove dups dups
    seen = set()
    unique_chunks=[]
    for chunk in chunks:
        if chunk.page_content not in seen:
            seen.add(chunk.page_content)
            unique_chunks.append(chunk)
            
    # build vector store, bhain kahi tho store karna hai nah
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")
    
    embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=unique_chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    
    # build the BM25 index
    tokenized_chunks = [
        chunk.page_content.lower().split()
        for chunk in unique_chunks
    ]
    # we did the firsts tep of tokeinsize, small case
    bm25=BM25Okapi(tokenized_chunks)
    
    # load reranker
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # load LLM 
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    print("Pipeline loaded successfullyyy!!!")
    return llm

llm=load_pipeline()

# Block 3 - Search functions
# same hybrid search + rerankung them

def hybrid_search_single(query, k=10):
    vector_results=vectorstore.similarity_search(query, k=10)
    
    tokenized_query=query.lower().split()
    bm25_scores=bm25.get_scores(tokenized_query)
    top_bm25_indices=np.argsort(bm25_scores)[::-1][:10]
    bm25_results=[unique_chunks[i] for i in top_bm25_indices]
    
    chunk_scores={}
    for rank, chunk in enumerate(vector_results):
        content = chunk.page_content
        if content not in chunk_scores:
            chunk_scores[content]=0
        chunk_scores[content]+=1/(rank+60)
    
    for rank, chunk in enumerate(bm25_results):
        content=chunk.page_content
        if content not in chunk_scores:
            chunk_scores[content] = 0
        chunk_scores[content]+=1/(rank+60)
    
    sorted_contents=sorted(
        chunk_scores.keys(),
        key=lambda x: chunk_scores[x],
        reverse=True
    )[:k]
    
    result_chunks=[]
    for content in sorted_contents:
        for chunk in unique_chunks:
            if chunk.page_content==content:
                result_chunks.append(chunk)
                break
            
    return result_chunks

def expand_query(question):
    prompt=f"""Generate 3 different search queries for this question.
Each query should approach the topic from a completely different angle.
Use different keywords, technical terms and perspectives.
Return ONLY the 3 queries, one per line, no numbering, no explanation.

Question: {question}
"""
    response=llm.invoke(prompt)
    queries=response.content.strip().split("\n")
    queries=[q.strip() for q in queries if q.strip()]
    # returning question along with the query we got
    return [question] + queries[:3]

def expanded_hybrid_search(question, k=10):
    queries=expand_query(question)
    all_chunk_scores={}
    
    for query in queries:
        results = hybrid_search_single(query, k=10)
        for rank, chunk in enumerate(results):
            content = chunk.page_content
            if content not in all_chunk_scores:
                all_chunk_scores[content]=0
            all_chunk_scores[content]+=1/(rank+60)
            
    sorted_contents= sorted(
        all_chunk_scores.keys(),
        key=lambda x: all_chunk_scores[x],
        reverse=True
    )[:k]
    
    result_chunks=[]
    for content in sorted_contents:
        for chunk in unique_chunks:
            if chunk.page_content == content:
                result_chunks.append(chunk)
                break
            
    return result_chunks

def rerank(query, chunks, top_k=3):
    pairs=[[query, chunk.page_content] for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked_indices=np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in ranked_indices]


# this will be a shared pipelne used by both api.py and everyday scirpts
# DRY principle: define once and import everywhere

# nothing above os
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
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
from pydantic import Field
import numpy as np

load_dotenv()

# global pipeline comps
vectorstore=None
bm25=None
unique_chunks=None
reranker=None
llm=None

def load_pipeline(pdf_path: str):
    global vectorstore, bm25, unique_chunks, reranker, llm
    
    print(".... Loading RAG PIPElinee.....")
    
    # 1 loader and doc
    loader=PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # 2 Split
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks=splitter.split_documents(docs)
    
    # 3 sirf unique picks!!
    seen = set()
    unique_chunks=[]
    for chunk in chunks:
        if chunk.page_content not in seen:
            seen.add(chunk.page_content)
            unique_chunks.append(chunk)
       
    # 4 chromadb should be empty     
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")
    
    # 5 add embeddings and vectorstore    
    embeddings= HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore=Chroma.from_documents(
        documents=unique_chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    
    # 6 toknize for the bm25 (keywords)
    tokenized_chunks=[
        chunk.page_content.lower().split()
        for chunk in unique_chunks
    ]
    bm25=BM25Okapi(tokenized_chunks)
    reranker=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # 7 llm sir
    llm=ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )
    
    print("Pipeline loadeddd successfullyyy")
    return llm

# time for hybrid search nowww
def hybrid_search_single(query, k=10):
    # first find similarity
    vector_results=vectorstore.similarity_search(query, k=10)
    # tokenize for BM25
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
            chunk_scores[content]=0
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
    
# now generating different questions to cross check
def expand_query(question):
    prompt=f"""Generate 3 different search queries for this question.
Each query should approach the topic from a completely different angle.
Use different keywords, technical terms and perspectives.
Return ONLY the 3 queries, one per line, no numbering, no explanation.

Question: {question}"""
    response=llm.invoke(prompt)
    queries=response.content.strip().split("\n")
    queries=[q.strip() for q in queries if q.strip()]
    return [question] + queries[:3]
    
def expanded_hybrid_search(question, k=10):
    queries=expand_query(question)
    all_chunk_scores={}
    
    for query in queries:
        results=hybrid_search_single(query, k=10)
        for rank, chunk in enumerate(results):
            content=chunk.page_content
            if content not in all_chunk_scores:
                all_chunk_scores[content]=0
            all_chunk_scores[content]+=1/(rank+60)
    
    sorted_contents=sorted(
        all_chunk_scores.keys(),
        key=lambda x:all_chunk_scores[x],
        reverse=True
    )[:k]
    
    result_chunks=[]
    for content in sorted_contents:
        for chunk in unique_chunks:
            if chunk.page_content==content:
                result_chunks.append(chunk)
                break
    return result_chunks
    
def rerank(query, chunks,top_k=3):
    pairs=[[query, chunk.page_content] for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked_indices=np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in ranked_indices]
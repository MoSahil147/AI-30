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
    return AskResponse(
        question=request.question,
        answer="Pipeline not loaded yet"
    )
   
   
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
    
    # buld the BM25 index
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

# Block 3
# this will be a shared pipelne used by both api.py and everyday scirpts
# DRY principle: define once and import everywhere

from day2_rag import unique_chunks
from api import vectorstore, rerank
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
    global vectorstore, bm25, 
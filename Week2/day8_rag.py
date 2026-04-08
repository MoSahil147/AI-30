# Goal is that we will search accross 3 AI papers simultaneously
# metadata tracking every chunk knows its source

# os ke upare kuch nahi
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

# pehle env le
load_dotenv()

# remove if some values exists pehle se db mein
if os.path.exists("chroma_db"):
    shutil.rmtree("chroma_db")
    
# after cleanace load pdf
# pdf paths
pdf_paths=[
    "papers/attention.pdf",
    "papers/bert.pdf",
    "papers/rag.pdf"
]

# load them
all_docs=[]
for path in pdf_paths:
    loader=PyPDFLoader(path)
    docs=loader.load()
    all_docs.extend(docs) # add to master list
    print(f"Loaded {path}: {len(docs)} pages")
    
print(f"Total pages across all papers: {len(all_docs)}")

# Block 2
# split and store the chunk

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks=splitter.split_documents(all_docs)

# unqiue chunks will be stored
seen = set()
unique_chunks=[]
for chunk in chunks:
    if chunk.page_content not in seen:
        seen.add(chunk.page_content)
        unique_chunks.append(chunk)
        
print(f"Total chunks: {len(chunks)}")
print(f"Unique chunks: {len(unique_chunks)}")

# Metacheck
print("\n-----MetaBhai-----")
for chunk in unique_chunks[:5]:
    source=chunk.metadata.get("source", "unknown")
    page=chunk.metadata.get("page","?")
    print(f"Source: {source} | Page: {page} | Preview: {chunk.page_content[:50]}........")
    
# Block 3: vector + BM25\

print("\n Building vector store.....")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore=Chroma.from_documents(
    documents=unique_chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)
print(f"vector store: {len(unique_chunks)} chunks indexed")

# BM25
tokenized_chunks=[
    chunk.page_content.lower().split()
    for chunk in unique_chunks
]
bm25=BM25Okapi(tokenized_chunks)
print(f"BM25: {len(unique_chunks)} chunks indexed")

# okay after getting similairty and key word that is BM25
# get reranker 
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
llm=ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

print("MKC readyyyy!!!")

# Block4 
# hybrid search + metadata aware answering
def hybrid_search(query, k=10):
    # first will store the similarity!!
    vector_results=vectorstore.similarity_search(query, k=10)
    
    #now bm25
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

def rerank(query, chunks, top_k=3):
    pairs=[[query, chunk.page_content] for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked_indices=np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in ranked_indices]

def answer_with_sources(question):
    candidates=hybrid_search(question, k=10)
    best_chunks=rerank(question, candidates, top_k=3)
    
    # context with source labels
    context_parts=[]
    for chunk in best_chunks:
        source=chunk.metadata.get("source", "unknown")
        page=chunk.metadata.get("page", "?")
        context_parts.append(
            f"[Source: {source}, Page: {page}]\n{chunk.page_content}"
        )
    context="\n\n".join(context_parts)
    
    prompt = f"""Answer the question using only the context below.
Each chunk is labeled with its source paper and page.
If the answer is not in the context, say "I don't know".
When answering, mention which paper the information came from.

Context:
{context}

Question: {question}
Answer:"""

    response = llm.invoke(prompt)
    return response.content, best_chunks

# test
question="What is the attention mecha?"
answer, sources=answer_with_sources(question)
print(f"\nQuestion: {question}")
print(f"\nAnswer: {answer}")
print(f"\n----Sources used---")
for chunk in sources:
    print(f"{chunk.metadata['source']} | Page {chunk.metadata['page']}") 
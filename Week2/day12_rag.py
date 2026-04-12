# Gradio day

# os ke upar kuch nahi Jassii
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
import gradio as gr

load_dotenv()

# blcok2 (load+chunk+index)
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
    docs = loader.load()
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

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore=Chroma.from_documents(
    documents=unique_chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)
tokenized_chunks=[
    chunk.page_content.lower().split()
    for  chunk in unique_chunks
]
bm25=BM25Okapi(tokenized_chunks)

reranker=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
llm=ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0
)
print("Pipeline Ready!")

# block 3
def hybrid_search(query, k=10):
    vector_results=vectorstore.similarity_search(query, k=10)
    
    # now bm25
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
        content = chunk.page_content
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

def rerank(query, chunks, top_k=3):
    pairs=[[query, chunk.page_content] for chunk in chunks]
    scores=reranker.predict(pairs)
    ranked_indices=np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in ranked_indices]

# blcok 4
conversation_history=[]

def rewrite_query(question):
    if not conversation_history:
        return question
    
    history_text=""
    for turn in conversation_history:
        role=turn['role'].upper()
        history_text+=f"{role}: {turn['content']}\n"
    
    rewrite_prompt=f"""Given this conversation history and a vague follow-up question,
rewrite the follow-up question to be a clear standalone question.
Only output the rewritten question, nothing else.

Conversation history:
{history_text}

Vague question: {question}
Rewritten question:"""

    response=llm.invoke(rewrite_prompt)
    return response.content.strip()

def chat(message, history):
    # rewrite vague question
    clear_question=rewrite_query(message)
    
    # search with clear question
    candidates = hybrid_search(clear_question, k=10)
    best_chunks=rerank(clear_question, candidates, top_k=3)
    
    # build conetxt
    context_parts=[]
    for chunk in best_chunks:
        source=chunk.metadata.get("source", "unknown")
        page=chunk.metadata.get("page", "?")
        context_parts.append(
            f"[Source: {source}, Page: {page}]\n{chunk.page_content}"
        )
    context="\n\n".join(context_parts)
    
    history_text=""
    for turn in conversation_history:
        role=turn["role"].upper()
        history_text+=f"{role}: {turn['content']}\n"
    
    prompt=f"""You are a research assistant. Answer using the context below.
If the answer is not in the context, say "I don't know".
Mention which paper the information came from.

Conversation history:
{history_text}

Context:
{context}

Question: {message}
Answer:"""

    # streaming Gradio
    answer=""
    for chunk in llm.stream(prompt):
        answer+=chunk.content
        yield answer
    
    # save to history
    conversation_history.append({"role": "user", "content": message})
    conversation_history.append({"role": "assistant", "content": answer})
    
# Launch Gradio
demo=gr.ChatInterface(
    fn=chat,
    title="AI Research Assistant",
    description="Ask question about Attention, BERT and RAG papers!"
)

demo.launch()


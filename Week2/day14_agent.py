# agent with web search
# rag fr papers and web search for everythinh 
#

# os ke upar kuh nahi
import os
import shutil
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from ddgs import DDGS
import numpy as np

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

print("LLM Readyyy!!!!")

# block2 Load RAG pipeline
if os.path.exists("chroma_db"):
    shutil.rmtree("chroma_db")
    
pdf_paths = [
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
        
embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore=Chroma.from_documents(
    documents=unique_chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

# bm25 ke liye tokenisze
tokenized_chunks=[
    chunk.page_content.lower().split()
    for chunk in unique_chunks
]
bm25=BM25Okapi(tokenized_chunks)

# chunk and question pair ke lyr, reranker
reranker=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("RAG Pipeline is ReadyyY!!!")

# block 3- Definining tools
@tool
def rag_search(query:str)->str:
    """Search through the attention, BERT and RAG research papers.
    Use this for questions about these specific AI papers, their methods,
    architectures, or technical details."""
    
    vector_results=vectorstore.similarity_search(query, k=10)
    
    # now bm25 ki baari
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
        key=lambda x: chunk_scores[x],
        reverse=True
        )[:5]
    
    results=[]
    for content in sorted_contents:
        for chunk in unique_chunks:
            if chunk.page_content==content:
                source=chunk.metadata.get("source", "unknown")
                page=chunk.metadata.get("page", "?")
                results.append(f"[{source}, Page {page}]: {content[:200]}")
                break
    
    return "\n\n".join(results)

@tool
def web_search(query: str)->str:
    """Search the internet for current information, news, or anything
    not covered in the research papers. Use for general knowledge questions."""
    
    try:
        with DDGS() as ddgs:
            results=list(ddgs.text(query, max_results=3))
        
        if not results:
            return "No results found"
        
        output=[]
        for r in results:
            output.append(f"Title: {r['title']}\n Summary: {r['body']}")
            
        return "\n\n".join(output)
    except Exception as e:
        return f"Web Search Error: {e}"
        
tools=[rag_search, web_search]
print(f"Tools ready: {[t.name for t in tools]}")

# block 4 create agent and testing
prompt=PromptTemplate.from_template("""You are a helpful research assistant.
                                    
{tools}

IMPORTANT: You MUST use exactly this format, spelling THOUGH correctly:

Question: {input}
Thought: (think about which tool to use)
Action: (one of [{tool_names}])
Action Input: (input for the tool)
Observation: (tool result)
Thought: I now know the final answer
Final Answer: (your answer)


{agent_scratchpad}""")

agent = create_react_agent(llm, tools, prompt)
agent_executor=AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)
print("Agent Ready!")

## Test 1 - should use RAG
print("\n--- Test 1: RAG question ---")
result = agent_executor.invoke({"input": "What is self-attention in transformers?"})
print(f"Answer: {result['output']}")

## Test 2 - should use web search
print("\n--- Test 2: Web question ---")
result = agent_executor.invoke({"input": "What is the latest version of Python?"})
print(f"Answer: {result['output']}")

## Test 3 - agent decides!
print("\n--- Test 3: Agent decides ---")
result = agent_executor.invoke({"input": "How does BERT compare to GPT in 2024?"})
print(f"Answer: {result['output']}")


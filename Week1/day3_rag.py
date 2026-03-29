## Goal today is to imporve the contextracll fro 0.8 using corss encoder ranking
## hybrid search like kal -> rerankk -> top3 -> LLM

## Block 1
import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from rank_bm25 import BM25Okapi
## new commer today
from sentence_transformers import CrossEncoder
from ragas.metrics import faithfulness, context_recall
from ragas import evaluate
from ragas.embeddings import HuggingFaceEmbeddings as RagasHuggingfaceEmbeddings
from datasets import Dataset
import numpy as np

## Block 2
## alswyas overlap 10% of the chunks size 

load_dotenv()

## bhai delete pehle so hai chromadb mein
if os.path.exists("chroma_db"):
    shutil.rmtree("chroma_db")
    
## first load
loader=PyPDFLoader("Paper/Detection_of_Fake_Accounts_on_Social_Media_Using_Multimodal_Data_With_Deep_Learning.pdf")
docs=loader.load()
print(f"Loaded {len(docs)} pages")

## now split split
splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)
print(f"Chunks {len(chunks)} itne hai Loaded")

## now will store unique and acche wale
seen = set()
unique_chunks=[]
for chunk in chunks:
    if chunk.page_content not in seen:
        seen.add(chunk.page_content)
        unique_chunks.append(chunk)

print(f"Unique chunks: {len(unique_chunks)}")

## now vector store
embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore=Chroma.from_documents(
    documents=unique_chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)
print(f"Vector Store: {vectorstore._collection.count()} chunks stored")

## now BM25 index
tokenized_chunks=[
    chunk.page_content.lower().split() ## jo chunk hai uska lowercase karenge and then will split
    for chunk in unique_chunks
]
bm25=BM25Okapi(tokenized_chunks)
print(f"BM25 index: {len(tokenized_chunks)} chunks indexed")


## Block 3
## crossencoder + hybrid search with reranking

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
print("Cross encoder loaded!")

## hybrid search function same as day2
def hybrid_search(query, k=10):
    ## vector search is find my meaninggg
    vector_results=vectorstore.similarity_search(query, k=10)
    
    ## Ladies and Gentleman you are not ready for this
    ## her comes BM25
    tokenized_query=query.lower().split()
    bm25_scores=bm25.get_scores(tokenized_query)
    top_bm25_indices=np.argsort(bm25_scores)[::-1][:10]
    bm25_results=[unique_chunks[i] for i in top_bm25_indices]
    
    ## RRF merge
    chunk_scores={}
    for rank, chunk in enumerate(vector_results):
        content = chunk.page_content
        if content not in chunk_scores:
            chunk_scores[content]=0 ## agar yeh contenet chunk score mein nahi hai tho we will use this to add in scores
        chunk_scores[content] +=1/(rank+60)
        
    for rank, chunk in enumerate(bm25_results):
        content=chunk.page_content
        if content not in chunk_scores:
            chunk_scores[content]=0 ## will set to 0, agar score nahi hai
        chunk_scores[content] += 1/(rank+60)
    
    sorted_contents = sorted (
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

## reranking now

def rerank(query, chunks, top_k=3):
    ## this si pairs
    ## pair is (question + chunks) isko hi tho rank karte hai ham neeche, reranker se
    pairs=[
        [query, chunk.page_content]
        for chunk in chunks
    ]
    
    ## reranker.predict(pairs) returns a list of relevance scores — one float number per pair:
    scores = reranker.predict(pairs)
    
    ## Reranker sorts highest to lowest directly, its argssort that gives lowest to highest, then we reveerse its using [::-1]
    ranked_indices=np.argsort(scores)[::-1][:top_k]
    
    ##return top chunks
    return [chunks[i] for i in ranked_indices]
print("Hybrid search + reranker ready!")
    
## Block 4 LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

## testing 
test_query="What API was used to collecet the Twitter Data?"
print(f"\n Testing full pipeline: {test_query}")

## 1st hybrid search
## will get 10 candiastea susing BM25 + vector + RRF
candidates=hybrid_search(test_query, k=10)
print(f"Hybrid search found: {len(candidates)} candidates")


## 2nd step renak and get the top3
##scores the 10 what we got and will return the best 3
best_chunks=rerank(test_query, candidates, top_k=3)
print(f"Reranker selected: {len(best_chunks)} chunks")

## Show them.... what we got
print("\n-----RERANKED TOP 3 CHUNKS------")
for i, chunk in enumerate(best_chunks):
    print(f"\nChunk {i+1}:")
    print(chunk.page_content[:200]) ## will show first 200 chars for debug, rember the LLM gets full chunk
    print("----")
    
    ## 157 chunks → hybrid search → 10 candidates → reranker → 3 final chunks
    
## Block 5

## conecting Reranker to RetrievalQA
## will wrap the function in LnagChain retriver calss
## now get relevant docs function will call rerank after hybrind search
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
from pydantic import Field

class RerankingRetriever(BaseRetriever):
    ## kitan candiates the hybrid search will fetch
    candidates_k:int=Field(default=10)
    ## last mein kitne chunk after reranking
    final_k:int=Field(default=3)
    
    def _get_relevant_documents(self, query:str) -> List[Document]:
        ## step1: hybrid search se will fetch candidates
        candidates=hybrid_search(query, k=self.candidates_k)
        ## Step2:rerank picks the best final_k
        return rerank(query, candidates, top_k=self.final_k)
    
## create retriver instance
reranking_retriever = RerankingRetriever(
    candidates_k=10,
    final_k=3
)
 
## now the RAG chain with reranking retriever
qa_chain=RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=reranking_retriever
)   

## testing testing
question="What API was used to collect Twitter data?"
answer=qa_chain.invoke(question)
print(f"\nQuestion: {question}")
print(f"Answer: {answer['result']}")
    

## Block 6: RAGAS evaluation
## will use Ground Truth

test_data = [
    {
        "question": "What F1 score did the proposed model achieve?",
        "ground_truth": "The proposed model achieved an F1 score of 0.96"
    },
    {
        "question": "How many total accounts were in the dataset?",
        "ground_truth": "The dataset contains 12,355 accounts"
    },
    {
        "question": "How many genuine accounts were in the dataset?",
        "ground_truth": "There were 8,267 genuine accounts"
    },
    {
        "question": "What API was used to collect Twitter data?",
        "ground_truth": "Twitter API called Tweepy was used"
    },
    {
        "question": "What deep learning models were used?",
        "ground_truth": "CNN for visual data, LSTM for textual data, GCN for network-based data"
    },
]

eval_data={
    "question":[],
    "answer":[],
    "contexts":[],
    "ground_truth":[]
}

print("\n---- Day 3 RAGAS Eval ------")
for item in test_data:
    q=item["question"]
    gt=item["ground_truth"]
    
    result=qa_chain.invoke(q)
    
    ## get reranked chunks for thsi question
    candidates=hybrid_search(q, k=10)
    reranked=rerank(q, candidates, top_k=3)
    
    eval_data["question"].append(q)
    eval_data["answer"].append(result["result"])
    eval_data["contexts"].append(
        [chunk.page_content for chunk in reranked]
    )
    eval_data["ground_truth"].append(gt)
    print(f"Done: {q[:50]}...")
    
dataset=Dataset.from_dict(eval_data)

ragas_embeddings=RagasHuggingfaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, context_recall],
    llm=llm,
    embeddings=ragas_embeddings
)

print("\n ---- DAY 3 RAGAS -----")
print(results)

print("\n--- FULL COMPARISON ---")
print(f"Day 1 — faithfulness: 0.8135 | context_recall: N/A")
print(f"Day 2 — faithfulness: 0.9600 | context_recall: 0.8000")
print(f"Day3 - faithfulness: {results['faithfulness']} | context_recall: {results['context_recall']}")
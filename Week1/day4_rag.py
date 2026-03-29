## Day 4 - Query Expansion RAG
## Goal: fix Q5 context_recall 0.0 using query expansion
## Pipeline: question → LLM generates 3 variations → 
##           search each → RRF merge → rerank → LLM answers

##mere ghar pe sirf mere logo allowed, s ke upar walo ko thop dunga
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
from sentence_transformers import CrossEncoder
from ragas.metrics import faithfulness, context_recall
from ragas import evaluate
from ragas.embeddings import HuggingFaceEmbeddings as RagasHuggingfaceEmbeddings
from datasets import Dataset
import numpy as np

## env loading 
load_dotenv()

## agar koi values db mein hoga, tho nikal saale pehli fursat mein
if os.path.exists("chroma_db"):
    shutil.rmtree("chroma_db")


## now all clear load the PDF
loader = PyPDFLoader("Paper/Detection_of_Fake_Accounts_on_Social_Media_Using_Multimodal_Data_With_Deep_Learning.pdf")
## load hone ke baad doc mein chnage kar
docs = loader.load()
print(f"Loaded {len(docs)} itne pages haiii")


### Now splitttttt
splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks=splitter.split_documents(docs)
print(f"Chunks {len(chunks)} itne hai chunkss kalia")

## abhi apun ko sirf unique maal magta hai, no old
seen = set() ## yeh hash use karta hai search karne, so mast maal hai, because uses hash
unique_chunks=[]
for chunk in chunks:
    if chunk.page_content not in seen:
        ## chunks ka agar page conetet nahi hai seen mein pahele se tho add karne ka, tension nako
        seen.add(chunk.page_content)
        ## pehle se nahi tha seen par so seen pe bhi daala and then unique mein bhi daala
        unique_chunks.append(chunk)
        
print(f"Unique chunks: {len(unique_chunks)} itne hai unique mast maal, doubt nako")


## abhiii vectorr ki baariii, so will embed first and then store
## beause we need to convert everything to vectorrs righttt!
## Yeh bro continue
embeddings= HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore=Chroma.from_documents(
    documents=unique_chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)
print(f"Vector Store: {vectorstore._collection.count()} chunks stored")


## now BM25 index
## for that we neeeddddd to tokenixed everything
tokenized_chunks=[
    ## jo bhi chunks hai lowercase that and then splittt
    chunk.page_content.lower().split()
    for chunk in unique_chunks
]
bm25=BM25Okapi(tokenized_chunks)
print(f"BMO25 index: {len(tokenized_chunks)}")

## Block 3 (CrorssEncoder + Query Expansion)
## Cross encoder was reading question and answer together!

reranker=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Attention everyone!!! Cross Encoder Loaded!!!!")


## llm
load_dotenv()
llm= ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

## here comes to hero to recscue Context_Recall

def expand_query(question):
    ## we need to give a mindblwoing prompt, for the LLM to generate different anges of queires
    ## not synonyms, geniunely different search angles
    prompt= f"""Generate 3 different search queries for this question.
    Each query should approach the topic from a completely different angle.
    Use different keywords, technical terms and perspectives.
    Return ONLY the 3 queries, one per line, no numbering, no explanation.
    
Question: {question}"""

    ## will call groq directky, remeber RAG se nahi bulanaa bhai
    respone=llm.invoke(prompt)
    
    ## now slit response into 3 queries
    ## upar response se we will get 3 queries, usko dodna hai bass
    queries = respone.content.strip().split("\n")
    
    ## cleaning process, clean up empty lines
    queries = [q.strip() for q in queries if q.strip()]
    
    ## also original question should also be inlcuded, warna what is the point beta without baap
    all_queries = [question] + queries[:3]
    
    print(f"We expanded the orginal question: '{question[:40]}...' into {len(all_queries)} queries")
    return all_queries

print('Query Expansion Ready!!!')
    
    
## Blcok 4: Hybrid Search + query expansion + reranking
## This si the main thing! Dont ignore

def hybrid_search_single(query, k=10):
    ## day3 hybrid search
    ## this is for the similarity test
    vector_results = vectorstore.similarity_search(query, k=10)
    
    ## we reuqire the below for BM25, tokenize everything
    tokenized_query= query.lower().split()
    bm25_scores=bm25.get_scores(tokenized_query)
    top_bm25_indices=np.argsort(bm25_scores)[::-1][:10]
    ## making sure that these unique chunks are used to top indices
    bm25_results=[unique_chunks[i] for i in top_bm25_indices]
    
    chunk_scores = {}
    
    for rank, chunk in enumerate(vector_results):
        content = chunk.page_content
        if content not in chunk_scores:
            chunk_scores[content] =0
        chunk_scores[content] += 1/(rank+60)
    
    for rank, chunk in enumerate(bm25_results):
        content = chunk.page_content
        if content not in chunk_scores:
            chunk_scores[content] =0
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

def expanded_hybrid_search(question, k=10):
    ## pehla step: will geerate query variations
    ## expand query is the prompt
    queries= expand_query(question)
    
    ## Step 2, will search eith each Variation and accumulate the RRF scores
    ## if a chunks is apperaing in multiple query it meannns = higher score
    all_chunk_scores ={}
    
    for query in queries:
        results = hybrid_search_single(query, k=10)
        
        for rank, chunk in enumerate(results):
            content = chunk.page_content
            if content not in all_chunk_scores:
                all_chunk_scores[content]=0
            ## each appeaencce adds to score of 4 queries... therefre we will have 4x more score
            all_chunk_scores[content]+=1/(rank+60)
    
    ## step3: now sort time
    sorted_contents= sorted(
        all_chunk_scores.keys(),
        key=lambda x: all_chunk_scores[x],
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
    ## cross encoder will read the question + chunk together
    pairs=[[query, chunk.page_content] for chunk in chunks]
    scores=reranker.predict(pairs)
    ranked_indices=np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in ranked_indices]

print("Full pipeline ready!!!!")
    


## Block 5 connecting RetrievalQA + testinnnggg

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
from pydantic import Field


class QueryExpansionRetriever(BaseRetriever):
    candidates_k: int = Field(default=10)
    final_k: int=Field(default=3)
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        ## the expanded search will get us candidates accross 4 qery variationss
        candidates=expanded_hybrid_search(query, k=self.candidates_k)
        ## rerankk picks the besttt3
        return rerank(query, candidates, top_k=self.final_k)
    
retriever = QueryExpansionRetriever(candidates_k=10, final_k=5)

qa_chain=RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

## now test theee legendary questionnnn, that needs a comeback, last time we got zero, but we need someeeethingggg nowww
## Comeeeeon
question="What deep learning models were used to detect fake accounts?"
answer=qa_chain.invoke(question)
print(f"\nQuestion: {question}")
print(f"Answer: {answer['result']}")

# The answer missed LSTM, we need RAGAAASSS, thoda issue ho gaya,pure answer nahi aaya

# Block 6: Ragas

test_data=[
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
        "ground_truth": "CNN for visual features, GCN for network-based features, and a unified deep learning architecture combining textual visual and network data"
    },
]

eval_data={
    "question":[],
    "answer":[],
    "contexts":[],
    "ground_truth":[]
}

print("\n----- Q5 ka Mustakbil abhi RAGAS decide karegaaa... AHHHHHHH ----")
for item in test_data:
    q = item["question"]
    gt=item["ground_truth"]
    
    result=qa_chain.invoke(q)
    
    ## will get contexts - expanded search + rerank
    candidates = expanded_hybrid_search(q, k=10)
    reranked = rerank(q,candidates, top_k=3)
    
    eval_data["question"].append(q)
    eval_data["answer"].append(result["result"])
    eval_data["contexts"].append(
        [chunk.page_content for chunk in reranked]
    )
    eval_data["ground_truth"].append(gt)
    print(f"Done: {q[:50]}.... tereko kya laga pura reveal karunga mein, panduuu")
    
dataset=Dataset.from_dict(eval_data)

## Raga bhai ko thanda karne chaiye hamko uska embeddings
ragas_embeddings=RagasHuggingfaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

results=evaluate(
    dataset=dataset,
    metrics=[faithfulness, context_recall],
    llm=llm,
    embeddings=ragas_embeddings
)

print("\n--- DAY 4 RAGAS SCORES, Kachra ab sab tumahre hattho hai... hamko aakhri gend pe 6 run chahii ---")
print(results)
print("\n--- FULL COMPARISON ---")
print(f"Day 1 -> faithfulness: 0.8135 | context_recall: N/A")
print(f"Day 2 -> faithfulness: 0.9600 | context_recall: 0.8000")
print(f"Day 3 -> faithfulness: 1.0000 | context_recall: 0.8000")
print(f"Day 4 -> faithfulness: {results['faithfulness']} | context_recall: {results['context_recall']}")
    
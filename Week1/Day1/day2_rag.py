## day2 HYBRID Search


## Block1
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
from ragas.metrics import faithfulness, context_recall
from ragas import evaluate
from ragas.embeddings import HuggingFaceEmbeddings as RagasHuggingfaceEmbeddings
from datasets import Dataset
import numpy as np

## Block2
load_dotenv()

## first will delete the old db everytime  we run, we dont need any dups to affect the system
if os.path.exists("chroma_db"):
    shutil.rmtree("chroma_db")

## load PDF
loader=PyPDFLoader("Paper/Detection_of_Fake_Accounts_on_Social_Media_Using_Multimodal_Data_With_Deep_Learning.pdf")
docs = loader.load()
print(f"Loaded {len(docs)} pages")

## chunk chunk
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks=splitter.split_documents(docs)
print(f"Total chunks: {len(chunks)}")

##dedups
seen=set()
unique_chunks=[]
for chunk in chunks:
    if chunk.page_content not in seen:
        seen.add(chunk.page_content)
        unique_chunks.append(chunk)
        
print(f"Unique chunks: {len(unique_chunks)}")

## Blcok3
## Vetore store + BM25Index
embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore=Chroma.from_documents(
    documents=unique_chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print(f"Vector store: {vectorstore._collection.count()} chunks stored")

## now BM25
## this guuy needs tokenized texts, split each chunks into words
tokenized_chunks=[
    chunk.page_content.lower().split() ##chunk ka jo page content hai usko lowercase kar and then split
    for chunk in unique_chunks
]

bm25=BM25Okapi(tokenized_chunks)
print(f"BM25 index: {len(tokenized_chunks)} chunks indexed")

## Block 4: Hybrid Search Function
## will combine vector search + BM25 using RRF
## RRF formula: score = 1/(rank + 60) for each system, then add scores

def hybrid_search(query, k=5):
    ## Step 1 first will vector search, smilairty search
    vector_results=vectorstore.similarity_search(query, k=10)
    
    ## Step 2 now BM35 search exact keywords search
    tokenized_query=query.lower().split()
    bm25_scores=bm25.get_scores(tokenized_query)
    
    ## get the top 10 BM25 chuk indices sorted by score
    top_bm25_indices=np.argsort(bm25_scores)[::-1][:10]
    bm25_results=[unique_chunks[i] for i in top_bm25_indices]
    
    ## Step 3 RRF erging
    ## har ek chunk ko will give a score based on its rank in each sysmter
    chunk_scores={}
    
    ## score for vector search
    for rank, chunk in enumerate(vector_results):
        content = chunk.page_content
        if content not in chunk_scores:
            chunk_scores[content]=0
        chunk_scores[content] += 1/(rank+60)
    
    ## score for BM25
    for rank, chunk in enumerate(bm25_results):
        content=chunk.page_content
        if content not in chunk_scores:
            chunk_scores[content]=0
        chunk_scores[content] += 1/(rank+60)
        
    ## step 4: will sort the comined score and return the top k
    sorted_chunks=sorted(
        chunk_scores.keys(),
        key=lambda x:chunk_scores[x], ## this tells hwo to sort instead of sorting by text itself
        reverse=True ## highest to lowest dega
    )[:k]
        
    ## now return the catual chunk objs not just text
    result_chunks=[]
    for content in sorted_chunks:
        for chunk in unique_chunks:
            if chunk.page_content == content:
                result_chunks.append(chunk)
                break
            
    return result_chunks
    
## Blcok 5

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

## the hybrid search results
test_query="What API was used to collect the Twitter data?"
print(f"\nTesting hybrid search for : {test_query}")
hybrid_chunks=hybrid_search(test_query, k=5)

print("\n----Hybrid Search results----")
for i, chunk in enumerate(hybrid_chunks):
    print(f"\nChunk {i+1}:")
    print(chunk.page_content[:200]) ## this is only for DEBUG printing nt what gets sent to the LLM, remeber LLM still get sthe full text, only over terminal we are prining less
    print("---")


from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
from pydantic import Field

class HybridRetriever(BaseRetriever):
    ## hwo many chunks to return
    ## why deafult 5 instead of k=5, pydantci is strict about how we will define theclass atribut, it needs Field() to propely register the attribute
    ## used this because its a strict parent and strict parents always propect the children from going in wrong way
    k: int=Field(default=5)
    
    def _get_relevant_documents(self,query:str) -> List[Document]:
        ## List[Doc] is type hint, it will tell what the functions returns
        return hybrid_search(query, k=self.k)
    
hybrid_retriever=HybridRetriever(k=5)

## RAG chain
qa_chain=RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=hybrid_retriever
)

question="What API is used to collecte the Twitter Data?"
answer=qa_chain.invoke(question)
print(f"\nQuestion: {question}")
print(f"Answer: {answer['result']}")

#### RAGAS bhai ka aaya phonee

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

eval_data = {
    "question":[],
    "answer": [],
    "contexts":[],
    "ground_truth":[]
}

print("\n--- Day2 RAGAS Evaluation -----")
for item in test_data:
    q=item["question"]
    gt=item["ground_truth"]
    
    result=qa_chain.invoke(q)
    
    ## get retrieved chunks
    retrieved=hybrid_search(q,k=5)
    
    eval_data["question"].append(q)
    eval_data["answer"].append(result["result"])
    eval_data["contexts"].append(
        [chunk.page_content for chunk in retrieved]
    )
    eval_data["ground_truth"].append(gt)
    print(f"DoneL {q[:50]}...")
    
## convert to dataset
dataset = Dataset.from_dict(eval_data)

ragas_embeddings = RagasHuggingfaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)


## evaluate with 3 metrics now - faithfulness, answer_relevancy, context_recall
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, context_recall],
    llm=llm,
    embeddings=ragas_embeddings
)

print("\n--- DAY 2 RAGAS SCORES ---")
print(results)
print("\n--- COMPARISON ---")
print(f"Day 1 faithfulness: 0.8135")
print(f"Day 2 faithfulness: {results['faithfulness']}")
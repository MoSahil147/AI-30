## Block1 imports
import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader ## Indexing
from langchain_text_splitters import RecursiveCharacterTextSplitter ## Indexing
#from langchain_community.embeddings import HuggingFaceEmbeddings ## Indexing
from langchain_huggingface import HuggingFaceEmbeddings ## indexing
from langchain_community.vectorstores import Chroma ## Indexing ## Retrieval
from langchain_groq import ChatGroq ##Generation
from langchain_classic.chains import RetrievalQA ## Retrieval ##Generation

## RAGAS part
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset ## import a data container, RAGAS only accept this specific format!
## will use huggingface embeddigns not openai embeddings
from ragas.embeddings import HuggingFaceEmbeddings as RagaHuggingFaceEmbeddings


## "The PDF gets loaded by ___, split by ___, converted to numbers by ___, stored in ___. At query time, ___ searches for relevant chunks and ___ sends them to ___ for the final answer."

## This is to remove already existing things inside chroma, so no storing thinsg again anad again!
## Clearing memory
if os.path.exists("chroma_db"):
    shutil.rmtree("chroma_db")

## Block2 load
load_dotenv()
# first will load
loader=PyPDFLoader("Paper/Detection_of_Fake_Accounts_on_Social_Media_Using_Multimodal_Data_With_Deep_Learning.pdf")
docs =loader.load() ## put bracket, if you wont then pages will be a function obj instead of lits of pages!

print(f"Loaded {len(docs)} pages")

## Block3 Splitters

## "Chunk overlap prevents context loss at boundaries — so a question whose answer spans two chunks can still be retrieved correctly."
splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks=splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks")

## Block4 now embeddings
embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2"
)

## will remove duplicates chunks before storing
## using sets uses hash jumps directly to answer, rather than going every item one by one from start to end like list
## ## list gets slower as the list grows, always instant, no matter how ig it gets 
seen = set()
unique_chunks=[]
for chunk in chunks:
    if chunk.page_content not in seen:
        seen.add(chunk.page_content)
        unique_chunks.append(chunk)
        
print(f"Unique chunks: {len(unique_chunks)}")

## Black5 Vectorstore
## now we will only store unique chunks
vectorstore=Chroma.from_documents(
    documents=unique_chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print(f"Stored {vectorstore._collection.count()} chunks in ChromaDB")

## Block 6: LLM ask questions

llm= ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

##Debugging, this will show which 3 chunks

question= "What is CNN LSTM GCN used for in this paper?"

retriever=vectorstore.as_retriever(
    search_kwargs={
        ## will search the 20 chunks first and then will pick the 5 most diverse ones
        "k":5,
        "fetch_k":20
        },
    search_type="mmr" ## maximal marginal relevance, avoids returning duplicate chunks
    )

retrieved_chunks=retriever.invoke(question)
print("\n----The Retrieved Chunks------")
for i, chunk in enumerate(retrieved_chunks):
    print(f"\nChunk {i+1}")
    print(chunk.page_content)
    print("------")

qa_chain= RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", ## stuff all the retrived chunks into one single prompt and sending to LLM
    ## everything gts literally stuffed into one prompt! fast simple works well for small context
    retriever=vectorstore.as_retriever(
        search_kwargs={
            "k":5,
            "fetch_k":20
            },
        search_type="mmr"
    )
)

## Question and answer
answer = qa_chain.invoke(question)
print(f"\nQuestion:{question}")
print(f"Answer:{answer['result']}")

## RAGA bhai ka aaya phone
## RAGAS part

## 3 questions 
test_questions=[
    "What deep learning models were used to detect fake accounts?",
    "What dataset was used to evaluate the model?",
    "What F1 score did the proposed model achieve?"
]

## empty containers for now
## its like making empty table with 3 columsn
## this is for the question, answer and getting the conetxt ki kaha se uthaya
eval_data = {
    "question":[],
    "answer":[],
    "contexts":[]
}

print("\n------ Running RAGAS ----------")
for q in test_questions:
    result=qa_chain.invoke(q) ## answer ke liye
    retrieved=retriever.invoke(q) ## for chunk
    eval_data["question"].append(q)
    eval_data["answer"].append(result["result"])
    eval_data["contexts"].append(
        ## RAGAS need raw text strings not Langchain chunk objects, thats why page content
        [chunk.page_content for chunk in retrieved]
        ## the above is a shortcut for
        ## context_texts = []
        ## for chunk in retrieved:
        ##     context_texts.append(chunk.page_content)
    )
    print(f"Done: {q[:50]}...")
    
## convertng to HuggingFace Dataset format
dataset=Dataset.from_dict(eval_data)

## will use Hugface embeddings
## it needs embedding for internal similarity callculation
ragas_embeddings=RagaHuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

answer_relevancy.strictness = 1

## now we will evaluate
results=evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=llm, ## LLM for judging
    embeddings=ragas_embeddings ## embedding for similarity
)

print("\n----- RAGAS Sscoresss-------")
print(results)




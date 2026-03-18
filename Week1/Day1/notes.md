RAGAS (Assesment of RAGS)

test suite for RAgs but for AI quality. without it we are guess wether the RAG works, in RAGAS we have numbers

4 metrics: FACC
    1. Faithfulness (did answer comes from chunks or did model hallucinate?)
    Did the answer came from chunks or the model made up some anwers
    like we actually checks if it exists in the retirved chunks we made
    Agar low score that means model is hallucinating
    needs are ansers + retrieved chunks

    2. Answer Relevancy (does the answer matches the quetsion, even after reversing)
    Does the answer is actualy addreses what was asked?
    Aso reverse questions, like same type ka question it asks, if it matches the orginal questtion? ex: what F1 score did the model achieve, what was the f1 score.. simailr question, so works better! so AR will be high!  need are question+answ

    3. Context Recall (needs ground truth, answer key)
    its like did retirver find all the chunks needed to answer completely? it checks kya woh answer tha retrived chunks mein! low score measn retiver is missing relevant docs
    We need ground truth + retrived chunks (will do on day2)

    4. Context Precison 
    are the retieved chunk actually useful or is there noise! it checks what fraction of retived chunks actually contributed to the answer! 
    low score means we are retieveing too many irrelevat chunks
    needs ground truth + retrived chunks

    in SHORT 
    Your question
      ↓
Retriever finds chunks
      ↓
LLM generates answer
      ↓
RAGAS scores:
  faithfulness    → answer vs chunks    (is it grounded?)
  answer_relevancy → answer vs question  (is it on topic?)
  context_recall  → chunks vs ground truth (did retriever find everything?)
  context_precision → chunks vs ground truth (is retrieval precise?)

  RAGAS uses LLM as a JUDGE it sends the answer and chunks to the LLM and asks to score quality! LLM evelautes the LLM's output!

  Day 1 Baseline:
- faithfulness: 0.8593
- answer_relevancy: nan (Groq compatibility issue)
- chunks: 157 unique
- chunk_size: 500, overlap: 50
- retrieval: MMR, k=5


day 1 summary

rag is something, which helps LLm reduces halucination, when we give a PDF, it chuks it and turns to number, store in a database and then when we ask a question to llm, converts the quetsion to numbers or vecetors sand tehn match the relevant answer

RAGAS (retriveal Augmeneted Generation Assessment)
measures RAG quality with metrics, like faithfulness, answer relevacy, context recall and precision also (today used faithfunlness and answer relevancy)

MISTAKES 

Bug 1: loader.load missing () 
→ pages was a function object not a list
→ Fix: added ()

Bug 2: Wrong import path for RecursiveCharacterTextSplitter
→ moved from langchain to langchain_text_splitters
→ Fix: updated import path

Bug 3: Spelling mistake - RetrevalQA
→ Python is exact - one wrong letter breaks everything  
→ Fix: RetrievalQA

Bug 4: ChromaDB accumulating duplicates on every run
→ 163 chunks became 1135 after 7 runs
→ Fix: shutil.rmtree("chroma_db") before every run

Bug 5: Duplicate chunks in retrieval
→ Same chunk returned 3 times
→ Fix: MMR search + set() deduplication before storing

Bug 6: HuggingFaceEmbeddings deprecated
→ Moved from langchain_community to langchain_huggingface
→ Fix: uv add langchain-huggingface, updated import

Bug 7: RAGAS trying to use OpenAI embeddings
→ We don't have OpenAI key
→ Fix: explicitly passed RagasHuggingFaceEmbeddings, used huggingfaceembeddings as Ragas

Bug 8: answer_relevancy nan
→ RAGAS 0.4.x incompatible with Groq's n parameter
→ Fix: deferred to Day 2

like readind error cleary, undertstanding hwat iis the issue and also refering to google for issues, rather than LLMs, fix answer relevancy also
# day13 first langchain agent!
# agent will decide which agent to use to based on the question
# Tool: AgentExecutor, create_react_agent

# os ke upar kuch nahi!
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm=ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0
)
print("LLM Ready!")

@tool
def calculator(expression:str)->str:
    """Calculate a mathematical expression. Input should be math expression like '25 * 2017 / 100'"""
    try:
        result=eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"
    
@tool
def word_counter(text:str)->str:
    """Count the number of words in text. Input should be string of text."""
    words=len(text.split())
    return f"The text has {words} words"

@tool
def paper_info(paper_name: str)->str:
    """Get basic info about AI papers. Input should be: attention, bert or rag."""
    
    papers = {
        "attention": "Published 2017 by Google Brain. Introduced the Transformer architecture.",
        "bert": "Published 2018 by Google. Bidirectional transformer for NLP tasks.",
        "rag": "Published 2020 by Facebook AI. Combines retrieval with generation."
    }
    
    paper_name=paper_name.lower().strip()
    return papers.get(paper_name, "Paper not found. Try: attention, bert or rag")

tools=[calculator, word_counter, paper_info]
print(f"Tools ready: {[t.name for t in tools]}")

# block 3 
# create a agent and run it
prompt=PromptTemplate.from_template("""You are a helpful assistant! You have access to these tools:

{tools}

Use this format EXACTLY:
Question: the input question
Thought: think about what to do
Action: tool name (one of [{tool_names}])
Action Input: input to the tool
Observation: result of the tool
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now knw the final answer
Final Answer: the final answer

Begin!

Question: {input}
Thought: {agent_scratchpad}""")

agent = create_react_agent(llm, tools, prompt)
agent_executor=AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)
print("Agent Ready!")

# test with 3 questions
print("\n---- Test 1: Calculator ----")
result=agent_executor.invoke({"input": "What is 25% of 2017"})
print(f"Answer: {result['output']}")

print("\n---- Test 2: Paper Info ----")
result=agent_executor.invoke({"input":"When was the BERT paper published?"})
print(f"Answer: {result['output']}")

print("\n---- Test 3: Multi-tool ----")
result = agent_executor.invoke({"input": "What year was the attention paper published and what is 10% of that year?"})
print(f"Answer: {result['output']}")
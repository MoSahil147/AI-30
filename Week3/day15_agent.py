## Code writing agent

#os ke upar kuc nahi
import os
import io
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
import traceback

load_dotenv()

llm=ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

print("LLM Readyyyy!!!!")

@tool
def run_python_code(code:str)->str:
    """Write and run Python code to solve problems.
    Input should be valid Python code as a sring.
    Use this for calculations, data processing,or any task that needs code."""
    
    try:
        ## clean the code before running it
        code = code.strip()
        code = code.replace("```python", "")
        code = code.replace("```", "")
        code = code.strip()
        
        # capture print output
        import os
        import sys
        
        output_capture=io.StringIO()
        sys.stdout=output_capture
        
        # run the code
        exec_globals={}
        exec(code, exec_globals)
        
        # restore stdout 
        sys.stdout=sys.__stdout__
        output=output_capture.getvalue()
        
        if output:
            return f"Code ran successfully!\nOutput: \n{output}"
        else:
            return "Code ran successfully! No output."
    
    except Exception as e:
        sys.stdout = sys.__stdout__
        return f"Error: {traceback.format_exc()}"
    
tools=[run_python_code]
print(f"Tools ready: {[t.name for t in tools]}")

# block 2: ReAct Prompt
# thsi will tell the LLM, buddy think like this

react_prompt=PromptTemplate.from_template("""
You are a Python coding agent. Solve problems by writing and running Python code.
You have access to the following tools:
{tools}

Use this EXACT format:
Question: the input question you must answer
Thought: think about what code to write
Action: run_python_code
Action Input: the python code to run (NO backticks, plain code only)
Observation: the result of the code

Once you receive an Observation with output, you MUST immediately do:
Thought: I now know the final answer
Final Answer: [your answer here]

Do NOT run the same code again after getting a result.
Do NOT use ``` backticks in Action Input.

Tool names available: {tool_names}

Question: {input}
Thought: {agent_scratchpad}
""")

print("Prompt Ready!")

# Aget creaion and testing! 
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt
)

agent_executor=AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    handle_parsing_errors=True
)

print("Agent Raedy! Testing now... \n")

# # test1
# result1=agent_executor.invoke({
#     "input":"Calculate the factorial of 10 using Python"
# })

# print(f"\nFinal Answer: {result1['output']}")

# # test2
# result2=agent_executor.invoke({
#     "input":"Generate the first 10 Fibonacci numbers and calculate their sum"
# })
# print(f"\nTest Answer: {result2['output']}")

# Test 3 real world problem
result3 = agent_executor.invoke({
    "input": "I have a list of numbers [15, 3, 9, 8, 2, 7, 1, 6]. Sort them, find the median, and calculate the standard deviation."
})
print(f"\nTest 3 Answer: {result3['output']}")
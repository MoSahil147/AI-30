# multi tool agent
# read_file + write_file
# Agent will decide which tool to use

# nthing above os
import os
import io
import traceback
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm= ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

print("LLM Ready!")

# Three tools making
@tool
def run_python_code(code:str)->str:
    """Write and run Python code to solve problems.
    Use for calculation, data processing, or any math task."""
    try:
        # cleaning the backticks the LLM might send
        code=code.strip().replace("```python", "").replace("```", "").strip()
        import sys
        # fake terminal: only captures print() output instead of showing on scree
        output_capture=io.StringIO()
        
        # redirect all print() calls into our fake terminal
        sys.stdout=output_capture
        
        #empty sandbox: for the code runs here without accessing our real variables! baxxe ke andar baxxa
        exec_globals={}
        # run the code sttring as Python
        exec(code, exec_globals)
        # restore normal printing to terminal
        sys.stdout=sys.__stdout__
        # get whatever was printed inside the fake terminal
        output=output_capture.getvalue()
        return f"Code ran successfully!\nOutput:\n{output}" if output else "Code ran successfully! No output."
    except Exception as e:
        sys.stdout=sys.__stdout__ # restore even if code crashes
        return f"Error: {traceback.format_exc()}"
    
@tool
def read_file(filepath:str)->str:
    """Read contents of a file. Input should be the file path."""
    try:
        # open file in read mode, f is the file object
        with open(filepath, 'r') as f:
            content=f.read() # read the entire file as string
        return f"File contents:\n{content}"
    except Exception as e:
        return f"Error reading file {str(e)}"

@tool
def write_file(input:str)->str:
    """Write content to a file.
    Input format: 'filepath|||content'
    Example: 'output.txt|||Hello World'"""
    try:
        # splitting on ||| to get filepath and content separately
        # using comma may cause a problem as content might contain commas!
        filepath, content=input.split("|||",1)
        
        # open in write mode: create file if doesnt' exist
        with open(filepath.strip(), 'w') as f:
            f.write(content.strip())
        return f"Successfully written to {filepath}"
    except Exception as e:
        return f"Error writting file: {str(e)}"
    
tools=[run_python_code, read_file, write_file]
print(f"Tooks ready! {[t.name for t in tools]}")

# block 3: prompt timeee and agent
react_prompt = PromptTemplate.from_template("""
You are a helpful agent with access to Python, file reading, and file writing tools.
Solve tasks by using the right tool for each step.

You have access to the following tools:
{tools}

Use this EXACT format:
Question: the input question you must answer
Thought: think about which tool to use and why
Action: the tool to use (must be one of {tool_names})
Action Input: the input to the tool
Observation: the result of the tool

IMPORTANT RULES:
- After EACH Observation, move to the NEXT step. Never repeat the same Action.
- Once you have file contents, calculate immediately.
- Once you have the result, write immediately.
- Once you have written, give Final Answer immediately.

Thought: I now know the final answer
Final Answer: [your answer here]

Do NOT use ``` backticks in Action Input.
Do NOT repeat the same Action twice.

Tool names available: {tool_names}

Question: {input}
Thought: {agent_scratchpad}
""")

agent=create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

agent_executor=AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    handle_parsing_errors=True
)
print("Agent ready!")

# Block 4: Create testing file and run the agent

# first will create a test data file, like safety first sir
with open("numbers.txt", "w") as f:
    f.write("15\n3\n9\n8\n2\n7\n1\n6")

print("Test file created: numbers.txt")

# now will ask the agent sir please use all the 3 tools
result = agent_executor.invoke({
    "input":"Read the file numbers.txt, calculte the average of all numbers in it, then write the result to output.txt"
})

print(f"\nFinal Answer: {result['output']}")

# verifying the output.txt was created
print("\nChecking the output.txt....")
with open("output.txt","r") as f:
    print(f.read())
 
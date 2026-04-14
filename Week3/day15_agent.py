## Code writing agent

#os ke upar kuc nahi
import os
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
            return "Code ran successfullt! No output."
    
    except Exception as e:
        sys.stdout = sys.__stdout__
        return f"Error: {traceback.from_exc()}"
    
tools=[run_python_code]
print(f"Tools ready: {[t.name for t in tools]}")
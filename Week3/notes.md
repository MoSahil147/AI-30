Day 15 - Code writing agent
Agent writes Python code → runs it → returns result
Safety: sandbox/exec with try-catch
Production: use Docker containers for real isolation

sys.stdout redirect = capture print output into a string
Without it → agent writes code but never sees the results
io.StringIO() = fake terminal that stores output in memory

ReAct prompt placeholders:
{tools}             = tool descriptions (auto-filled)
{tool_names}        = tool names (auto-filled)
{input}             = user question (auto-filled)
{agent_scratchpad}  = reasoning history (auto-filled)
PromptTemplate = Mad Libs, LangChain fills the blanks

Defensive programming = sanitize inputs, don't trust them
Prompting = fragile (LLM ignores instructions sometimes)
Code fix = reliable (always strips backticks before exec)

Median = middle value when sorted (not same as mean)
Mean = sum / count (affected by outliers)
Median = robust to outliers

Std Dev = how far numbers spread from average
Small std dev = clustered together
Large std dev = spread out

Day 15 agent = 1 tool (run_python_code) = can calculate
Day 16 agent = multiple tools = can calculate + read + write
Tool selection = agent decides which tool fits the problem
Real agent power = right tool for right job

Data dependency = step N needs output from step N-1
Multi-tool order matters:
1. read_file  → get data
2. run_python_code → process data  
3. write_file → save results
Agent figures out this order automatically from the task!

io.StringIO()   = fake terminal in memory
sys.stdout      = where print() sends output
exec_globals    = sandbox so code can't touch our variables
exec(code)      = run a string as Python code
'r' mode        = read file
'w' mode        = write file (creates if missing)
|||             = separator that won't appear in normal content

Day 16: Multi-tool agent
3 tools: run_python_code, read_file, write_file
Agent selects correct tool based on docstrings
Data dependency: read → calculate → write
Controlled testing = create input file yourself

Q1: exec_globals={} means the code runs in an empty namespace — it can't access your real variables like llm, tools, your API key. Without it, user code could do print(os.getenv("GROQ_API_KEY")) and steal your secrets. It's a security sandbox.
Q2: Content can contain commas, spaces, newlines — all common characters. ||| is rare enough that it won't accidentally appear in content. Delimiter collision prevention.
Q3: Right — but what HAPPENS after 10 iterations? The agent stops and returns whatever it has instead of running forever. Without it, a confused agent loops infinitely and burns your API credits.
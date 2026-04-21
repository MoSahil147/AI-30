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
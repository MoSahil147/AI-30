Day 15 - Code writing agent
Agent writes Python code → runs it → returns result
Safety: sandbox/exec with try-catch
Production: use Docker containers for real isolation

sys.stdout redirect = capture print output into a string
Without it → agent writes code but never sees the results
io.StringIO() = fake terminal that stores output in memory
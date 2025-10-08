from fastmcp import FastMCP

app = FastMCP("MCP Demo")

@app.tool
def add(a: int, b: int) -> int:
    """Adds 2 integers"""
    return a + b


@app.tool
def multiply(a: int, b: int) -> int:
    """Multiplies 2 integers"""
    return a * b


@app.resource("resource://hello")
def say_hello()-> str:
    """Returns a greeting message"""
    return "Hello, FastMCP Learner!"

@app.resource("resource://greet/{user}")
def greet_user(user: str)->str:
    """Greets a user by name"""
    return f"Hello, {user}!"

@app.prompt
def bash_script_prompt(user_request: str)->str:
    """Generates a bash script based on user request"""
    system_prompt = f"""
You are an expert Bash Script Developer.
Provide me with a working bash script for the following request:
# {user_request}

Only provide the script, no explanations.
"""
    return system_prompt
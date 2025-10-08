"""
Bash Script Generator Agent using LangGraph
A robust agent that generates, tests, and fixes bash scripts automatically.
"""

import os
import subprocess
import tempfile
from typing import TypedDict, Annotated, Literal
from datetime import datetime
import operator
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ============================================================================
# STATE DEFINITION
# ============================================================================


class AgentState(TypedDict):
    """State schema for the bash script generation agent."""

    user_request: str
    current_script: str
    script_history: Annotated[list, operator.add]
    test_results: dict
    errors: list
    iteration_count: int
    status: str
    feedback: str
    max_iterations: int
    final_output: str


# ============================================================================
# CONFIGURATION
# ============================================================================


class Config:
    """Configuration for the agent."""

    MAX_ITERATIONS = 5
    EXECUTION_TIMEOUT = 30
    ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
    ENABLE_SHELLCHECK = True
    SAFETY_MODE = True

    # Dangerous commands to block in safety mode
    DANGEROUS_PATTERNS = [
        "rm -rf /",
        "dd if=/dev/zero",
        "mkfs.",
        "> /dev/sda",
        "wget | sh",
        "curl | sh",
        "fork()",
    ]


# ============================================================================
# TOOL FUNCTIONS
# ============================================================================


def execute_bash_script(script_content: str, timeout: int = 30) -> dict:
    """
    Execute a bash script in a controlled environment.

    Args:
        script_content: The bash script to execute
        timeout: Maximum execution time in seconds

    Returns:
        Dictionary with stdout, stderr, exit_code, and execution_time
    """
    try:
        # Create a temporary file for the script
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script_content)
            script_path = f.name

        # Make the script executable
        os.chmod(script_path, 0o755)

        # Execute the script
        start_time = datetime.now()
        result = subprocess.run(
            ["bash", script_path], capture_output=True, text=True, timeout=timeout
        )
        execution_time = (datetime.now() - start_time).total_seconds()

        # Clean up
        os.unlink(script_path)

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "execution_time": execution_time,
            "success": result.returncode == 0,
        }

    except subprocess.TimeoutExpired:
        os.unlink(script_path)
        return {
            "stdout": "",
            "stderr": f"Script execution timed out after {timeout} seconds",
            "exit_code": -1,
            "execution_time": timeout,
            "success": False,
        }
    except Exception as e:
        if "script_path" in locals():
            os.unlink(script_path)
        return {
            "stdout": "",
            "stderr": f"Execution error: {str(e)}",
            "exit_code": -1,
            "execution_time": 0,
            "success": False,
        }


def validate_syntax(script_content: str) -> dict:
    """
    Validate bash script syntax using bash -n.

    Args:
        script_content: The bash script to validate

    Returns:
        Dictionary with validation results
    """
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script_content)
            script_path = f.name

        result = subprocess.run(
            ["bash", "-n", script_path], capture_output=True, text=True, timeout=5
        )

        os.unlink(script_path)

        return {
            "valid": result.returncode == 0,
            "errors": result.stderr if result.returncode != 0 else "",
            "warnings": [],
        }

    except Exception as e:
        if "script_path" in locals():
            os.unlink(script_path)
        return {"valid": False, "errors": f"Validation error: {str(e)}", "warnings": []}


def run_shellcheck(script_content: str) -> dict:
    """
    Run shellcheck for static analysis (optional).

    Args:
        script_content: The bash script to analyze

    Returns:
        Dictionary with shellcheck results
    """
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script_content)
            script_path = f.name

        result = subprocess.run(
            ["shellcheck", "-f", "gcc", script_path],
            capture_output=True,
            text=True,
            timeout=10,
        )

        os.unlink(script_path)

        return {
            "available": True,
            "warnings": result.stdout,
            "error_count": result.stdout.count("error:"),
            "warning_count": result.stdout.count("warning:"),
        }

    except FileNotFoundError:
        return {
            "available": False,
            "warnings": "",
            "error_count": 0,
            "warning_count": 0,
        }
    except Exception as e:
        if "script_path" in locals():
            os.unlink(script_path)
        return {
            "available": True,
            "warnings": f"Shellcheck error: {str(e)}",
            "error_count": 0,
            "warning_count": 0,
        }


def check_dangerous_patterns(script_content: str) -> list:
    """
    Check for potentially dangerous commands in the script.

    Args:
        script_content: The bash script to check

    Returns:
        List of dangerous patterns found
    """
    dangerous_found = []
    for pattern in Config.DANGEROUS_PATTERNS:
        if pattern in script_content:
            dangerous_found.append(pattern)
    return dangerous_found


# ============================================================================
# NODE FUNCTIONS
# ============================================================================


def script_generator_node(state: AgentState) -> AgentState:
    """
    Generate or regenerate a bash script based on requirements and feedback.
    """
    llm = ChatOpenAI(
        model="gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),  # type: ignore
        base_url=os.getenv("BASE_URL"),
    )

    # Build the prompt based on iteration
    if state["iteration_count"] == 0:
        # Initial generation
        system_prompt = """You are an expert bash script developer. Generate robust, production-quality bash scripts.

Your scripts must:
1. Start with #!/bin/bash
2. Include error handling (set -euo pipefail)
3. Have clear comments explaining functionality
4. Validate inputs
5. Use proper quoting for variables
6. Include helpful usage messages
7. Handle edge cases gracefully
8. Use functions for better organization
9. Return appropriate exit codes

Generate ONLY the bash script, no explanations before or after."""

        user_prompt = f"""Generate a bash script that: {state["user_request"]}

Make it robust, secure, and production-ready."""

    else:
        # Regeneration with error feedback
        system_prompt = """You are an expert bash script debugger. Fix the errors in the previous script while maintaining its core functionality.

Analyze the errors carefully and provide a corrected version of the script.
Generate ONLY the bash script, no explanations before or after."""

        user_prompt = f"""The previous script failed with these errors:

{state["feedback"]}

Previous script:
```bash
{state["current_script"]}
```

Fix these errors and generate an improved version."""

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    response = llm.invoke(messages)
    script = response.content.strip()

    # Remove markdown code blocks if present
    if script.startswith("```bash"):
        script = script.split("```bash")[1].split("```")[0].strip()
    elif script.startswith("```"):
        script = script.split("```")[1].split("```")[0].strip()

    # Ensure shebang is present
    if not script.startswith("#!"):
        script = "#!/bin/bash\n\n" + script

    return {
        **state,
        "current_script": script,
        "script_history": [script],
        "status": "generated",
    }


def safety_check_node(state: AgentState) -> AgentState:
    """
    Perform safety checks on the generated script.
    """
    if not Config.SAFETY_MODE:
        return {**state, "status": "safety_passed"}

    dangerous_patterns = check_dangerous_patterns(state["current_script"])

    if dangerous_patterns:
        error_msg = f"Dangerous patterns detected: {', '.join(dangerous_patterns)}"
        return {
            **state,
            "status": "safety_failed",
            "errors": state["errors"] + [{"type": "security", "message": error_msg}],
            "feedback": f"Security Error: {error_msg}\nRemove or modify these dangerous commands.",
        }

    return {**state, "status": "safety_passed"}


def validator_node(state: AgentState) -> AgentState:
    """
    Validate bash script syntax before execution.
    """
    validation_result = validate_syntax(state["current_script"])

    if not validation_result["valid"]:
        errors = [{"type": "syntax", "message": validation_result["errors"]}]

        return {
            **state,
            "status": "validation_failed",
            "errors": errors,
            "feedback": f"Syntax Errors:\n{validation_result['errors']}",
        }

    # Run shellcheck if enabled
    if Config.ENABLE_SHELLCHECK:
        shellcheck_result = run_shellcheck(state["current_script"])
        if shellcheck_result["available"] and shellcheck_result["error_count"] > 0:
            return {
                **state,
                "status": "validation_warning",
                "feedback": f"Shellcheck warnings:\n{shellcheck_result['warnings']}",
            }

    return {**state, "status": "validated"}


def test_executor_node(state: AgentState) -> AgentState:
    """
    Execute the bash script in a controlled environment.
    """
    test_results = execute_bash_script(
        state["current_script"], timeout=Config.EXECUTION_TIMEOUT
    )

    return {**state, "test_results": test_results, "status": "tested"}


def error_analyzer_node(state: AgentState) -> AgentState:
    """
    Analyze test results and categorize errors.
    """
    test_results = state["test_results"]

    if test_results["success"]:
        return {
            **state,
            "status": "success",
            "errors": [],
            "feedback": "Script executed successfully!",
        }

    # Analyze the errors
    llm = ChatOpenAI(
        model="gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),  # type: ignore
        base_url=os.getenv("BASE_URL"),
    )

    system_prompt = """You are an expert bash debugger. Analyze the error and provide:
1. Error type (syntax, runtime, logic, permission, dependency, timeout)
2. Root cause
3. Specific fix recommendations
4. Whether the error is fixable

Be concise and specific."""

    user_prompt = f"""Analyze this bash script error:

Script:
```bash
{state["current_script"]}
```

Exit Code: {test_results["exit_code"]}
STDERR:
{test_results["stderr"]}

STDOUT:
{test_results["stdout"]}

Provide structured analysis."""

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    response = llm.invoke(messages)
    analysis = response.content

    errors = [
        {
            "type": "runtime",
            "message": test_results["stderr"],
            "analysis": analysis,
            "exit_code": test_results["exit_code"],
        }
    ]

    return {
        **state,
        "status": "analyzed",
        "errors": errors,
        "feedback": f"Error Analysis:\n{analysis}\n\nSTDERR:\n{test_results['stderr']}",
    }


def decision_node(state: AgentState) -> AgentState:
    """
    Decide the next action based on current state.
    """
    # Check if we've reached max iterations
    if state["iteration_count"] >= state["max_iterations"]:
        return {
            **state,
            "status": "max_iterations_reached",
            "final_output": f"Failed to generate working script after {state['max_iterations']} attempts.\n\nLast error:\n{state['feedback']}",
        }

    # Check if script succeeded
    if state["status"] == "success":
        return {
            **state,
            "status": "complete",
            "final_output": f"Successfully generated bash script:\n\n{state['current_script']}\n\nTest Results:\n{state['test_results']['stdout']}",
        }

    # Increment iteration and retry
    return {
        **state,
        "iteration_count": state["iteration_count"] + 1,
        "status": "retrying",
    }


# ============================================================================
# ROUTING FUNCTIONS
# ============================================================================


def route_after_safety(state: AgentState) -> Literal["validator", "generator"]:
    """Route after safety check."""
    if state["status"] == "safety_failed":
        return "generator"
    return "validator"


def route_after_validation(state: AgentState) -> Literal["executor", "generator"]:
    """Route after validation."""
    if state["status"] == "validation_failed":
        return "generator"
    return "executor"


def route_after_decision(state: AgentState) -> Literal["generator", "end"]:
    """Route after decision node."""
    if state["status"] in ["complete", "max_iterations_reached"]:
        return "end"
    return "generator"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================


def create_bash_agent_graph():
    """
    Create and compile the LangGraph workflow for bash script generation.
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("generator", script_generator_node)
    workflow.add_node("safety_check", safety_check_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("executor", test_executor_node)
    workflow.add_node("analyzer", error_analyzer_node)
    workflow.add_node("decision", decision_node)

    # Set entry point
    workflow.set_entry_point("generator")

    # Add edges
    workflow.add_edge("generator", "safety_check")
    workflow.add_conditional_edges(
        "safety_check",
        route_after_safety,
        {"validator": "validator", "generator": "generator"},
    )
    workflow.add_conditional_edges(
        "validator",
        route_after_validation,
        {"executor": "executor", "generator": "generator"},
    )
    workflow.add_edge("executor", "analyzer")
    workflow.add_edge("analyzer", "decision")
    workflow.add_conditional_edges(
        "decision", route_after_decision, {"generator": "generator", "end": END}
    )

    return workflow.compile()


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def generate_bash_script(user_request: str, max_iterations: int = 5) -> dict:
    """
    Main function to generate a bash script using the agent.

    Args:
        user_request: Description of what the bash script should do
        max_iterations: Maximum number of generation attempts

    Returns:
        Dictionary with final script and execution details
    """
    # Create the graph
    app = create_bash_agent_graph()

    # Initialize state
    initial_state = {
        "user_request": user_request,
        "current_script": "",
        "script_history": [],
        "test_results": {},
        "errors": [],
        "iteration_count": 0,
        "status": "initialized",
        "feedback": "",
        "max_iterations": max_iterations,
        "final_output": "",
    }

    # Run the graph
    final_state = app.invoke(initial_state)

    return {
        "success": final_state["status"] == "complete",
        "script": final_state["current_script"],
        "iterations": final_state["iteration_count"],
        "output": final_state["final_output"],
        "test_results": final_state["test_results"],
        "history": final_state["script_history"],
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example 1: Simple file backup script
    print("=" * 80)
    print("Example 1: File Backup Script")
    print("=" * 80)

    result = generate_bash_script(
        "Create a script that backs up a given directory to a specified backup location. The script should take two arguments: the source directory and the destination backup directory. It should create a timestamped tar.gz archive of the source directory in the backup location. Include error handling for cases such as missing arguments, non-existent source directory, and inaccessible backup location."
    )

    print(f"\nSuccess: {result['success']}")
    print(f"Iterations: {result['iterations']}")
    print(f"\nGenerated Script:\n{result['script']}")
    print(f"\n{result['output']}")

    print("\n" + "=" * 80)
    # print("Example 2: System Information Script")
    # print("=" * 80)

    # result = generate_bash_script(
    #    "Create a script that displays system information including CPU, memory, disk usage, and uptime in a formatted way"
    # )

    # print(f"\nSuccess: {result['success']}")
    # print(f"Iterations: {result['iterations']}")
    # print(f"\nGenerated Script:\n{result['script']}")
    # print(f"\n{result['output']}")

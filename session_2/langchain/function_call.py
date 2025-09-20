from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode

load_dotenv()

model = os.getenv("MODEL", "gemini-2.5-flash-lite")
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatOpenAI(
    model=model,
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


@tool
def get_sys_info():
    """
    Get the current system information.
    Get the OS, Kernel version and type of Linux System used.
    """
    import platform

    os_name = "Unknown"
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    os_name = line.strip().split("=", 1)[1].strip('"')
                    break
    except FileNotFoundError:
        os_name = platform.system()
    return f"Kernel: {platform.release()}, OS: {os_name}"


messages = [
    SystemMessage(
        content="You are a helpful assistant. If you are not clear about the user's OS, call `get_sys_info` tool."
    ),
    HumanMessage(content="provide me with the command to install neovim on my system"),
]

response = llm.bind_tools([get_sys_info]).invoke(messages)
messages.append(response)

tool_node = ToolNode(tools=[get_sys_info])

if response.tool_calls:
    response = tool_node.invoke({"messages": [response]})
    messages.extend(response["messages"])

    response = llm.invoke(messages)

print(response.content)

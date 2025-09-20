import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

model = os.getenv("MODEL", "gemini-2.5-flash-lite")
api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/",
)


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
    {
        "role": "system",
        "content": "You are a helpful assistant. If you are not clear about the user's OS, call `get_sys_info` tool.",
    },
    {
        "role": "user",
        "content": "provide me with the command to install neovim on my system",
    },
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_sys_info",
            "description": "Get the current system information. Get the OS, Kernel version and type of Linux System used.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }
]

response = client.chat.completions.create(
    model=model, messages=messages, tools=tools, tool_choice="auto"
)

messages.append(response.choices[0].message)

if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    if tool_call.function.name == "get_sys_info":
        tool_response = get_sys_info()

        messages.append(
            {
                "role": "tool",
                "name": tool_call.function.name,
                "content": tool_response,
                "tool_call_id": tool_call.id,
            }
        )

        response = client.chat.completions.create(model=model, messages=messages)

        messages.append(
            {
                "role": response.choices[0].message.role,
                "content": response.choices[0].message.content,
            }
        )

print(messages[-1]["content"])

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("MODEL", "gemini-2.5-flash-lite"),
    api_key=os.getenv("GEMINI_API_KEY"),  # type: ignore
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

messages = [
    SystemMessage(
        "You are a Linux Terminal running bash shell. Whatever prompt I give interpret it as a bash command and only respond with the expected output of the command. Do not include any explanations or additional text.",
    ),
    HumanMessage("lsblk"),
]


response = llm.invoke(messages)

print(response.content)

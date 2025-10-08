from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
from rich.pretty import pprint
import client
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

load_dotenv()

MODEL = os.getenv("MODEL", "gemini-2.5-flash")
BASE_URL = os.getenv("BASE_URL", "https://api.generativeai.google.com/v1beta2/models/")


llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, temperature=0.1)

config = {
    "filesystem": {
        "transport": "stdio",
        "command": "npx",
        "args": [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "/home/manu/Projects/",
        ]
    }
}


async def main():
    client = MultiServerMCPClient(config) # type: ignore

    tools = await client.get_tools()
    pprint(tools)

    # agent = create_react_agent(llm, tools)

    # message = [HumanMessage("What is the sum of 5 and 10, and then multiply the result by 2?")]

    # response = await agent.ainvoke({"messages": message})

    # pprint(response)


if __name__ == "__main__":
    asyncio.run(main())
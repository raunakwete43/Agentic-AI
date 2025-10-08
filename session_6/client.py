from fastmcp import Client
from rich.pretty import pprint
import asyncio

config = {
    "mcpServers": {
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
}


client = Client(config)

async def main():
    async with client:
        tools = await client.list_tools()
        # pprint(tools)

        result = await client.call_tool("read_file", {"path": "/home/manu/Projects/Python/agentic_ai/main.py"})

        pprint(result)


if __name__ == "__main__":
    asyncio.run(main())
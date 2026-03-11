import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def test():

    url = "http://127.0.0.1:8050/mcp/mcp"

    async with streamable_http_client(url) as (read, write, _):

        async with ClientSession(read, write) as session:

            await session.initialize()

            print("\nTOOLS DISPONIBILI:")
            tools = await session.list_tools()

            for tool in tools.tools:
                print("-", tool.name)

            print("\nTEST search_products\n")

            result = await session.call_tool(
                "search_products",
                {"query": "iphone 13"}
            )

            print(result)


if __name__ == "__main__":
    asyncio.run(test())
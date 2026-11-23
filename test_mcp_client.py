"""
test_mcp_client.py: pretends to be an AI app, to prove the MCP server works.

It does what an AI assistant (an "MCP client") does behind the scenes:
  1. starts the server (mcp_server.py)
  2. asks "which tools do you have?"
  3. calls one tool with inputs, and prints the answer
Run:  python test_mcp_client.py     (takes about 30-60 seconds; the optimizer runs several times)
"""
import asyncio, json, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server = StdioServerParameters(command=sys.executable, args=["mcp_server.py"])
    async with stdio_client(server) as (read, write):          # 1. start the server and connect to it
        async with ClientSession(read, write) as session:
            await session.initialize()                          #    say hello (MCP handshake)
            tools = await session.list_tools()                  # 2. which tools do you have?
            print("Tools the AI can see:", [t.name for t in tools.tools])
            print("\nAsking: compare 3 what-if scenarios...\n")
            result = await session.call_tool("compare_scenarios", {"scenarios": [   # 3. call one tool
                {"name": "Base $100M", "budget_musd": 100},
                {"name": "Budget cut to $85M", "budget_musd": 85},
                {"name": "Lose a quarter of the data/AI team", "budget_musd": 100, "capacity_change": {"data_ai": 0.75}},
            ]})
            print(json.dumps(json.loads(result.content[0].text), indent=1))


asyncio.run(main())

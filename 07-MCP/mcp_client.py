import asyncio
import os
from dotenv import load_dotenv
from fastmcp import Client

load_dotenv()

async def main():
    # Load and verify MCP server path
    server_path = os.path.join(os.getcwd(), "rag_mcp_server.py")
    if not os.path.exists(server_path):
        print(f"[ERROR] MCP server not found at {server_path}")
        return

    print(f"[INFO] Launching MCP server from: {server_path}")

    # Initialize client
    client = Client(server_path)

    async with client:
        # Ping to check connection
        await client.ping()
        print("[INFO] Connected to MCP server.\n")

        # List available tools
        tools = await client.list_tools()
        print("🧰 Available Tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")

        # List available resources
        resources = await client.list_resources()
        print("\n📂 Available Resources:")
        for res in resources:
            print(f"  - {res.uri}: {res.name}")

        print("\nYou can now ask natural language questions to the server.")
        print("Type 'exit' to quit.\n")

        # Take user questions dynamically
        while True:
            question = input("❓ Enter your question: ").strip()
            if question.lower() in ["exit", "quit", "q"]:
                print("👋 Exiting client...")
                break

            print(f"\n{'='*80}")
            print(f"Question: {question}")
            print(f"{'='*80}")

            # Call the server tool (assumed tool name: smart_query)
            try:
                result = await client.call_tool("smart_query", {"question": question})

                if hasattr(result, "data") and result.data:
                    print("\nAnswer:\n" + str(result.data))
                elif hasattr(result, "content") and result.content:
                    first = result.content[0]
                    text = getattr(first, "text", str(first))
                    print("\nAnswer:\n" + text)
                else:
                    print("\n[WARN] No content returned from the tool.")
            except Exception as e:
                print(f"[ERROR] Failed to query tool: {e}")

if __name__ == "__main__":
    asyncio.run(main())

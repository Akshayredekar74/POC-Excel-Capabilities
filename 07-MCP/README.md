# RAG MCP Server

An MCP server that provides RAG (Retrieval Augmented Generation) capabilities using vector search and SQL queries.

## Features

- **Vector Search**: Semantic search through documents using FAISS embeddings
- **SQL Query**: Execute SQL queries against in-memory SQLite database
- **Smart Query**: Automatically routes queries to vector search or SQL based on keywords
- **Resources**: Access database and file schemas

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
DATA_FOLDER=test
```

3. Place your CSV/Excel files in the `test` folder

## Running the Server

### As Standalone Server
```bash
python rag_mcp_server.py
```

### With MCP Client
```bash
python mcp_client.py
```

### With Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "rag-agent": {
      "command": "python",
      "args": ["/path/to/rag_mcp_server.py"],
      "env": {
        "OPENAI_API_KEY": "your_key",
        "GROQ_API_KEY": "your_key",
        "DATA_FOLDER": "/path/to/test"
      }
    }
  }
}
```

## Available Tools

1. **vector_search**: Semantic search using embeddings
   - Input: `query` (string)
   - Output: Answer based on vector retrieval

2. **sql_query**: Execute SQL queries
   - Input: `question` (string)
   - Output: Answer based on SQL results

3. **smart_query**: Auto-routing query handler
   - Input: `question` (string)
   - Output: Answer using appropriate method

## Available Resources

1. `schema://database` - Database table schemas
2. `schema://files` - File structure and columns

## Example Queries

```python
vector_search("Which organizations mention AI?")
sql_query("How many organizations were founded in the 1990s?")
smart_query("What is the average number of employees?")
```
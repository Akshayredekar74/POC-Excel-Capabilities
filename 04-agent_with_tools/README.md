# AI Agent with SQL Database

## Overview
An intelligent AI agent that automatically cleans data, detects types, and answers questions about CSV/Excel files using natural language.

## Architecture
```
Excel/CSV → Data Cleaner → DuckDB → LangGraph Agent → Gemini LLM → SQL → Answer
```

## Features

### 1. Intelligent Data Type Detection
The system automatically detects and converts:
- **Boolean**: Detects values like true/false, yes/no, 1/0, y/n
- **Integer**: Detects whole numbers (with currency symbols, commas removed)
- **Float**: Detects decimal numbers (handles $, ₹, €, £, ¥, %, commas)
- **String**: Keeps text as-is

### 2. Automatic Data Cleaning
- Removes currency symbols: $, ₹, €, £, ¥
- Removes commas and formatting
- Handles parentheses for negative numbers: (100) → -100
- Fills null values: 0 for numbers, False for booleans
- Reports all cleaning operations

### 3. SQL Tools Available
- `get_table_info()`: Schema and sample data
- `execute_aggregation()`: SUM, AVG, COUNT, MIN, MAX
- `get_distinct_values()`: Unique values in column
- `group_by_aggregate()`: Group by with aggregation
- `filter_data()`: Filter with conditions
- `get_column_stats()`: Mean, min, max, stddev
- `get_top_n()`: Top N rows by column

### 4. Self-Correcting Agent
- Uses LangGraph for workflow management
- Automatically retries on errors (up to 3 times)
- Provides error context to LLM for fixes
- Generates better SQL on each retry

## Files Structure
```
ai_agent/
├── agent.py          # Main agent with LangGraph workflow
├── cleaner.py        # Intelligent data type detection and cleaning
├── tools.py          # SQL operation tools
├── app.py            # Gradio UI
├── requirements.txt  # Dependencies
└── README.md         # This file
```

## Installation

```bash
cd D:\PROJECTS\chat_with_excel\ai_agent
pip install -r requirements.txt
```

## Setup

Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

## Usage

```bash
python app.py
```

Then open your browser to `http://127.0.0.1:7860`

## Technical Details

### LangGraph Workflow
```
get_schema → generate_sql → execute_sql → [retry if error] → generate_answer
```

### Type Detection Logic
- Analyzes 100 sample values per column
- Uses regex patterns to detect numbers
- Threshold: 80% match to convert type
- Graceful fallback to string if conversion fails

### DuckDB Integration
- In-memory database for fast queries
- Supports complex SQL operations
- PRAGMA for schema introspection
- Efficient aggregations and joins

## Troubleshooting

### Issue: "Type mismatch error"
**Solution**: The agent automatically retries with CAST() conversions

### Issue: "Column not found"
**Solution**: Check the Schema tab for exact column names (case-sensitive)

### Issue: "No results"
**Solution**: Try rephrasing question or check if data exists

## Advanced Usage

### Custom SQL Tools
You can extend `tools.py` with new operations:

```python
def custom_operation(self, params):
    query = f"YOUR SQL HERE"
    result = self.conn.execute(query).fetchdf()
    return {"results": result}
```

### Modify Retry Logic
Change max iterations in `agent.py`:
```python
self.max_iterations = 5  # Default is 3
```

## Performance

- **Loading**: Fast with Polars
- **Cleaning**: O(n) per column
- **Queries**: Optimized by DuckDB
- **Type Detection**: ~100 samples per column


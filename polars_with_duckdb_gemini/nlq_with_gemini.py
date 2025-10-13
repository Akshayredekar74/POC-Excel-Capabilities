"""
Fast Chat with CSV/Excel
Polars + DuckDB + Gemini + LlamaIndex
"""

import os
import polars as pl
import pandas as pd
import duckdb
import gradio as gr
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-2.0-flash-exp')

conn = duckdb.connect(database=':memory:')

def load_file(file_path):
    if file_path.endswith('.csv'):
        try:
            df = pl.read_csv(
                file_path, 
                infer_schema_length=10000,
                truncate_ragged_lines=True
            )
        except:
            try:
                df = pl.read_csv(
                    file_path,
                    encoding="utf8-lossy",
                    infer_schema_length=10000,
                    truncate_ragged_lines=True
                )
            except:
                import pandas as pd
                df = pl.from_pandas(pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip'))
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pl.read_excel(file_path)
    else:
        return None
    
    conn.register("data", df.to_pandas())
    return df

def ask_question(question, file_path=None):
    if file_path:
        df = load_file(file_path)
        if df is None:
            return "Please provide a CSV or Excel file"
    
    schema_info = conn.execute("SELECT * FROM data LIMIT 0").description
    columns = [col[0] for col in schema_info]
    
    sample_data = conn.execute("SELECT * FROM data LIMIT 3").fetchdf()
    
    schema_text = f"""Table: data
Columns: {', '.join(columns)}

Sample data:
{sample_data.to_string(index=False)}"""
    
    sql_prompt = f"""Convert this natural language question to SQL.

Database Schema:
{schema_text}

Question: {question}

Rules:
- Table name is 'data'
- Return only SQL query
- Use DuckDB syntax

SQL:"""

    response = model.generate_content(sql_prompt)
    sql = response.text.strip()
    
    if "```sql" in sql:
        sql = sql.split("```sql")[1].split("```")[0].strip()
    elif "```" in sql:
        sql = sql.split("```")[1].split("```")[0].strip()
    sql = sql.replace('`', '')
    
    print(f"Generated SQL: {sql}")
    
    try:
        result_df = conn.execute(sql).fetchdf()
    except Exception as e:
        return f"Error: {str(e)}\n\nSQL: {sql}"
    
    display_df = result_df.head(20)
    total_rows = len(result_df)
    
    answer_prompt = f"""Question: {question}

Query: {sql}

Results ({len(display_df)} of {total_rows} rows):
{display_df.to_string(index=False)}

Provide a clear answer based on these results."""

    answer_response = model.generate_content(answer_prompt)
    answer = answer_response.text
    
    return f"""*Answer:*
{answer}

*SQL Query:*
```sql
{sql}
```

*Results:* ({len(display_df)} of {total_rows} rows)

{display_df.to_string(index=False)}
"""

def process_query(file, question):
    if file is None:
        return "Please upload a file first"
    if not question.strip():
        return "Please enter a question"
    
    return ask_question(question, file.name)

demo = gr.Interface(
    fn=process_query,
    inputs=[
        gr.File(label="Upload CSV or Excel", file_types=[".csv", ".xlsx", ".xls"]),
        gr.Textbox(label="Ask Question", placeholder="What is the total sales?", lines=2)
    ],
    outputs=gr.Markdown(label="Answer"),
    title="Fast Chat with CSV/Excel (Polars + DuckDB)",
    description="10x faster file loading with Polars!",
    examples=[
        [None, "What is the total revenue?"],
        [None, "Show top 5 products by sales"],
        [None, "What is the average price?"]
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    default_file = r"C:\Users\AkshayRedekar\OneDrive - VE3\Documents\POC-EXCEL\electronic_data.xlsx"
    if os.path.exists(default_file):
        load_file(default_file)
        print(f"Pre-loaded: {default_file}")
    
    print("Starting Fast Chat App with Polars...")
    demo.launch(share=False)
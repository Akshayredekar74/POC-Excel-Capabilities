import os
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
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        return None
    
    conn.register("data", df)
    return df

def ask_question(question, file_path=None):
    if file_path:
        df = load_file(file_path)
        if df is None:
            return "Error: Please provide a CSV or Excel file"
    
    schema_info = conn.execute("SELECT * FROM data LIMIT 0").description
    columns = [col[0] for col in schema_info]
    
    sample_data = conn.execute("SELECT * FROM data LIMIT 3").fetchdf()
    
    schema_text = f"""Table: data
Columns: {', '.join(columns)}

Sample data:
{sample_data.to_string(index=False)}"""
    
    sql_prompt = f"""You are a SQL expert. Convert this question to a DuckDB SQL query.

{schema_text}

Question: {question}

Important rules:
- Use table name 'data'
- Return ONLY the SQL query
- No explanations or markdown
- Use proper DuckDB syntax

SQL Query:"""

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
        return f"""*Error executing SQL:*
{str(e)}

*Generated SQL:*
```sql
{sql}
```

Please try rephrasing your question."""
    
    display_df = result_df.head(20)
    total_rows = len(result_df)
    
    answer_prompt = f"""Question: {question}

SQL query executed: {sql}

Results (showing {len(display_df)} of {total_rows} rows):
{display_df.to_string(index=False)}

Provide a clear, concise answer to the question based on these results."""

    answer_response = model.generate_content(answer_prompt)
    answer = answer_response.text
    
    result_text = f"""*Answer:*
{answer}

*SQL Query:*
```sql
{sql}
```

*Results:* (showing {len(display_df)} of {total_rows} total rows)

{display_df.to_string(index=False)}
"""
    
    return result_text

def process_query(file, question):
    if file is None:
        return "Please upload a CSV or Excel file first"
    
    if not question or question.strip() == "":
        return "Please enter a question"
    
    try:
        return ask_question(question, file.name)
    except Exception as e:
        return f"Error: {str(e)}"

demo = gr.Interface(
    fn=process_query,
    inputs=[
        gr.File(label="Upload CSV or Excel File", file_types=[".csv", ".xlsx", ".xls"]),
        gr.Textbox(
            label="Ask your question",
            placeholder="Example: What is the total sales?",
            lines=2
        )
    ],
    outputs=gr.Markdown(label="Answer"),
    title="Chat with CSV/Excel Files",
    description="Upload your data file and ask questions in plain English!",
    # examples=[
    #     [None, "What is the total revenue?"],
    #     [None, "Show me the top 5 products by sales"],
    #     [None, "What is the average price?"],
    #     [None, "How many rows are in the data?"]
    # ],
    theme=gr.themes.Glass()
)

if __name__ == "__main__":
    # default_file = r"d:\Lens\BusinessCases\Sales_analysis_Electronic Store\Sales_January_2019.csv"
    # if os.path.exists(default_file):
    #     load_file(default_file)
    #     print(f"Pre-loaded: {default_file}")
        
    #     schema_info = conn.execute("SELECT * FROM data LIMIT 0").description
    #     columns = [col[0] for col in schema_info]
    #     print(f"Columns: {', '.join(columns)}")
    
    print("Starting app...")
    demo.launch(share=False)
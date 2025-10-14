import os
import pandas as pd
import numpy as np
import gradio as gr
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash-exp")

global_df = None

def load_file(file_path):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path)
    return None

def execute_pandas_code(code_string, df):
    local_vars = {'df': df, 'pd': pd, 'np': np}
    try:
        exec(code_string, globals(), local_vars)
        result = local_vars.get('result', None)
        return result
    except Exception as e:
        return f"Execution Error: {str(e)}"

def ask_question(question, file_path=None):
    global global_df

    if file_path:
        df = load_file(file_path)
        if df is None:
            return "Error: Invalid file format. Use CSV or Excel."
        global_df = df
    elif global_df is None:
        return "Error: No data file loaded."

    df = global_df

    prompt = f"""
You are a data analyst. The user will ask questions about a pandas DataFrame called 'df'.
Generate executable pandas/numpy code to answer the question and store the result in a variable called 'result'.

DataFrame info:
- Shape: {df.shape}
- Columns: {list(df.columns)}
- First 3 rows:
{df.head(3).to_string()}

Question: {question}

Generate ONLY the python code that:
1. Uses pandas/numpy operations on 'df'
2. Stores the final answer in variable 'result'
3. Handles basic data analysis tasks

Code requirements:
- Use proper pandas syntax
- Include necessary imports (already available: pd, np)
- Result should be a string, number, or pandas object
- No print statements, only assign to 'result'

Return ONLY the code without any explanation.
"""

    response = model.generate_content(prompt)
    code = response.text.strip()
    
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]
    
    executed_result = execute_pandas_code(code, df)
    
    if executed_result is None:
        return f"Error: No result generated from code.\nGenerated Code:\n{code}"
    
    if isinstance(executed_result, str) and "Execution Error" in executed_result:
        return f"Code Execution Failed:\n{executed_result}\n\nGenerated Code:\n{code}"
    
    result_str = str(executed_result)
    if hasattr(executed_result, 'to_string'):
        result_str = executed_result.to_string()
    
    return f"**Question:** {question}\n\n**Generated Code:**\n```python\n{code}\n```\n\n**Result:**\n{result_str}"

def process_query(file, question):
    if file is None and global_df is None:
        return "Please upload a data file first."
    if not question.strip():
        return "Please enter a question."
    
    return ask_question(question, file.name if file else None)

demo = gr.Interface(
    fn=process_query,
    inputs=[
        gr.File(label="Upload CSV or Excel File", file_types=[".csv", ".xlsx", ".xls"]),
        gr.Textbox(label="Question", placeholder="e.g., What is the average sales by category? Show top 5 products.", lines=2)
    ],
    outputs=gr.Markdown(label="Analysis with Code Execution"),
    title="Pandas Code Generator & Executor",
    description="Ask questions and see generated pandas code executed on your data.",
)

if __name__ == "__main__":
    default_file = r"C:\Users\AkshayRedekar\OneDrive - VE3\Documents\POC-EXCEL\electronic_data.xlsx"
    if os.path.exists(default_file):
        global_df = load_file(default_file)
        print(f"Pre-loaded: {default_file}")
        print(f"Dataset shape: {global_df.shape}")

    demo.launch(share=False)
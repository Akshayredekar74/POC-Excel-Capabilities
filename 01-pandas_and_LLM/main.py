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
        return local_vars.get('result', None)
    except Exception as e:
        return f"Execution Error: {str(e)}"

def format_result(result):
    if result is None:
        return "No result generated."
    if isinstance(result, pd.DataFrame):
        return result.to_markdown(index=False)
    elif isinstance(result, pd.Series):
        return result.to_frame().to_markdown(index=False)
    else:
        return str(result)

def load_and_analyze(file):
    global global_df
    
    try:
        if file.name.endswith('.csv'):
            global_df = pd.read_csv(file.name)
        else:
            global_df = pd.read_excel(file.name)
        
        file_status = f"✓ Loaded: {os.path.basename(file.name)} | {global_df.shape[0]} rows × {global_df.shape[1]} columns"
        
        cleaning_info = f"**Data Quality:**\n- Missing values: {global_df.isnull().sum().sum()}\n- Duplicates: {global_df.duplicated().sum()}\n- Data types:\n```\n{global_df.dtypes.to_string()}\n```"
        
        schema_info = f"**Sample Data:**\n```\n{global_df.head(10).to_string()}\n```"
        
        return file_status, cleaning_info, schema_info
    except Exception as e:
        return f"Error loading file: {str(e)}", "", ""

def process_query(question):
    global global_df
    
    if global_df is None:
        return "Please load a file first."
    if not question.strip():
        return "Please enter a question."
    
    try:
        prompt = f"""You are a data analyst. Generate executable pandas/numpy code to answer this question about a DataFrame called 'df'.

DataFrame info:
- Shape: {global_df.shape}
- Columns: {list(global_df.columns)}
- Data Types: {global_df.dtypes.to_dict()}
- First 3 rows:
{global_df.head(3).to_string()}

Question: {question}

Generate ONLY Python code that:
1. Uses pandas/numpy operations on 'df'
2. Stores the final answer in variable 'result'
3. No print statements, only assign to 'result'

Return ONLY the code without any explanation or markdown."""

        response = model.generate_content(prompt)
        code = response.text.strip()
        
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
        
        executed_result = execute_pandas_code(code, global_df)
        
        if isinstance(executed_result, str) and "Execution Error" in executed_result:
            return f"**Question:** {question}\n\n**Generated Code:**\n```python\n{code}\n```\n\n**Error:** {executed_result}"
        
        formatted_result = format_result(executed_result)
        
        return f"""**Question:** {question}

**Generated Code:**
```python
{code}
```

**Result:**
{formatted_result}"""
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks(title="Smart Data Analyst", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Smart Data Analyst
    ### Powered by Gemini + Pandas
    
    Upload your CSV/Excel file and ask questions in natural language.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Step 1: Upload File")
            file_input = gr.File(label="Select CSV or Excel File", file_types=[".csv", ".xlsx", ".xls"])
            load_btn = gr.Button("Load & Analyze Data", variant="primary", size="lg")
            file_status = gr.Markdown()
            
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Cleaning Report"):
                    cleaning_info = gr.Markdown()
                with gr.Tab("Schema & Sample Data"):
                    schema_info = gr.Markdown()
    
    gr.Markdown("### Step 2: Ask Questions")
    
    with gr.Row():
        question_input = gr.Textbox(
            label="Your Question",
            placeholder="Example: What are the top 10 products by profit margin?",
            lines=3,
            scale=4
        )
        submit_btn = gr.Button("Get Answer", variant="primary", size="lg", scale=1)
    
    output = gr.Markdown(label="Results")
    
    gr.Markdown("### Example Questions")
    gr.Examples(
        examples=[
            ["What is the total revenue?"],
            ["Show me the top 10 products by profit margin"],
            ["What is the average unit price?"],
            ["How many unique products are there?"],
            ["Which product category has the highest sales?"],
            ["What is the profit margin for each product?"]
        ],
        inputs=question_input
    )
    
    load_btn.click(fn=load_and_analyze, inputs=file_input, outputs=[file_status, cleaning_info, schema_info])
    submit_btn.click(fn=process_query, inputs=question_input, outputs=output)
    question_input.submit(fn=process_query, inputs=question_input, outputs=output)

if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
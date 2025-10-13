
import os
import pandas as pd
import numpy as np
import gradio as gr
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")
global_df = None


def load_file(file_path):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path)
    return None


def format_result(result):
    if result is None:
        return "No result generated."

    if isinstance(result, (pd.DataFrame, pd.Series)):
        if isinstance(result, pd.Series):
            result = result.to_frame()
        return result.to_markdown(index=False)
    elif isinstance(result, (int, float, str, bool)):
        return str(result)
    else:
        return str(result)


def execute_pandas_code(code_string, df):
    local_vars = {'df': df, 'pd': pd, 'np': np}
    try:
        exec(code_string, {}, local_vars)
        return local_vars.get('result', None)
    except Exception as e:
        return f"Execution Error: {str(e)}"


def generate_natural_language_answer(question, code, result, data_context):
    prompt = f"""
Based on the following analysis, provide a clear and concise natural language answer.

Question: {question}

Executed Code:
```python
{code}
```

Result:
{result}

Dataset Context:
{data_context}

Guidelines:

* Directly answer the question
* Highlight specific numbers or trends from the result
* Explain findings in plain English (no technical jargon)
* Keep it brief and clear
"""
    response = model.generate_content(prompt)
    return response.text.strip()


def ask_question(question, file_path=None):
    global global_df

    if file_path:
        df = load_file(file_path)
        if df is None:
            return "Invalid file format. Please upload a CSV or Excel file."
        global_df = df
    elif global_df is None:
        return "No dataset loaded. Please upload a file first."

    df = global_df
    data_context = f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns. Columns: {list(df.columns)}"

    code_prompt = f"""
You are a data analyst working with a pandas DataFrame named 'df'.

DataFrame Details:

* Shape: {df.shape}
* Columns: {list(df.columns)}
* Data Types: {df.dtypes.to_dict()}
* First 2 rows:
{df.head(2).to_string()}

Question: {question}

Write executable pandas/numpy code that:

1. Answers the question directly
2. Assigns the final output to variable 'result'
3. Uses proper pandas operations (groupby, aggregation, filtering, etc.)
4. Produces a clean, displayable output (DataFrame, Series, or summary)
5. No print statements or explanations — only code.

Return ONLY the code (no markdown, no text).
"""
    response = model.generate_content(code_prompt)
    code = response.text.strip()

    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()

    executed_result = execute_pandas_code(code, df)

    if executed_result is None:
        return f"No result generated.\n\n**Generated Code:**\n```python\n{code}\n```"
    if isinstance(executed_result, str) and "Execution Error" in executed_result:
        return f"Code Execution Failed:\n```\n{executed_result}\n```\n\n**Generated Code:**\n```python\n{code}\n```"

    formatted_result = format_result(executed_result)
    natural_answer = generate_natural_language_answer(question, code, formatted_result, data_context)

    return f"""
### Question

{question}

### Generated Pandas Code

```python
{code}
```

### Execution Result

```
{formatted_result}
```

### Natural Language Explanation

{natural_answer}
"""


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
        gr.Textbox(
            label="Ask a question",
            placeholder="e.g., Show total revenue by product category",
            lines=2,
        ),
    ],
    outputs=gr.Markdown(label="Analysis Results"),
    title="Smart Data Analyst (Pandas + Gemini)",
    description="Ask natural language questions about your data. The LLM generates pandas code, executes it, and explains results in plain English.",
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    default_file = r"C:\Users\AkshayRedekar\OneDrive - VE3\Documents\POC-EXCEL\electronic_data.xlsx"
    if os.path.exists(default_file):
        global_df = load_file(default_file)
        print(f"Pre-loaded: {default_file}")
        print(f"Dataset shape: {global_df.shape}")

    print("Starting Gradio app...")
    demo.launch(share=False)

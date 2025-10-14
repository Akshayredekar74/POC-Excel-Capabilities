"""
Chat with CSV/Excel using Google Gemini + Pandas
No DuckDB, no SQL — Direct reasoning on data
"""

import os
import pandas as pd
import gradio as gr
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash-exp")

# Global variable to hold loaded DataFrame
global_df = None

def load_file(file_path):
    """Load CSV or Excel file into a Pandas DataFrame"""
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)
    else:
        return None
    return df

def summarize_dataframe(df: pd.DataFrame, max_rows: int = 5):
    """Generate concise summary of the dataset for LLM context"""
    info = f"Rows: {len(df)}, Columns: {len(df.columns)}\n"
    info += f"Columns: {', '.join(df.columns.tolist())}\n\n"
    sample = df.head(max_rows).to_string(index=False)
    return info + f"Sample data (first {max_rows} rows):\n{sample}"

def ask_question(question, file_path=None):
    """Ask Gemini question directly on Pandas data"""
    global global_df

    if file_path:
        df = load_file(file_path)
        if df is None:
            return "Error: Please provide a valid CSV or Excel file"
        global_df = df
    elif global_df is None:
        return "Error: Please upload a file first"

    df = global_df

    # Prepare concise dataset summary
    data_context = summarize_dataframe(df)

    # Build prompt
    prompt = f"""
You are a data analyst. Answer the user's question using the provided dataset.

Dataset Summary:
{data_context}

User Question: {question}

Rules:
- Use reasoning directly from the data.
- Perform simple aggregations, filtering, or trends if needed.
- Respond clearly and factually.
- If data is insufficient, say so.
"""

    # Send to Gemini
    response = model.generate_content(prompt)
    answer = response.text.strip()

    return f"""**Question:** {question}

**Answer:**  
{answer}

**Data Summary:**  
{data_context}
"""

def process_query(file, question):
    """Handle Gradio UI events"""
    if file is None and (global_df is None):
        return "Please upload a CSV or Excel file first."
    if not question or question.strip() == "":
        return "Please enter a question."
    try:
        return ask_question(question, file.name if file else None)
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Interface
demo = gr.Interface(
    fn=process_query,
    inputs=[
        gr.File(label="Upload CSV or Excel File", file_types=[".csv", ".xlsx", ".xls"]),
        gr.Textbox(label="Ask your question", placeholder="Example: What is the total sales?", lines=2)
    ],
    outputs=gr.Markdown(label="Answer"),
    title="Chat with CSV/Excel Files (Direct LLM + Pandas)",
    description="Ask questions directly about your data using Google Gemini. No SQL, no DuckDB — just pure LLM reasoning.",
)

if __name__ == "__main__":
    default_file = r"C:\Users\AkshayRedekar\OneDrive - VE3\Documents\POC-EXCEL\electronic_data.xlsx"
    if os.path.exists(default_file):
        global_df = load_file(default_file)
        print(f"Pre-loaded: {default_file}")
        print(f"Columns: {', '.join(global_df.columns)}")

    print("Starting direct LLM + Pandas app...")
    demo.launch(share=False)

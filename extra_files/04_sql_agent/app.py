"""
Gradio App for AI Agent with SQL Database
"""

import os
import gradio as gr
from agent import SQLAgent

agent = SQLAgent()
current_file = None

def load_file(file_path):
    global current_file
    if file_path is None:
        return "Please upload a file", "", ""
    
    df = agent.load_data(file_path.name)
    if df is None:
        return "Failed to load file. Please upload CSV or Excel file", "", ""
    
    current_file = file_path.name
    
    # Get schema info
    result = agent.query("dummy")
    
    # Format status
    status = f"""### File Loaded Successfully

**Total Rows:** {len(df)}  
**Total Columns:** {len(df.columns)}  
**File:** {os.path.basename(file_path.name)}
"""
    
    # Format cleaning report
    cleaning = f"""### Data Cleaning Report

{result['cleaning']}
"""
    
    # Format schema
    schema = f"""### Table Schema

{result['schema']}
"""
    
    return status, cleaning, schema

def process_query(question):
    if current_file is None:
        return "**Error:** Please upload a file first"
    
    if not question.strip():
        return "**Error:** Please enter a question"
    
    result = agent.query(question)
    
    # Format response
    response = f"""## Answer

{result['answer']}

---

## SQL Query Generated

```sql
{result['sql']}
```

**Attempts:** {result['iterations']} / 3

---

## Query Results

```
{result['result'][:2500]}
```
"""
    
    if result['error']:
        response += f"""

---

## Error Details

{result['error']}
"""
    
    return response

# Create Gradio interface
with gr.Blocks(title="AI SQL Agent", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # AI Agent with SQL Database
    ### Powered by LangGraph + Gemini + DuckDB
    
    Upload your CSV/Excel file and ask questions in natural language.  
    The agent automatically detects data types and cleans the data.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Step 1: Upload File")
            file_input = gr.File(
                label="Select CSV or Excel File",
                file_types=[".csv", ".xlsx", ".xls"]
            )
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
    
    # Event handlers
    load_btn.click(
        fn=load_file,
        inputs=file_input,
        outputs=[file_status, cleaning_info, schema_info]
    )
    
    submit_btn.click(
        fn=process_query,
        inputs=question_input,
        outputs=output
    )
    
    question_input.submit(
        fn=process_query,
        inputs=question_input,
        outputs=output
    )

if __name__ == "__main__":
    # Try to pre-load default file
    default_file = r"C:\Users\AkshayRedekar\OneDrive - VE3\Documents\POC-EXCEL\electronic_data.xlsx"
    if os.path.exists(default_file):
        agent.load_data(default_file)
        current_file = default_file
        print(f"Pre-loaded: {default_file}")
    
    print("\n" + "="*50)
    print("Starting AI SQL Agent with Auto Type Detection")
    print("="*50 + "\n")
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)

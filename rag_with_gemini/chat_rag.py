"""
RAG Pipeline with LanceDB (Local Vector Database)
Polars + LanceDB + Gemini
"""

import os
import polars as pl
import pandas as pd
import gradio as gr
from dotenv import load_dotenv
import google.generativeai as genai
import lancedb

load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-2.0-flash-exp')

db = lancedb.connect("./.lancedb")
table = None
dataframe = None

def embed_text(texts):
    response = genai.embed_content(
        model="models/embedding-001",
        content=texts,
        task_type="retrieval_document"
    )
    return [item['embedding'] for item in response['embedding']]

def load_file(file_path):
    global dataframe, table
    
    if file_path.endswith('.csv'):
        try:
            df = pl.read_csv(file_path, infer_schema_length=10000)
        except:
            df = pl.read_csv(file_path, encoding="utf8-lossy", infer_schema_length=10000)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pl.read_excel(file_path)
    else:
        return None
    
    dataframe = df.to_pandas()
    
    docs = []
    for idx, row in dataframe.iterrows():
        row_text = " ".join([f"{col}: {val}" for col, val in row.items()])
        docs.append(row_text)
    
    embeddings = embed_text(docs)
    
    data = []
    for idx, (doc, embedding) in enumerate(zip(docs, embeddings)):
        data.append({
            "id": idx,
            "text": doc,
            "embedding": embedding
        })
    
    table = db.create_table("documents", data=data, mode="overwrite")
    
    return df

def retrieve_relevant_rows(question, top_k=5):
    query_embedding = embed_text([question])[0]
    
    results = table.search(query_embedding).limit(top_k).to_list()
    
    row_indices = [result['id'] for result in results]
    relevant_data = dataframe.iloc[row_indices]
    retrieved_docs = [result['text'] for result in results]
    
    return relevant_data, retrieved_docs

def ask_question(question):
    relevant_data, chunks = retrieve_relevant_rows(question, top_k=5)
    
    context = f"""Dataset Schema:
Columns: {', '.join(dataframe.columns)}

Relevant Data:
{relevant_data.to_string(index=False)}

Original Chunks:
{chr(10).join(chunks)}"""
    
    rag_prompt = f"""You are a data analyst. Answer the following question based on the provided data context.

Context:
{context}

Question: {question}

Provide a clear and concise answer."""
    
    response = model.generate_content(rag_prompt)
    answer = response.text
    
    return f"""*Answer:*
{answer}

*Retrieved Data:* ({len(relevant_data)} rows)

{relevant_data.to_string(index=False)}
"""

def process_query(file, question):
    if file is None:
        return "Please upload a file first"
    if not question.strip():
        return "Please enter a question"
    
    if table is None:
        load_file(file.name)
    
    return ask_question(question)

demo = gr.Interface(
    fn=process_query,
    inputs=[
        gr.File(label="Upload CSV or Excel", file_types=[".csv", ".xlsx", ".xls"]),
        gr.Textbox(label="Ask Question", placeholder="What is the average price?", lines=2)
    ],
    outputs=gr.Markdown(label="Answer"),
    title="RAG Chat with CSV/Excel (Polars + LanceDB + Gemini)",
    description="Local vector database powered semantic search",
    examples=[
        [None, "What are the top products?"],
        [None, "Show expensive items"],
        [None, "What is the distribution of categories?"]
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    print("Starting RAG Chat App with LanceDB...")
    demo.launch(share=False)
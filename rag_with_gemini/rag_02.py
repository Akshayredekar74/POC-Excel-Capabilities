import os
import polars as pl
import pandas as pd
import gradio as gr
from dotenv import load_dotenv
import requests
import lancedb
from groq import Groq

load_dotenv()

ollama_embed_url = "http://localhost:11434/api/embed"
ollama_model = "embeddinggemma:latest"

db = lancedb.connect("./.lancedb")
table = None
dataframe = None

client = Groq()

def embed_text(texts):
    embeddings = []
    for text in texts:
        response = requests.post(
            ollama_embed_url,
            json={"model": ollama_model, "input": text}
        )
        res_json = response.json()
        if 'embeddings' not in res_json:
            raise ValueError(f"Embedding response error: {res_json}")
        embedding = res_json['embeddings'][0]
        embeddings.append(embedding)
    return embeddings

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
    
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": rag_prompt}],
        temperature=1,
        max_completion_tokens=8192,
        top_p=1,
        reasoning_effort="medium",
        stream=False
    )
    
    answer = completion.choices[0].message.content
    
    return f"""*Answer:*
{answer}

*Retrieved Data:* ({len(relevant_data)} rows)

{relevant_data.to_string(index=False)}
"""

def process_query(file, question):
    global table, dataframe
    if file is None:
        return "Please upload a file first"
    if not question.strip():
        return "Please enter a question"
    if table is None or dataframe is None:
        file_path = getattr(file, "name", None)
        if hasattr(file, "file"):
            file_path = file.file.name
        if file_path is None:
            return "Failed to read uploaded file"
        load_file(file_path)
    return ask_question(question)

demo = gr.Interface(
    fn=process_query,
    inputs=[
        gr.File(label="Upload CSV or Excel", file_types=[".csv", ".xlsx", ".xls"]),
        gr.Textbox(label="Ask Question", placeholder="Enter your query here", lines=2)
    ],
    outputs=gr.Markdown(label="Answer"),
    title="RAG Chat with CSV/Excel (LanceDB + Groq + Ollama)",
    description="Local embeddings + Groq LLM powered semantic search",
    examples=[
        [None, "What are the top hospitals?"],
        [None, "Show hospitals in Florida"],
        [None, "Count of hospitals by type"]
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    print("Starting RAG Chat App with Groq + Ollama...")
    demo.launch(share=False)

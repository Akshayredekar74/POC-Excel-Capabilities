
# import os
# import polars as pl
# import pandas as pd
# import gradio as gr
# from dotenv import load_dotenv
# import requests
# import lancedb
# from groq import Groq
# import time

# # Load environment variables
# load_dotenv()

# # Configuration
# ollama_embed_url = "http://localhost:11434/api/embed"
# # ollama_model = "embeddinggemma:latest"
# ollama_model = "nomic-embed-text:latest"

# # Initialize database and global variables
# db = lancedb.connect("./.lancedb")
# table = None
# dataframe = None

# # Initialize Groq client
# groq_client = Groq()

# def check_ollama_health():
#     """Check if Ollama is running and the model is available."""
#     try:
#         response = requests.get("http://localhost:11434/api/tags", timeout=10)
#         if response.status_code == 200:
#             models = response.json().get('models', [])
#             model_names = [model['name'] for model in models]
#             print(f"Available Ollama models: {model_names}")
#             if ollama_model not in model_names:
#                 print(f"Warning: {ollama_model} not found in available models")
#             return True
#         return False
#     except Exception as e:
#         print(f"Ollama health check failed: {e}")
#         return False

# def embed_text(texts):
#     """Generate embeddings for a list of texts using Ollama."""
#     embeddings = []
#     for i, text in enumerate(texts):
#         try:
#             # Truncate very long texts to avoid timeout
#             truncated_text = text[:10000] if len(text) > 10000 else text
            
#             response = requests.post(
#                 ollama_embed_url,
#                 json={"model": ollama_model, "input": truncated_text},
#                 timeout=60
#             )
#             response.raise_for_status()
#             res_json = response.json()
            
#             if 'embeddings' not in res_json:
#                 print(f"Unexpected response format: {res_json}")
#                 raise ValueError(f"Embedding response error: {res_json}")
            
#             embedding = res_json['embeddings'][0]
#             embeddings.append(embedding)
            
#             # Progress indicator for large files
#             if (i + 1) % 100 == 0:
#                 print(f"Embedded {i + 1}/{len(texts)} documents")
                
#         except requests.RequestException as e:
#             print(f"Ollama request failed for text {i}: {e}")
#             # Return a zero vector as fallback
#             embeddings.append([0.0] * 1024)  # Adjust size based on your model
#         except Exception as e:
#             print(f"Unexpected error during embedding: {e}")
#             embeddings.append([0.0] * 1024)
    
#     return embeddings

# def load_file(file_path):
#     """Load CSV or Excel file and create LanceDB table with embeddings."""
#     global dataframe, table
#     try:
#         print(f"Loading file: {file_path}")
        
#         # Check file size
#         file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
#         print(f"File size: {file_size:.2f} MB")
        
#         if file_path.endswith('.csv'):
#             try:
#                 df = pl.read_csv(file_path, infer_schema_length=10000)
#             except Exception as e:
#                 print(f"CSV read failed, trying with different encoding: {e}")
#                 df = pl.read_csv(file_path, encoding="utf8-lossy", infer_schema_length=10000)
#         elif file_path.endswith(('.xlsx', '.xls')):
#             df = pl.read_excel(file_path)
#         else:
#             print(f"Unsupported file format: {file_path}")
#             return None
        
#         print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
#         print(f"Columns: {df.columns}")
        
#         # Sample first few rows for debugging
#         print("Sample data:")
#         print(df.head(3))
        
#         dataframe = df.to_pandas()
        
#         # Create document texts
#         docs = []
#         for idx, row in dataframe.iterrows():
#             row_text = " ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
#             docs.append(row_text)
        
#         print(f"Generating embeddings for {len(docs)} documents...")
#         start_time = time.time()
        
#         embeddings = embed_text(docs)
        
#         end_time = time.time()
#         print(f"Embedding generation took {end_time - start_time:.2f} seconds")
        
#         # Create LanceDB table
#         data = [{"id": idx, "text": doc, "embedding": emb} for idx, (doc, emb) in enumerate(zip(docs, embeddings))]
        
#         table = db.create_table("documents", data=data, mode="overwrite")
#         print(f"Created LanceDB table with {len(data)} documents")
        
#         return df
        
#     except Exception as e:
#         print(f"Error loading file: {e}")
#         import traceback
#         traceback.print_exc()
#         return None

# def retrieve_relevant_rows(question, top_k=5):
#     """Retrieve top-k relevant rows from LanceDB based on query embedding."""
#     try:
#         print(f"Retrieving relevant rows for: {question}")
#         query_embedding = embed_text([question])[0]
#         results = table.search(query_embedding).limit(top_k).to_list()
        
#         if not results:
#             print("No results found from vector search")
#             return None, []
            
#         row_indices = [result['id'] for result in results]
#         relevant_data = dataframe.iloc[row_indices]
#         retrieved_docs = [result['text'] for result in results]
        
#         print(f"Retrieved {len(relevant_data)} relevant rows")
#         return relevant_data, retrieved_docs
        
#     except Exception as e:
#         print(f"Error in retrieve_relevant_rows: {e}")
#         return None, []

# def ask_question(question):
#     """Answer a question using Groq with RAG context."""
#     try:
#         print(f"Processing question: {question}")
        
#         relevant_data, chunks = retrieve_relevant_rows(question, top_k=5)
        
#         if relevant_data is None or relevant_data.empty:
#             return "No relevant data found for your question. Please try a different query."
        
#         # Prepare context
#         context = f"""Dataset Schema:
# Columns: {', '.join(dataframe.columns)}

# Relevant Data Samples:
# {relevant_data.head(10).to_string(index=False)}

# Number of relevant records: {len(relevant_data)}"""

#         rag_prompt = f"""You are a helpful data analyst. Answer the question based ONLY on the provided context.

# Context Data:
# {context}

# Question: {question}

# Instructions:
# - Answer concisely based only on the provided context
# - If the context doesn't contain relevant information, say so
# - Format numbers and facts clearly
# - Don't make up information not in the context"""

#         print("Sending request to Groq...")
        
#         completion = groq_client.chat.completions.create(
#             model="llama3-8b-8192",
#             messages=[{"role": "user", "content": rag_prompt}],
#             temperature=0.3,  # Lower temperature for more consistent answers
#             max_tokens=1024,
#             top_p=1,
#             stream=False,
#             stop=None
#         )
        
#         answer = completion.choices[0].message.content
        
#         return f"""**Answer:**
# {answer}

# **Retrieved Data Preview:**
# ```
# {relevant_data.head(3).to_string(index=False)}
# ```

# *Showing {min(3, len(relevant_data))} of {len(relevant_data)} relevant records*"""
        
#     except Exception as e:
#         error_msg = f"Error processing question: {str(e)}"
#         print(error_msg)
#         import traceback
#         traceback.print_exc()
#         return error_msg

# def process_query(file, question):
#     """Handle file upload and question processing for Gradio."""
#     global table, dataframe
#     try:
#         print("Processing query...")
        
#         if file is None:
#             return "Please upload a CSV or Excel file first."
        
#         if not question.strip():
#             return "Please enter a question about your data."
        
#         # Check Ollama health
#         if not check_ollama_health():
#             return "Error: Ollama is not running or accessible. Please start Ollama first."
        
#         file_path = file.name
#         print(f"File path: {file_path}")
        
#         # Load file if not already loaded
#         if table is None or dataframe is None:
#             print("Loading file for the first time...")
#             result = load_file(file_path)
#             if result is None:
#                 return "Failed to load the file. Please check if it's a valid CSV or Excel file."
        
#         return ask_question(question)
        
#     except Exception as e:
#         error_msg = f"Unexpected error: {str(e)}"
#         print(error_msg)
#         import traceback
#         traceback.print_exc()
#         return error_msg

# # Create Gradio interface
# with gr.Blocks(theme=gr.themes.Soft()) as demo:
#     gr.Markdown("# RAG Chat with CSV/Excel Files")
#     gr.Markdown("Upload your CSV or Excel file and ask questions about your data using local embeddings + Groq LLM")
    
#     with gr.Row():
#         with gr.Column(scale=1):
#             file_input = gr.File(
#                 label="Upload CSV or Excel File", 
#                 file_types=[".csv", ".xlsx", ".xls"],
#                 type="filepath"
#             )
#             status = gr.Textbox(label="Status", interactive=False)
        
#         with gr.Column(scale=2):
#             question_input = gr.Textbox(
#                 label="Ask Question", 
#                 placeholder="e.g., What are the top categories? Show trends over time...",
#                 lines=3
#             )
#             submit_btn = gr.Button("Ask Question", variant="primary")
#             output = gr.Markdown(label="Answer")
    
#     # Examples
#     gr.Examples(
#         examples=[
#             ["What are the main categories in the data?"],
#             ["Show summary statistics"],
#             ["What patterns or trends do you see?"],
#             ["What are the top 5 items by value?"]
#         ],
#         inputs=question_input
#     )
    
#     def update_status(file):
#         if file is None:
#             return "Please upload a file"
#         try:
#             df = pl.read_csv(file) if file.endswith('.csv') else pl.read_excel(file)
#             return f"File loaded: {len(df)} rows, {len(df.columns)} columns"
#         except:
#             return "File uploaded - will process when you ask a question"
    
#     file_input.change(update_status, inputs=file_input, outputs=status)
#     submit_btn.click(process_query, inputs=[file_input, question_input], outputs=output)
#     question_input.submit(process_query, inputs=[file_input, question_input], outputs=output)

# if __name__ == "__main__":
#     print("Starting RAG Chat App with Groq + Ollama...")
#     print("Checking Ollama health...")
    
#     if check_ollama_health():
#         print("Ollama is running ✓")
#     else:
#         print("Warning: Ollama may not be running properly")
    
#     demo.launch(share=False, server_port=7860)




import os
import polars as pl
import pandas as pd
import gradio as gr
from dotenv import load_dotenv
import requests
import lancedb
from groq import Groq
import time

# Load environment variables (e.g., Groq API key)
load_dotenv()

# Configuration
ollama_embed_url = "http://localhost:11434/api/embed"
ollama_model = "nomic-embed-text:latest"

# Initialize database and global variables
db = lancedb.connect("./.lancedb")
table = None
dataframe = None

# Initialize Groq client
groq_client = Groq()

def check_ollama_health():
    """Check if Ollama is running and the model is available."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            print(f"Available Ollama models: {model_names}")
            if ollama_model not in model_names:
                print(f"Warning: {ollama_model} not found in available models")
            return True
        return False
    except Exception as e:
        print(f"Ollama health check failed: {e}")
        return False

def embed_text(texts):
    """Generate embeddings for a list of texts using Ollama."""
    embeddings = []
    for i, text in enumerate(texts):
        try:
            # Truncate very long texts to avoid timeout
            truncated_text = text[:10000] if len(text) > 10000 else text
            
            response = requests.post(
                ollama_embed_url,
                json={"model": ollama_model, "input": truncated_text},
                timeout=60
            )
            response.raise_for_status()
            res_json = response.json()
            
            if 'embeddings' not in res_json:
                print(f"Unexpected response format: {res_json}")
                raise ValueError(f"Embedding response error: {res_json}")
            
            embedding = res_json['embeddings'][0]
            embeddings.append(embedding)
            
            # Progress indicator for large files
            if (i + 1) % 100 == 0:
                print(f"Embedded {i + 1}/{len(texts)} documents")
                
        except requests.RequestException as e:
            print(f"Ollama request failed for text {i}: {e}")
            # Return a zero vector as fallback
            embeddings.append([0.0] * 1024)  # Adjust size based on your model
        except Exception as e:
            print(f"Unexpected error during embedding: {e}")
            embeddings.append([0.0] * 1024)
    
    return embeddings

def load_file(file_path):
    """Load CSV or Excel file and create LanceDB table with embeddings."""
    global dataframe, table
    try:
        print(f"Loading file: {file_path}")
        
        # Check file size
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"File size: {file_size:.2f} MB")
        
        if file_path.endswith('.csv'):
            try:
                df = pl.read_csv(file_path, infer_schema_length=10000)
            except Exception as e:
                print(f"CSV read failed, trying with different encoding: {e}")
                df = pl.read_csv(file_path, encoding="utf8-lossy", infer_schema_length=10000)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pl.read_excel(file_path)
        else:
            print(f"Unsupported file format: {file_path}")
            return None
        
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {df.columns}")
        
        # Sample first few rows for debugging
        print("Sample data:")
        print(df.head(3))
        
        dataframe = df.to_pandas()
        
        # Create document texts
        docs = []
        for idx, row in dataframe.iterrows():
            row_text = " ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            docs.append(row_text)
        
        print(f"Generating embeddings for {len(docs)} documents...")
        start_time = time.time()
        
        embeddings = embed_text(docs)
        
        end_time = time.time()
        print(f"Embedding generation took {end_time - start_time:.2f} seconds")
        
        # Create LanceDB table
        data = [{"id": idx, "text": doc, "embedding": emb} for idx, (doc, emb) in enumerate(zip(docs, embeddings))]
        
        table = db.create_table("documents", data=data, mode="overwrite")
        print(f"Created LanceDB table with {len(data)} documents")
        
        return df
        
    except Exception as e:
        print(f"Error loading file: {e}")
        import traceback
        traceback.print_exc()
        return None

def retrieve_relevant_rows(question, top_k=5):
    """Retrieve top-k relevant rows from LanceDB based on query embedding."""
    try:
        print(f"Retrieving relevant rows for: {question}")
        query_embedding = embed_text([question])[0]
        results = table.search(query_embedding).limit(top_k).to_list()
        
        if not results:
            print("No results found from vector search")
            return None, []
            
        row_indices = [result['id'] for result in results]
        relevant_data = dataframe.iloc[row_indices]
        retrieved_docs = [result['text'] for result in results]
        
        print(f"Retrieved {len(relevant_data)} relevant rows")
        return relevant_data, retrieved_docs
        
    except Exception as e:
        print(f"Error in retrieve_relevant_rows: {e}")
        return None, []

def ask_question(question):
    """Answer a question using Groq with RAG context."""
    try:
        print(f"Processing question: {question}")
        
        relevant_data, chunks = retrieve_relevant_rows(question, top_k=5)
        
        if relevant_data is None or relevant_data.empty:
            return "No relevant data found for your question. Please try a different query."
        
        # Prepare context
        context = f"""Dataset Schema:
Columns: {', '.join(dataframe.columns)}

Relevant Data Samples:
{relevant_data.head(10).to_string(index=False)}

Number of relevant records: {len(relevant_data)}"""

        rag_prompt = f"""You are a helpful data analyst. Answer the question based ONLY on the provided context.

Context Data:
{context}

Question: {question}

Instructions:
- Answer concisely based only on the provided context
- If the context doesn't contain relevant information, say so
- Format numbers and facts clearly
- Don't make up information not in the context"""

        print("Sending request to Groq...")
        
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": rag_prompt}],
            temperature=0.3,  # Lower temperature for more consistent answers
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None
        )
        
        answer = completion.choices[0].message.content
        
        return f"""**Answer:**
{answer}

**Retrieved Data Preview:**
```
{relevant_data.head(3).to_string(index=False)}
```

*Showing {min(3, len(relevant_data))} of {len(relevant_data)} relevant records*"""
        
    except Exception as e:
        error_msg = f"Error processing question: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg

def process_query(file, question):
    """Handle file upload and question processing for Gradio."""
    global table, dataframe
    try:
        print("Processing query...")
        
        if file is None:
            return "Please upload a CSV or Excel file first."
        
        if not question.strip():
            return "Please enter a question about your data."
        
        # Check Ollama health
        if not check_ollama_health():
            return "Error: Ollama is not running or accessible. Please start Ollama first."
        
        file_path = file.name
        print(f"File path: {file_path}")
        
        # Load file if not already loaded
        if table is None or dataframe is None:
            print("Loading file for the first time...")
            result = load_file(file_path)
            if result is None:
                return "Failed to load the file. Please check if it's a valid CSV or Excel file."
        
        return ask_question(question)
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# RAG Chat with CSV/Excel Files")
    gr.Markdown("Upload your CSV or Excel file and ask questions about your data using local embeddings + Groq LLM")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload CSV or Excel File", 
                file_types=[".csv", ".xlsx", ".xls"],
                type="filepath"
            )
            status = gr.Textbox(label="Status", interactive=False)
        
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Ask Question", 
                placeholder="e.g., What are the top categories? Show trends over time...",
                lines=3
            )
            submit_btn = gr.Button("Ask Question", variant="primary")
            output = gr.Markdown(label="Answer")
    
    # Examples
    gr.Examples(
        examples=[
            ["What are the main categories in the data?"],
            ["Show summary statistics"],
            ["What patterns or trends do you see?"],
            ["What are the top 5 items by value?"]
        ],
        inputs=question_input
    )
    
    def update_status(file):
        if file is None:
            return "Please upload a file"
        try:
            df = pl.read_csv(file) if file.endswith('.csv') else pl.read_excel(file)
            return f"File loaded: {len(df)} rows, {len(df.columns)} columns"
        except:
            return "File uploaded - will process when you ask a question"
    
    file_input.change(update_status, inputs=file_input, outputs=status)
    submit_btn.click(process_query, inputs=[file_input, question_input], outputs=output)
    question_input.submit(process_query, inputs=[file_input, question_input], outputs=output)

if __name__ == "__main__":
    print("Starting RAG Chat App with Groq + Ollama...")
    print("Checking Ollama health...")
    
    if check_ollama_health():
        print("Ollama is running ✓")
    else:
        print("Warning: Ollama may not be running properly")
    
    demo.launch(share=False, server_port=7860)

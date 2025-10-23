"""
RAG Pipeline with LlamaIndex and Zilliz Cloud
"""

import os
import json
import random
import textwrap
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.milvus import MilvusVectorStore
from pymilvus import MilvusClient, DataType

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")

if not all([OPENAI_API_KEY, ZILLIZ_URI, ZILLIZ_TOKEN]):
    raise ValueError("Missing required credentials in .env file")

Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

milvus_client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
COLLECTION_NAME = "customer_collection"

if milvus_client.has_collection(collection_name=COLLECTION_NAME):
    milvus_client.drop_collection(collection_name=COLLECTION_NAME)

schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=1536)
schema.add_field("text", DataType.VARCHAR, max_length=65535)

index_params = milvus_client.prepare_index_params()
index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")

milvus_client.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema,
    index_params=index_params
)

DATA_FOLDER = "data"
csv_files = list(Path(DATA_FOLDER).glob("*.csv"))

if not csv_files:
    Path(DATA_FOLDER).mkdir(exist_ok=True)
    sample_data = {
        "customer_id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Carol", "David", "Eve"],
        "total_spent": [5420.50, 3210.75, 8950.00, 1580.25, 6730.80],
    }
    pd.DataFrame(sample_data).to_csv(Path(DATA_FOLDER) / "customers.csv", index=False)
    csv_files = [Path(DATA_FOLDER) / "customers.csv"]

all_json_records = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    for idx, row in df.iterrows():
        all_json_records.append({
            "source_file": csv_file.name,
            "row_number": idx + 1,
            "data": row.to_dict()
        })

with open("customers_canonical.json", "w", encoding="utf-8") as f:
    json.dump(all_json_records, f, indent=2, ensure_ascii=False)

TARGET_CHUNK_SIZE = 1200
MAX_CHUNK_SIZE = 1500
chunks, current_chunk_text, current_chunk_records = [], "", []

for record in all_json_records:
    text = f"File: {record['source_file']}, Row: {record['row_number']}\n" + \
           "Data: " + ", ".join([f"{k}={v}" for k, v in record['data'].items()]) + "\n\n"
    if len(current_chunk_text) + len(text) > MAX_CHUNK_SIZE and current_chunk_text:
        chunks.append({
            "chunk_id": len(chunks) + 1,
            "text": current_chunk_text.strip(),
            "record_count": len(current_chunk_records),
            "metadata": {"source_files": list({r['source_file'] for r in current_chunk_records})}
        })
        current_chunk_text, current_chunk_records = text, [record]
    else:
        current_chunk_text += text
        current_chunk_records.append(record)
    if len(current_chunk_text) >= TARGET_CHUNK_SIZE and current_chunk_text:
        chunks.append({
            "chunk_id": len(chunks) + 1,
            "text": current_chunk_text.strip(),
            "record_count": len(current_chunk_records),
            "metadata": {"source_files": list({r['source_file'] for r in current_chunk_records})}
        })
        current_chunk_text, current_chunk_records = "", []

if current_chunk_text:
    chunks.append({
        "chunk_id": len(chunks) + 1,
        "text": current_chunk_text.strip(),
        "record_count": len(current_chunk_records),
        "metadata": {"source_files": list({r['source_file'] for r in current_chunk_records})}
    })

with open("Customers_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

embed_model = Settings.embed_model
chunk_embeddings = [embed_model.get_text_embedding(c["text"]) for c in chunks]

insert_data = [{"vector": e, "text": c["text"], "chunk_id": c["chunk_id"]} for c, e in zip(chunks, chunk_embeddings)]
milvus_client.insert(collection_name=COLLECTION_NAME, data=insert_data)

vector_store = MilvusVectorStore(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN, collection_name=COLLECTION_NAME, dim=1536)
documents = [Document(text=c["text"], metadata={"chunk_id": c["chunk_id"]}) for c in chunks]
index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)
query_engine = index.as_query_engine(similarity_top_k=3, response_mode="compact")

question = "Which customer spent the most money?"
response = query_engine.query(question)

print("Answer:", response.response)
for i, node in enumerate(response.source_nodes, 1):
    print(f"Source {i}: {node.text[:150]}...")

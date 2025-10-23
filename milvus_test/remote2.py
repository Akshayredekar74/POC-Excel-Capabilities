from dotenv import load_dotenv
import os
from pymilvus import MilvusClient, DataType
import numpy as np
from sentence_transformers import SentenceTransformer  # Assuming you have this installed for real-world text embeddings

load_dotenv()

def real_zilliz_similarity_search():
    uri = os.getenv("ZILLIZ_URI")
    token = os.getenv("ZILLIZ_TOKEN")
    
    if not uri or not token:
        print("Error: Missing Zilliz Cloud credentials in .env file.")
        return

    client = MilvusClient(uri=uri, token=token)
    collection_name = "real_text_collection"
    dimension = 384  # Standard dimension for all-MiniLM-L6-v2 model
    
    # Drop collection if it exists to start fresh (optional for testing)
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    
    # Create schema properly with primary key
    schema = client.create_schema(
        auto_id=True,
        enable_dynamic_field=True
    )
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dimension)
    
    # Create collection with schema
    client.create_collection(
        collection_name=collection_name,
        schema=schema
    )
    
    # Sample documents for a realistic RAG-like use case (e.g., product descriptions)
    documents = [
        "A high-performance laptop with Intel Core i7 processor, 16GB RAM, and 512GB SSD for professional use.",
        "Budget-friendly gaming mouse with RGB lighting and 10,000 DPI sensor for casual gamers.",
        "Wireless noise-cancelling headphones with 30-hour battery life and premium sound quality.",
        "Ergonomic office chair with adjustable lumbar support and breathable mesh fabric.",
        "Smart fitness tracker that monitors heart rate, steps, and sleep patterns with water resistance."
    ]
    
    # Load a pre-trained sentence transformer model for generating embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Prepare data with embeddings (no 'id' since auto_id=True)
    data = []
    for doc in documents:
        embedding = model.encode(doc).tolist()
        data.append({"text": doc, "embedding": embedding})
    
    # Insert data
    res = client.insert(collection_name, data)
    print(f"Inserted {len(data)} documents. Insert result: {res}")
    
    # Create an index for faster search (HNSW for approximate nearest neighbors)
    index_params = MilvusClient.prepare_index_params()  # Use the helper
    index_params.add_index(
        field_name="embedding",
        index_params={
            "index_type": "HNSW",
            "metric_type": "L2",
            "params": {"M": 16, "efConstruction": 200}
        }
    )
    client.create_index(collection_name, index_params)
    
    # Load the collection into memory
    client.load_collection(collection_name)
    
    # Perform similarity searches with different query examples
    queries = [
        "best laptop for work",  # Should match the laptop doc closely
        "gaming accessories",   # Should match the mouse
        "headphones for travel" # Should match the headphones
    ]
    
    for query in queries:
        query_embedding = model.encode([query]).tolist()[0]
        
        results = client.search(
            collection_name=collection_name,
            data=[query_embedding],
            limit=3,  # Top 3 similar
            output_fields=["text"]  # Retrieve the original text
        )
        
        print(f"\nQuery: '{query}'")
        print("Top similar documents:")
        for hit in results[0]:
            print(f"  - Score: {hit['distance']:.4f} | Text: {hit['entity']['text']}")
    
    # Optional: Clean up
    # client.release_collection(collection_name)
    # client.drop_collection(collection_name)

if __name__ == "__main__":
    real_zilliz_similarity_search()
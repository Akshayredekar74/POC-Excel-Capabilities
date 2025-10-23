from pymilvus import MilvusClient
import os

def run_milvus_server():
    # Connect to Milvus server instead of Milvus Lite
    try:
        client = MilvusClient(
            uri="http://localhost:19530"
        )
        print("Connected to Milvus Server")
    except Exception as e:
        print(f"Connection failed: {e}")
        return
    
    # Check if collection exists, drop it if it does (for clean testing)
    if client.has_collection("demo_collection"):
        client.drop_collection("demo_collection")
        print("Dropped existing collection 'demo_collection'")
    
    # Create collection
    client.create_collection(
        collection_name="demo_collection",
        dimension=8
    )
    print("Created collection 'demo_collection'")
    
    # Insert data
    data = [{"id": 1, "vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}]
    res = client.insert("demo_collection", data)
    print(f"Inserted one vector (insert IDs: {res['insert_count']})")
    
    # Search
    results = client.search(
        collection_name="demo_collection",
        data=[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]],
        limit=1
    )
    print("Search results:")
    for hit in results[0]:
        print(f"  - ID: {hit['id']} | Distance: {hit['distance']:.4f}")
    
    # Close connection
    client.close()
    print("Connection closed")

if __name__ == "__main__":
    run_milvus_server()
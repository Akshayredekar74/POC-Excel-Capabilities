from dotenv import load_dotenv
import os
from pymilvus import MilvusClient

load_dotenv()

def test_zilliz_cloud():
    uri = os.getenv("ZILLIZ_URI")
    token = os.getenv("ZILLIZ_TOKEN")
    
    if not uri or not token:
        print("Error: Missing Zilliz Cloud credentials in .env file.")
        return

    client = MilvusClient(uri=uri, token=token)
    
    if not client.has_collection("test_collection"):
        client.create_collection(
            collection_name="test_collection",
            dimension=8
        )
    
    data = [{"id": 1, "vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}]
    client.insert("test_collection", data)
    
    results = client.search(
        collection_name="test_collection",
        data=[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]],
        limit=5
    )
    
    print("Zilliz Cloud Results:", results)

if __name__ == "__main__":
    test_zilliz_cloud()

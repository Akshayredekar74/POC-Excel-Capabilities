# from pymilvus import MilvusClient


# client = MilvusClient(
#     uri="milvus://localhost:19530",
#     token="root:Milvus"
# )


# client.create_database(
#     db_name="my_database_1",
#     properties={
#         "database.replica.number":3
#     }
# )


# client.list_databases()


# from pymilvus import MilvusClient

# # Use local embedded Milvus Lite
# client = MilvusClient(
#     uri="./milvus_lite.db"  # this creates a local file database
# )

# client.create_database(db_name="my_database_1")
# print("Databases:", client.list_databases())



from pymilvus import MilvusClient

client = MilvusClient(uri="./milvus_lite.db")

print("Connected to Milvus Lite")

if not client.has_collection("demo_collection"):
    client.create_collection(collection_name="demo_collection", dimension=8)
    print("Created collection 'demo_collection'")

data = [{"id": 1, "vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}]
client.insert("demo_collection", data)

print("Inserted one vector")

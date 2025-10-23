from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")

# results = client.query(
#     collection_name="documents",
#     filter="",  # no filter → return any data
#     output_fields=["id", "text", "metadata"],  # exclude 'embedding' if too large
#     limit=5
# )

# for r in results:
#     print(r)



results = client.query(
    collection_name="documents",
    filter="",
    output_fields=["id", "text", "embedding"],
    limit=2
)

for r in results:
    print(r["id"], r["text"])
    print("Embedding length:", len(r["embedding"]))
    print("Sample embedding slice:", r["embedding"][:10])  # just first 10 dims

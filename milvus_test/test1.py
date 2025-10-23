
# Create DB
from pymilvus import connections, db
 
# 1. Connect to Milvus
connections.connect(alias="default", host="127.0.0.1", port="19530")
 
# 2. Create a new database (namespace)
db_name = "local_test_database"
if db_name not in db.list_database():
    db.create_database(db_name)
    print(f"Database '{db_name}' created.")
else:
    print(f"Database '{db_name}' already exists.")
 
# 3. List all databases
print("Databases:", db.list_database())
 
# 4. Disconnect
connections.disconnect("default")
print("Disconnected from Milvus")
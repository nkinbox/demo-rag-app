import weaviate

# Connect to local Weaviate instance
client = weaviate.connect_to_local()

# Check connection
if not client.is_ready():
    print("Weaviate is not ready.")
    exit()

# List all collections and object counts
collections = client.collections.list_all()
for name in collections:
    collection = client.collections.get(name)
    count = collection.aggregate.over_all()
    print(f"Collection: {name}, Object Count: {count}")

client.close()
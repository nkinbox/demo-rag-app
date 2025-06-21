import weaviate
from weaviate.classes.aggregate import GroupByAggregate
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
    print(f"Collection: {name}, Object Count: {count.total_count}")
    groupBy = collection.aggregate.over_all(group_by=GroupByAggregate(prop="file_id"))
    for group in groupBy.groups:
        print(f"\tfile: {group.grouped_by.value} Count: {group.total_count}")

client.close()
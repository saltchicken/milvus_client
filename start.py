from pymilvus import MilvusClient, Collection, connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from pymilvus import model
from sentence_transformers import SentenceTransformer
import numpy as np

connections.connect(host="10.0.0.7", port="19530")
def test():


    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=100),
    ]

    scheme = CollectionSchema(fields=fields, description="test collection")
    collection = Collection(name="test_collection", schema=scheme)
    print("Collection created")

def test2():
    collection = Collection(name="test_collection")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    text = "Hello world"

    text_embedding = model.encode(text)
    collection.insert([{"vector": text_embedding, "text": text, "subject": "history"}])

def test4():
    collection = Collection(name="test_collection")
    index_params = {
        "metric_type": "L2",
        "index_type": "GPU_CAGRA",
        "params": {"nlist": 1024}
    }
    collection.create_index(field_name="vector", index_params=index_params)
    

def test3():
    collection = Collection(name="test_collection")
    collection.load()
    print(utility.has_collection("test_collection"))
    index = collection.index()
    print(f"Index: {index}")
    result = collection.query(expr="id >= 0", output_fields=['id', 'text'])
    print(result)

# client = MilvusClient(uri="http://10.0.0.7:19530", token="root:Milvus")

# def create_or_replace_collection(collection_name):

    # if client.has_collection(collection_name=collection_name):
    #     client.drop_collection(collection_name=collection_name)
    #     client.create_collection(
    #         collection_name=collection_name,
    #         dimension=768,  # The vectors we will use in this demo has 768 dimensions
    #     )
    #     print(f"Collection {collection_name} replaced and created.")
    # else:
    #     client.create_collection(
    #         collection_name=collection_name,
    #         dimension=768,  # The vectors we will use in this demo has 768 dimensions
    #     )
    #     print(f"Collection {collection_name} created.")


# def build_collection(collection_name):
#     embedding_fn = model.DefaultEmbeddingFunction()
#
#     docs = [
#         "Some guy named John",
#         "John was 35 years old",
#         "John was a software engineer"
#     ]
#
#
#     vectors = embedding_fn.encode_documents(docs)
#     print("Dim:", embedding_fn.dim, vectors[0].shape)  # Dim: 768 (768,)
#
#     data = [
#         {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
#         for i in range(len(vectors))
#     ]
#
#     print("Data has", len(data), "entities, each with fields: ", data[0].keys())
#     print("Vector dim:", len(data[0]["vector"]))
#
#     res = client.insert(
#         collection_name=collection_name,
#         data=data,
#     )
#     print(res)

def query(query_text):
    embedding_fn = model.DefaultEmbeddingFunction()
    query_vectors = embedding_fn.encode_documents([query_text])
    print(type(query_vectors))

    res = client.search(
        collection_name="demo_collection",
        data=query_vectors,
        limit=10,
        output_fields=["text", "subject"],
    )

    print(res)

def single_vector_query(query_text):
    embedding_fn = model.DefaultEmbeddingFunction()
    query_vectors = embedding_fn.encode_documents([query_text])
    res = client.search(
        collection_name="demo_collection",
        anns_field="vector",
        data=query_vectors,
        limit=3,
        search_params={"metric_type": "COSINE"}
    )

    for hits in res:
        for hit in hits:
            print(hit)

def gpu_index(collection_name):
    connections.connect(alias='default', host='10.0.0.7', port='19530')
    index_params = {
        "metric_type": "L2",
        "index_type": "GPU_CAGRA",
        "params": {
            'intermediate_graph_degree': 64,
            'graph_degree': 32
        }
    }
    collection = Collection(collection_name)
    print(collection.indexes)
    collection.release()
    collection.drop_index()
    print("Dropping index")
    collection.create_index(field_name="vector", index_params=index_params)
    collection.load()

def gpu_search(collection_name):
    connections.connect(alias='default', host='10.0.0.7', port='19530')
    collection = Collection(collection_name)

# Load collection into memory (if not already loaded)
    collection.load()

# Define a sample query vector (must match the dimension of stored vectors)
    query_vector = [np.random.rand(768).tolist()]  # Assuming 128-dimensional vectors

# Define search parameters based on index type
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}  # Adjust based on your index type
    }

# Perform the search
    results = collection.search(
        data=query_vector,       # Query vector
        anns_field="vector",     # Indexed field name
        param=search_params,     # Search parameters
        limit=5,                 # Number of nearest neighbors to return
        output_fields=["id"]     # Fields to return in results
    )

# Print the search results
    for result in results:
        for hit in result:
            print(f"ID: {hit.id}, Distance: {hit.distance}")

if __name__ == "__main__":
    # test3()
    # create_or_replace_collection("demo_collection")
    # build_collection("demo_collection")
    query("Who is John?")
    # single_vector_query("Who is John?")
    # gpu_index("demo_collection")
    # gpu_search("demo_collection")


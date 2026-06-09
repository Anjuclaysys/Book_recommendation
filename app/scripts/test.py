from app.services.vector_store_service import (
    VectorStoreService,
)

vector_store = VectorStoreService()

result = vector_store.client.scroll(
    collection_name=vector_store.collection_name,
    limit=1,
    with_vectors=True,
)

point = result[0][0]

print("Collection:", vector_store.collection_name)

print("\nVector Names:")
print(point.vector.keys())

print("\nDense Vector Length:")
print(len(point.vector["dense"]))

print("\nSparse Vector:")
print(point.vector["sparse"])
from app.services.vector_store_service import (
    VectorStoreService
)

vector_store = VectorStoreService()

vector_store.create_collection()

print(vector_store.count())
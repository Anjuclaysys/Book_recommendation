from app.services.sparse_embedding_service import (
    SparseEmbeddingService
)

service = SparseEmbeddingService()

vector = service.get_embedding(
    "young wizard discovers magical powers"
)

print(type(vector))
print(vector.indices[:10])
print(vector.values[:10])
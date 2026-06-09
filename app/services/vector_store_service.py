from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
    SparseVector
)


from app.core.config import settings


class VectorStoreService:

    def __init__(self):

        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=300,
        )

        self.collection_name = settings.collection_name

    def create_collection(self):
        """
        Create hybrid collection
        (Dense + Sparse)
        """

        if self.client.collection_exists(
            self.collection_name
        ):
            print(
                f"Collection already exists: {self.collection_name}"
            )
            return

        self.client.create_collection(
            collection_name=self.collection_name,

            vectors_config={
                "dense": VectorParams(
                    size=1536,
                    distance=Distance.COSINE,
                )
            },

            sparse_vectors_config={
                "sparse": SparseVectorParams()
            },
        )

        print(
            f"Collection created: {self.collection_name}"
        )

    def upsert_points(
        self,
        points: list[PointStruct],
    ):
        """
        Store points in Qdrant
        """

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def dense_search(
        self,
        dense_vector,
        limit: int = 50,
    ):
        """
        Dense semantic search
        """

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vector,
            using="dense",
            limit=limit,
        )

        return results.points

    def sparse_search(
        self,
        sparse_vector,
        limit: int = 50,
    ):
        """
        Sparse BM25 search
        """

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=SparseVector(
                indices=sparse_vector.indices.tolist(),
                values=sparse_vector.values.tolist(),
            ),
            using="sparse",
            limit=limit,
        )

        return results.points

    def count(self):
        """
        Count total points
        """

        return self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )

    def delete_collection(self):
        """
        Delete collection
        """

        if self.client.collection_exists(
            self.collection_name
        ):
            self.client.delete_collection(
                self.collection_name
            )

            print(
                f"Deleted collection: {self.collection_name}"
            )

    def collection_info(self):
        """
        Collection metadata
        """

        return self.client.get_collection(
            self.collection_name
        )
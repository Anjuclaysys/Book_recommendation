import asyncio

from app.services.embedding_service import (
    EmbeddingService,
)

from app.services.sparse_embedding_service import (
    SparseEmbeddingService,
)

from app.services.vector_store_service import (
    VectorStoreService,
)

from app.schemas.recommendation import (
    BookRecommendation,
    RecommendationResponse,
)

from app.core.config import settings


class RecommendationService:

    def __init__(self):

        self.embedding_service = (
            EmbeddingService()
        )

        self.sparse_embedding_service = (
            SparseEmbeddingService()
        )

        self.vector_store = (
            VectorStoreService()
        )

    async def recommend_books(
        self,
        query: str,
        top_k: int = 5,
    ) -> RecommendationResponse:

        # Generate Dense + Sparse Embeddings concurrently
        dense_vector, sparse_vector = (
            await asyncio.gather(
                asyncio.to_thread(
                    self.embedding_service.get_embedding,
                    query,
                ),
                asyncio.to_thread(
                    self.sparse_embedding_service.get_embedding,
                    query,
                ),
            )
        )

        # Run Dense + Sparse Search concurrently
        dense_results, sparse_results = (
            await asyncio.gather(
                asyncio.to_thread(
                    self.vector_store.dense_search,
                    dense_vector,
                    50,
                ),
                asyncio.to_thread(
                    self.vector_store.sparse_search,
                    sparse_vector,
                    50,
                ),
            )
        )

        alpha = settings.alpha

        merged = {}

        for point in dense_results:

            merged[str(point.id)] = {
                "point": point,
                "dense_score": point.score,
                "sparse_score": 0.0,
            }

        for point in sparse_results:

            point_id = str(point.id)

            if point_id not in merged:

                merged[point_id] = {
                    "point": point,
                    "dense_score": 0.0,
                    "sparse_score": point.score,
                }

            else:

                merged[point_id][
                    "sparse_score"
                ] = point.score

        final_results = []

        for item in merged.values():

            dense_score = item[
                "dense_score"
            ]

            sparse_score = item[
                "sparse_score"
            ]

            final_score = (
                alpha * dense_score
                + (1 - alpha)
                * sparse_score
            )

            item["final_score"] = (
                final_score
            )

            final_results.append(
                item
            )

        final_results.sort(
            key=lambda x: x[
                "final_score"
            ],
            reverse=True,
        )

        final_results = final_results[
            :top_k
        ]

        recommendations = []

        for item in final_results:

            payload = item[
                "point"
            ].payload

            recommendations.append(
                BookRecommendation(
                    book_id=payload.get(
                        "bookID",
                        "",
                    ),
                    title=payload.get(
                        "title",
                        "",
                    ),
                    authors=payload.get(
                        "authors",
                        "",
                    ),
                    average_rating=float(
                        payload.get(
                            "average_rating",
                            0.0,
                        )
                    ),
                    ratings_count=int(
                        payload.get(
                            "ratings_count",
                            0,
                        )
                    ),
                    language_code=payload.get(
                        "language_code",
                        "",
                    ),
                    publisher=payload.get(
                        "publisher",
                        "",
                    ),
                    similarity_score=round(
                        item[
                            "final_score"
                        ],
                        4,
                    ),
                )
            )

        return RecommendationResponse(
            recommendations=
            recommendations
        )
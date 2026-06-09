import uuid

import pandas as pd

from qdrant_client.models import (
    PointStruct,
    SparseVector,
)

from app.services.embedding_service import (
    EmbeddingService,
)

from app.services.sparse_embedding_service import (
    SparseEmbeddingService,
)

from app.services.vector_store_service import (
    VectorStoreService,
)
from app.core.config import settings


BATCH_SIZE = settings.batch_size


def create_embeddings():

    print("Loading dataset...")

    df = pd.read_csv(
        "app/data/books_final.csv"
    )

    df.columns = df.columns.str.strip()

    print(
        f"Books Loaded: {len(df)}"
    )

    embedding_service = (
        EmbeddingService()
    )

    sparse_embedding_service = (
        SparseEmbeddingService()
    )

    vector_store = (
        VectorStoreService()
    )

    # Create collection if it doesn't exist
    vector_store.create_collection()

    total_books = len(df)

    for start in range(
        0,
        total_books,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            total_books,
        )

        batch_df = df.iloc[start:end]

        texts = []

        metadata_list = []

        for _, row in batch_df.iterrows():

            text = f"""
                Title: {row['title']}

                Author: {row['authors']}

                Description: {row['description']}
                """

            texts.append(text)

            metadata_list.append(
                {
                    "bookID": str(row["bookID"]),

                    "title": str(row["title"]),

                    "authors": str(row["authors"]),

                    "average_rating": (
                        float(row["average_rating"])
                        if pd.notna(row["average_rating"])
                        else 0.0
                    ),

                    "ratings_count": (
                        int(row["ratings_count"])
                        if pd.notna(row["ratings_count"])
                        else 0
                    ),

                    "language_code": str(
                        row["language_code"]
                    ),

                    "publisher": str(
                        row["publisher"]
                    ),

                    "isbn13": str(
                        row["isbn13"]
                    ),

                    "num_pages": (
                        int(row["num_pages"])
                        if pd.notna(row["num_pages"])
                        else 0
                    ),
                }
            )

        print(
            f"Generating embeddings {start}-{end}"
        )

        dense_vectors = (
            embedding_service.get_embeddings(
                texts
            )
        )

        sparse_vectors = (
            sparse_embedding_service.get_embeddings(
                texts
            )
        )

        points = []

        for (
            text,
            metadata,
            dense_vector,
            sparse_vector,
        ) in zip(
            texts,
            metadata_list,
            dense_vectors,
            sparse_vectors,
        ):

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),

                    vector={
                        "dense": dense_vector,

                        "sparse": SparseVector(
                            indices=sparse_vector.indices.tolist(),
                            values=sparse_vector.values.tolist(),
                        ),
                    },

                    payload={
                        "text": text,
                        **metadata,
                    },
                )
            )

        vector_store.upsert_points(
            points
        )

        print(
            f"Uploaded {end}/{total_books}"
        )

    print(
        "Embedding creation completed."
    )

    count = vector_store.count()

    print(
        f"Total points in collection: "
        f"{count.count}"
    )


if __name__ == "__main__":
    create_embeddings()
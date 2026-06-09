from typing import List
import time

from fastembed import SparseTextEmbedding
from app.core.config import settings


class SparseEmbeddingService:

    def __init__(self):

        self.model = SparseTextEmbedding(
            model_name=settings.sparse_embedding_model
        )

    def get_embedding(
        self,
        text: str,
    ):

        retries = 3

        for attempt in range(retries):

            try:

                return next(
                    self.model.embed([text])
                )

            except Exception as e:

                wait_time = 2 ** attempt

                print(
                    f"Sparse embedding failed "
                    f"(attempt {attempt+1}/{retries}) "
                    f"- {e}"
                )

                time.sleep(wait_time)

        raise RuntimeError(
            "Failed to generate sparse embedding"
        )

    def get_embeddings(
        self,
        texts: List[str],
    ):

        retries = 3

        for attempt in range(retries):

            try:

                return list(
                    self.model.embed(texts)
                )

            except Exception as e:

                wait_time = 2 ** attempt

                print(
                    f"Sparse embedding batch failed "
                    f"(attempt {attempt+1}/{retries}) "
                    f"- {e}"
                )

                time.sleep(wait_time)

        raise RuntimeError(
            "Failed to generate sparse embeddings"
        )
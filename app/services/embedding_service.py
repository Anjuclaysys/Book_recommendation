import time

from langchain_openai import OpenAIEmbeddings

from app.core.config import settings


class EmbeddingService:
    def __init__(self):
        self.embedding_model = OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            request_timeout=120,
            max_retries=0,  # we'll handle retries ourselves
        )

    def get_embedding(self, text: str) -> list[float]:

        retries = 5

        for attempt in range(retries):

            try:
                return self.embedding_model.embed_query(text)

            except Exception as e:

                wait_time = 2 ** attempt

                print(
                    f"Embedding failed "
                    f"(attempt {attempt + 1}/{retries}) "
                    f"- {e}"
                )

                print(
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

        raise RuntimeError(
            "Failed to generate embedding"
        )

    def get_embeddings(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        retries = 5

        for attempt in range(retries):

            try:
                return self.embedding_model.embed_documents(
                    texts
                )

            except Exception as e:

                wait_time = 2 ** attempt

                print(
                    f"Batch embedding failed "
                    f"(attempt {attempt + 1}/{retries}) "
                    f"- {e}"
                )

                print(
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

        raise RuntimeError(
            "Failed to generate embeddings"
        )
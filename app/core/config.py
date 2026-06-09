from pydantic_settings import BaseSettings

import os
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str

    # Qdrant
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None

    # Collection
    collection_name: str = "goodreads_books_new"

    # Embedding Model
    embedding_model: str = "text-embedding-3-small"
    sparse_embedding_model: str = "Qdrant/bm25"


    top_k :int= 5
    batch_size: int = 50
    alpha: float = 0.7

    class Config:
        env_file = ".env"


settings = Settings()

# if __name__ == "__main__":
#     print("Collection:", settings.collection_name)
#     print("Embedding Model:", settings.embedding_model)
#     print("Qdrant URL:", settings.qdrant_url)
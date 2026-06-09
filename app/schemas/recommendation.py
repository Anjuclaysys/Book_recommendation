from pydantic import BaseModel, Field
from typing import List


class RecommendationRequest(BaseModel):
    query: str = Field(
        ...,
        description="Natural language query for book recommendation"
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of recommendations to return"
    )


class BookRecommendation(BaseModel):
    book_id: str
    title: str
    authors: str

    average_rating: float
    ratings_count: int

    language_code: str
    publisher: str

    similarity_score: float


class RecommendationResponse(BaseModel):
    recommendations: List[BookRecommendation]
from fastapi import APIRouter

from app.schemas.recommendation import (
    RecommendationRequest,
    RecommendationResponse,
)

from app.services.recommendation_service import (
    RecommendationService,
)


router = APIRouter(
    prefix="/recommendations",
    tags=["Recommendations"],
)

recommendation_service = (
    RecommendationService()
)


@router.post(
    "",
    response_model=RecommendationResponse,
)
async def recommend_books(
    request: RecommendationRequest,
):

    return await recommendation_service.recommend_books(
        query=request.query,
        top_k=request.top_k,
    )
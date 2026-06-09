from fastapi import FastAPI

from app.api.recommendation import router as recommendation_router

app = FastAPI(
    title="Book Recommendation System",
    version="1.0.0",
    docs_url="/swagger",
    redoc_url="/redoc"
)

app.include_router(
    recommendation_router
)


@app.get("/")
def health_check():

    return {
        "status": "healthy",
        "service": "book-recommendation-system",
    }
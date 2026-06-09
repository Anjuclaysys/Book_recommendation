from app.services.recommendation_service import RecommendationService

service = RecommendationService()

response = service.recommend_books(
    query="Harry Potter and the Half-Blood Prince",
    top_k=5,
)

for book in response.recommendations:

    print(f"Title: {book.title}")
    print( f"Author: {book.authors}")
    print(f"Rating: {book.average_rating}")
    print(f"Score: {book.similarity_score}")
    print("-" * 50)
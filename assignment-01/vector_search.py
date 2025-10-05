import os
from qdrant_client import QdrantClient, models
from agents import function_tool
from telemetry import init_tracing

collection_name = "fed_speeches"
model_name = "BAAI/bge-small-en"

@function_tool()
def search_vector_database_by_query_text(query_text: str):
    """Perform semantic search on collection."""
    
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),   
    )
    
    return client.query_points(
        collection_name=collection_name,
        query=models.Document(text=query_text, model=model_name),
        limit=5).points

 
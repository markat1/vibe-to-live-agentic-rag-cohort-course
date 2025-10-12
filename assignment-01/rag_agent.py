from agents import Agent, ModelSettings, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
import os
from vector_search import search_vector_database_by_query_text
from telemetry import init_tracing
#from phoenix.otel import Tracer

#tracer, _ = init_tracing()

client = AsyncOpenAI(base_url=os.getenv("OPENAI_API_ENDPOINT"))
#client = AsyncOpenAI()
model = OpenAIChatCompletionsModel(openai_client=client, model="gpt-4.1")

#@tracer.chain
def create_rag_agent() -> Agent:
    """
    Agent is search for the best dataset to answer the user's question.
    """
    return Agent(
        name="rag_agent",
        instructions=(
            "You are a retrieval-augmented generation (RAG) agent. "
            "Always use the tool to search for relevant information to answer the user's question. "
            "If you find relevant information, use it to provide a comprehensive answer. "
            "If no relevant information is found, respond with 'I don't know'."),
        tools=[search_vector_database_by_query_text],
        model=model,
        model_settings=ModelSettings(temperature=0.0),
    )
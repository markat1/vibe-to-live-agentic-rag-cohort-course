from fastapi import FastAPI
from agents import Runner
from rag_agent import create_rag_agent
import os
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.client import Client
from phoenix.client.types.spans import SpanQuery
from phoenix.client.types import spans

app = FastAPI()

tracer_provider = register(
        project_name=os.getenv("PHOENIX_PROJECT_NAME"),
        endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT"),
)

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

@app.get("/chat")
async def ask_agent(question: str):
    result = await Runner.run(create_rag_agent(), question)
    return {"answer": result.final_output}
        

@app.get("/health")
async def health_check():
    px_client = Client()
    return spans.SpanQuery()
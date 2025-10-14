from fastapi import FastAPI
from agents import Runner
from rag_agent import create_rag_agent
import os
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor


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
        
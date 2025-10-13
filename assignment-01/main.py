from fastapi import FastAPI
from agents import Runner
from rag_agent import create_rag_agent
import os
from telemetry import init_tracing
from phoenix.trace import using_project
from phoenix.otel import register

app = FastAPI()

tracer_provider = register(
        project_name=os.getenv("PHOENIX_PROJECT_NAME"),
        endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT"),
        protocol="http/protobuf",
        auto_instrument=True,
        batch=True,
)


@app.get("/chat")
async def ask_agent(question: str):
    with using_project(os.getenv("PHOENIX_PROJECT_NAME")):
        result = await Runner.run(create_rag_agent(), question)
        return {"answer": result.final_output}
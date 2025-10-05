from fastapi import FastAPI
from agents import Runner
from rag_agent import create_rag_agent
import os
from telemetry import init_tracing

init_tracing(
        project_name=os.environ.get("PHOENIX_PROJECT_NAME"),
        auto_instrument=True,
        endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT"),
)

app = FastAPI()

@app.get("/ask")
async def ask_agent(question: str):
    result = await Runner.run(create_rag_agent(), question)
    return {"answer": result.final_output}

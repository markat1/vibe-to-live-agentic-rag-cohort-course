from fastapi import FastAPI
from agents import Runner
from rag_agent import create_rag_agent
import os
from telemetry import init_tracing
from phoenix.trace import using_project
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

# Enable automatic tracing for all agent operations
# Traces will be sent to Phoenix hosted platform
OpenAIAgentsInstrumentor().instrument()

init_tracing(
        project_name=os.environ.get("PHOENIX_PROJECT_NAME"),
        auto_instrument=True,
        endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT"),
)

app = FastAPI()

@app.get("/chat")
async def ask_agent(question: str):
    with using_project("customer-service-agent"):
        result = await Runner.run(create_rag_agent(), question)
        return {"answer": result.final_output}
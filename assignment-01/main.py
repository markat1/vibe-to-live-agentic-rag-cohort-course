from fastapi import FastAPI
from agents import Runner
from rag_agent import create_rag_agent
import os
from telemetry import init_tracing
from phoenix.trace import using_project
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

app = FastAPI()

tracer, _ = init_tracing()

@app.get("/chat")
async def ask_agent(question: str):
    result = await Runner.run(create_rag_agent(), question)
    return {"answer": result.final_output}
    #with using_project("customer-service-agent"):
       
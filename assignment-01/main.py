import uuid
from fastapi import FastAPI
from agents import Runner
from rag_agent import create_rag_agent
import os
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.client import Client
from phoenix.client.types.spans import SpanQuery
from phoenix.client.types import spans
from openinference.instrumentation import using_attributes
from openinference.instrumentation import using_prompt_template
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from helper_functions import run_rag

app = FastAPI()

tracer_provider = register(
        project_name=os.getenv("PHOENIX_PROJECT_NAME"),
        endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT"),
)

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

@app.get("/chat")
async def ask_agent(question: str):
    result = await run_rag(question)
    return {"answer": result.final_output}
        
# @app.get("/chat")
# async def ask_agent(question: str):
#     result = await Runner.run(create_rag_agent(), question)
#     return {"answer": result.final_output}

# @app.get("/chat")
# async def ask_agent(question: str):
#     request_id = str(uuid.uuid4())
#     with using_attributes(
#         tags=["route:/chat", "agent:rag"],
#         metadata={"request_id": request_id},
#         # you can also set prompt_template info here later if you want
#         # prompt_template="...", prompt_template_version="v1",
#         # prompt_template_variables={"...": "..."},
#     ):
#         result = await Runner.run(create_rag_agent(), question)
#         return {"answer": result.final_output, "request_id": request_id}
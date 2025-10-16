from openinference.instrumentation import using_attributes
from openinference.instrumentation import using_prompt_template
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from agents import Runner
from rag_agent import create_rag_agent

PROMPT_TEMPLATE = f"""{RECOMMENDED_PROMPT_PREFIX}
            You are a retrieval-augmented generation (RAG) agent. 
            Always use the tool to search for relevant information to answer the user's question. 
            Remember to analyze users question before before handing it over to the syntheizer.
            Finally, hand off the retrieved information to the SynthesizerAgent to formulate a comprehensive response.
"""
PROMPT_TEMPLATE_VARS = {"prefix": "RECOMMENDED_PROMPT_PREFIX"} 

@using_prompt_template(
template=PROMPT_TEMPLATE,
variables=PROMPT_TEMPLATE_VARS,
version="v1.0",
)
async def run_rag(question: str):
    return await Runner.run(create_rag_agent(), question)

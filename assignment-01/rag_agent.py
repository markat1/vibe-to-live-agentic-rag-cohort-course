from agents import Agent, ModelSettings, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
import os
from vector_search import search_vector_database_by_query_text
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from pydantic import BaseModel

client = AsyncOpenAI(base_url=os.getenv("OPENAI_API_ENDPOINT"))
model = OpenAIChatCompletionsModel(openai_client=client, model="gpt-4.1")

class SynthesizerOutput(BaseModel):
    synthesized_answer: str
    sources: list[str]

def create_rag_agent() -> Agent:
    """
    Agent is search for the best dataset to answer the user's question.
    """
    return Agent(
        name="rag_agent",
          instructions=f"""
            {RECOMMENDED_PROMPT_PREFIX}
            You are a retrieval-augmented generation (RAG) agent. 
            Always use the tool to search for relevant information to answer the user's question. 
            Remember to analyze users question before before handing it over to the syntheizer.
            Finally, hand off the retrieved information to the SynthesizerAgent to formulate a comprehensive response.
            "),
        """,
        tools=[search_vector_database_by_query_text],
        model=model,
        model_settings=ModelSettings(temperature=0.0, parallel_tool_calls=True),
        handoffs=[create_synthesizer_agent],
    )

def create_synthesizer_agent() -> Agent:
    """
    Agent synthesizes information from multiple sources to provide a comprehensive answer.
    """
    return Agent(
        name="synthesizer_agent",
        instructions=f"""
            {RECOMMENDED_PROMPT_PREFIX}
            You are a helpful assistant that synthesizes information from multiple sources
            to provide a comprehensive answer to the user's question.
            - synthesized_answer: A comprehensive answer to the user's question.
            - sources: A list of sources used to formulate the answer - include a list of the original queries here.
    
        """,
        model=model,
        model_settings=ModelSettings(temperature=0.0),
    )
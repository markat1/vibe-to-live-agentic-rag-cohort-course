import pandas as pd
from phoenix.evals.llm import LLM
from phoenix.evals.metrics import HallucinationEvaluator
from agents import Agent, ModelSettings, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from phoenix.evals import ClassificationEvaluator
import os

df = pd.DataFrame([   
    {
        "reference": "Both the European Central Bank (ECB) and the Federal Reserve target 2 percent inflation.",
        "query": "What inflation target do the ECB and the Fed share?",
        "response": "The European Central Bank (ECB) and the U.S. Federal Reserve (Fed) both share an explicit inflation target of 2 percent",
    },
    {
        "reference": "The revised framework documents de-emphasized operating near the effective lower bound (ELB).",
        "query": "Did the revisions increase the focus on the ELB?",
        "response": "The recent revisions to the Federal Reserve's monetary policy framework **did not increase the focus on the ELB (Effective Lower Bound)**",
    },
    {
        "reference": "The fourth International Monetary Policy Conference was hosted by the Bank of Finland in Helsinki, Finland, and Governor Olli Rehn invited the speaker.",
        "query": "In which city was the fourth International Monetary Policy Conference held, who hosted it, who invited the speaker, and which numbered iteration of the conference was it?",
        "response": "ere are the answers to your questions about the fourth International Monetary Policy Conference:\n\n- **City:** The conference was held in Helsinki, Finland.\n- **Host:** The event was hosted by the Bank of Finland.\n- **Who invited the speaker:** The invitation to the speaker was extended by Governor Olli Rehn of the Bank of Finland.\n- **Numbered iteration:** This was the fourth iteration of the International Monetary Policy Conference.\n\nI will now hand off this information for synthesis into a comprehensive response.",
    },
    {
        "reference": "Both the ECB and the Fed target 2 percent inflation, commit to revising their frameworks periodically, and the revised documents de-emphasized the ELB.",
        "query": "What shared inflation target do the ECB and the Fed have, how did the revisions treat the ELB, and what ongoing commitment do both institutions make?",
        "response": "Here is a synthesis of the key points regarding the shared inflation target, the treatment of the effective lower bound (ELB) in recent framework revisions, and the ongoing commitments made by both the European Central Bank (ECB) and the Federal Reserve (Fed):\n\n**1. Shared Inflation Target:**\n- Both the ECB and the Fed have set a 2 percent inflation target",
    },
    {
        "reference": "The ECB's statement dates to 1998; the Fed's was first issued in 2012 under Chair Ben Bernanke; the FOMC informally calls it the 'consensus statement.'",
        "query": "When was the ECB's strategy statement first issued, when was the Fed's Statement first issued and under which chair, and what informal name does the FOMC use and why?",
        "response": "Here are the answers to your questions, based on authoritative Federal Reserve sources:\n\n**1. When was the ECB's strategy statement first issued?**\n- The European Central Bank's (ECB) Monetary Policy Strategy Statement dates to the central bank's founding in 1998",
    },
    {
        "reference": "The public review had three elements (Fed Listens; academic research conference including former Chair Bernanke; policymaker discussions at FOMC meetings supported by staff analysis) and aimed to improve transparency and accountability.",
        "query": "List the three elements of the public review and what the statement was designed to provide the public.",
        "response": "The three elements of the public review conducted by the Federal Open Market Committee (FOMC) were:\n\n1. **Fed Listens events** around the U.S., where members of the public could share their views with policymakers.\n2. **An academic research conference** that featured presentations from leading experts, including former Chair Bernanke.\n3",
    },
])

#print(df.head())

llm = LLM(model="gpt-4.1", provider="openai")

#client = AsyncOpenAI(base_url=os.getenv("OPENAI_API_ENDPOINT"))
#model = OpenAIChatCompletionsModel(openai_client=client, model="gpt-4.1")

hallucination = HallucinationEvaluator(llm=llm)
hallucination.bind({"input": "query", "output": "response", "context": "reference"})

#scores = hallucination.evaluate(df.iloc[1].to_dict())
#print(scores[0])


completeness_prompt = """
You are an expert at judging the completeness of a response to a query.
Given a query and response, rate the completeness of the response.
A response is complete if it fully answers all parts of the query.
A response is partially complete if it only answers part of the query.
A response is incomplete if it does not answer any part of the query or is not related to the query.

Query: {{query}}
Response: {{response}}

Is the response complete, partially complete, or incomplete?
"""


completeness = ClassificationEvaluator(
    llm=llm, # use the same LLM instance from above
    name="completeness",
    prompt_template=completeness_prompt,
    choices={"complete": 1.0, "partially complete": 0.5, "incomplete": 0.0},
)

# test on one example
scores = completeness.evaluate(df.iloc[0].to_dict())
print(scores[0])
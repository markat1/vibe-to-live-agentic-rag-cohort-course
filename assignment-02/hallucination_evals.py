import pandas as pd
from phoenix.evals.llm import LLM
from phoenix.evals.metrics import HallucinationEvaluator
from agents import Agent, ModelSettings, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
import os

df = pd.DataFrame([   
    {
        "reference": "Both the European Central Bank (ECB) and the Federal Reserve target 2 percent inflation.",
        "query": "What inflation target do the ECB and the Fed share?",
        "response": "They both target 2.5 percent inflation.",
    },
    {
        "reference": "The revised framework documents de-emphasized operating near the effective lower bound (ELB).",
        "query": "Did the revisions increase the focus on the ELB?",
        "response": "Yes, the revisions increased focus on the ELB.",
    },
    {
        "reference": "The fourth International Monetary Policy Conference was hosted by the Bank of Finland in Helsinki, Finland, and Governor Olli Rehn invited the speaker.",
        "query": "In which city was the fourth International Monetary Policy Conference held, who hosted it, who invited the speaker, and which numbered iteration of the conference was it?",
        "response": "A) Stockholm, Sweden; B) Sveriges Riksbank; C) Stefan Ingves; D) Second.",
    },
    {
        "reference": "Both the ECB and the Fed target 2 percent inflation, commit to revising their frameworks periodically, and the revised documents de-emphasized the ELB.",
        "query": "What shared inflation target do the ECB and the Fed have, how did the revisions treat the ELB, and what ongoing commitment do both institutions make?",
        "response": "A) 2.5 percent; B) Increased focus on the ELB; C) They fixed the frameworks permanently.",
    },
    {
        "reference": "The ECB's statement dates to 1998; the Fed's was first issued in 2012 under Chair Ben Bernanke; the FOMC informally calls it the 'consensus statement.'",
        "query": "When was the ECB's strategy statement first issued, when was the Fed's Statement first issued and under which chair, and what informal name does the FOMC use and why?",
        "response": "A) 2003; B) 2008 under Janet Yellen; C) 'Forward guidance memo'—to emphasize short-run tactics.",
    },
    {
        "reference": "The public review had three elements (Fed Listens; academic research conference including former Chair Bernanke; policymaker discussions at FOMC meetings supported by staff analysis) and aimed to improve transparency and accountability.",
        "query": "List the three elements of the public review and what the statement was designed to provide the public.",
        "response": "A) Online public comment portal; B) Congressional hearings; C) Town halls with state governors; D) A binding rule like the Taylor Rule.",
    },
])

#print(df.head())

llm = LLM(model="gpt-4.1", provider="openai")

#client = AsyncOpenAI(base_url=os.getenv("OPENAI_API_ENDPOINT"))
#model = OpenAIChatCompletionsModel(openai_client=client, model="gpt-4.1")

hallucination = HallucinationEvaluator(llm=llm)
hallucination.bind({"input": "query", "output": "response", "context": "reference"})

scores = hallucination.evaluate(df.iloc[1].to_dict())
print(scores[0])
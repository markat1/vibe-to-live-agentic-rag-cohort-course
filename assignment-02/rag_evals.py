from phoenix.evals import (
    RAG_RELEVANCY_PROMPT_RAILS_MAP,
    RAG_RELEVANCY_PROMPT_TEMPLATE,
    llm_classify,
    OpenAIModel
)
import pandas as pd
from phoenix.evals.llm import LLM
import matplotlib.pyplot as plt
import pandas as pd
from pycm import ConfusionMatrix
from sklearn.metrics import classification_report

pd.set_option("display.max_colwidth", None)

df_sample = pd.DataFrame([   
    {
        "reference": "Both the European Central Bank (ECB) and the Federal Reserve target 2 percent inflation.",
        "input": "What inflation target do the ECB and the Fed share?",
        "response": "The European Central Bank (ECB) and the U.S. Federal Reserve (Fed) both share an explicit inflation target of 2 percent",
        "relevant": "relevant",    
    },
    {
        "reference": "The revised framework documents de-emphasized operating near the effective lower bound (ELB).",
        "input": "Did the revisions increase the focus on the ELB?",
        "response": "The recent revisions to the Federal Reserve's monetary policy framework **did not increase the focus on the ELB (Effective Lower Bound)**",    
        "relevant": "relevant",

    },
    {
        "reference": "The fourth International Monetary Policy Conference was hosted by the Bank of Finland in Helsinki, Finland, and Governor Olli Rehn invited the speaker.",
        "input": "In which city was the fourth International Monetary Policy Conference held, who hosted it, who invited the speaker, and which numbered iteration of the conference was it?",
        "response": "ere are the answers to your questions about the fourth International Monetary Policy Conference:\n\n- **City:** The conference was held in Helsinki, Finland.\n- **Host:** The event was hosted by the Bank of Finland.\n- **Who invited the speaker:** The invitation to the speaker was extended by Governor Olli Rehn of the Bank of Finland.\n- **Numbered iteration:** This was the fourth iteration of the International Monetary Policy Conference.\n\nI will now hand off this information for synthesis into a comprehensive response.",
        "relevant": "relevant",
    
    },
    {
        "reference": "Both the ECB and the Fed target 2 percent inflation, commit to revising their frameworks periodically, and the revised documents de-emphasized the ELB.",
        "input": "What shared inflation target do the ECB and the Fed have, how did the revisions treat the ELB, and what ongoing commitment do both institutions make?",
        "response": "Here is a synthesis of the key points regarding the shared inflation target, the treatment of the effective lower bound (ELB) in recent framework revisions, and the ongoing commitments made by both the European Central Bank (ECB) and the Federal Reserve (Fed):\n\n**1. Shared Inflation Target:**\n- Both the ECB and the Fed have set a 2 percent inflation target",
        "relevant": "relevant",    
    },
    {
        "reference": "The ECB's statement dates to 1998; the Fed's was first issued in 2012 under Chair Ben Bernanke; the FOMC informally calls it the 'consensus statement.'",
        "input": "When was the ECB's strategy statement first issued, when was the Fed's Statement first issued and under which chair, and what informal name does the FOMC use and why?",
        "response": "Here are the answers to your questions, based on authoritative Federal Reserve sources:\n\n**1. When was the ECB's strategy statement first issued?**\n- The European Central Bank's (ECB) Monetary Policy Strategy Statement dates to the central bank's founding in 1998",
        "relevant": "relevant",

    },
    {
        "reference": "The public review had three elements (Fed Listens; academic research conference including former Chair Bernanke; policymaker discussions at FOMC meetings supported by staff analysis) and aimed to improve transparency and accountability.",
        "input": "List the three elements of the public review and what the statement was designed to provide the public.",
        "response": "The three elements of the public review conducted by the Federal Open Market Committee (FOMC) were:\n\n1. **Fed Listens events** around the U.S., where members of the public could share their views with policymakers.\n2. **An academic research conference** that featured presentations from leading experts, including former Chair Bernanke.\n3",
        "relevant": "relevant",

    },

    {
        "reference": "Overall, the revised document continues to explain how the Fed interprets its mandate, remains forwardlooking with respect to policy lags, maintains a 2 percent inflation objective, and commits to conducting a public review roughly every five years.",
        "input": "Summarize four continuities retained in the Fed's revised framework: mandate interpretation, stance toward lags, inflation objective, and review cadence.",
        "response": "Here is a summary of the four key continuities retained in the Federal Reserve's revised framework, focusing on mandate interpretation, stance toward lags, inflation objective, and review cadence:\n\n1. **Mandate Interpretation**: The revised framework continues to explain how the Fed interprets its dual mandate from Congress—maximum employment and price stability. The framework still provides a structure for policy decisions, emphasizing that the maximum level of employment is not directly measurable and can change over time for reasons unrelated to policy. The Fed maintains a flexible approach to interpreting and balancing its dual objectives, especially when they may be in tension.\n\n2. **Stance Toward Lags**: The Fed retains its view that monetary policy must be forward-looking and take into account the lags with which policy actions affect the economy. This means the Committee continues to base decisions not just on current data, but also on forecasts and the expected future path of the economy, recognizing that policy effects are not immediate.\n\n3. **Inflation Objective**: The commitment to a 2 percent inflation objective is unchanged. The revised framework reaffirms that this target is central to anchoring longer-term inflation expectations, which is critical for effective monetary policy. The Fed continues to use the 2 percent goal as a key benchmark for price stability.\n\n4. **Review Cadence**: The framework maintains the practice of conducting a public review of the monetary policy strategy roughly every five years. This regular review process is intended to ensure the framework remains robust and responsive to changes in the economy and in economic understanding.\n\nThese continuities reflect the Fed’s ongoing commitment to transparency, accountability, and a systematic approach to monetary policy, even as it adapts to new economic challenges.\n\n[Sources: Jefferson, \"Monetary Policy Frameworks and the U.S. Economic Outlook\" (Sep 30, 2025); FOMC Statement on Longer-Run Goals and Monetary Policy Strategy]",
        "relevant": "relevant",   
    },
    {
        "reference": "The revised framework clarifies maximum employment as 'the highest level of employment that can be achieved on a sustained basis in a context of price stability.' It also states that 'employment may at times run above real-time assessments of maximum employment without necessarily creating risks to price stability.'",
        "input": "Provide the revised definition of maximum employment and explain what it says about employment running above realtime assessments.",
        "response": "The revised definition of maximum employment, as articulated in the Federal Reserve's 2025 Statement on Longer-Run Goals and Monetary Policy Strategy, states:\n\n- \"Maximum employment is the highest level of employment that can be achieved on a sustained basis in a context of price stability.\"\n\nA key change in the revised statement is the removal of the term \"shortfalls\" from maximum employment. Previously, the Fed said it would only mitigate \"shortfalls\" from maximum employment, reflecting the idea that real-time estimates of the natural rate of unemployment (and thus maximum employment) are highly uncertain. The new language clarifies that:\n\n- \"The Committee recognizes that employment may at times run above real-time assessments of maximum employment without necessarily creating risks to price stability.\"\n\nThis means that if employment is higher than what current estimates suggest is sustainable, the Fed will not automatically tighten policy unless there are clear risks to price stability (i.e., rising inflation). The revised definition acknowledges the uncertainty in measuring maximum employment and allows for periods where employment exceeds real-time estimates, as long as inflation remains under control.\n\nIn summary, the revised definition emphasizes that maximum employment is not a fixed number and can be above real-time assessments without necessarily causing inflation. The Fed will only act preemptively if labor market tightness or other factors threaten price stability.\n\n(Source: Powell, \"Monetary Policy and the Fed's Framework Review,\" Jackson Hole, August 2025. See also the revised Statement on Longer-Run Goals and Monetary Policy Strategy.)",
        "relevant": "relevant",
    },
    {
        "reference": "The Bank of Finland hosted the International Monetary Policy Conference in Helsinki, Finland; Governor Olli Rehn extended the invitation to the speaker. The FOMC informally calls its framework document the 'consensus statement.' The revised documents deemphasized operating near the ELB.",
        "input": "Combine institutional and policy details: name the host and city of the conference and the inviter, then state the FOMC's informal name for its framework document and whether ELB emphasis increased or decreased in the revisions.",
        "response": "Here are the combined institutional and policy details you requested:\n\n- **Host and City of the Conference, and the Inviter:**  \n  The academic research conference for the FOMC framework review was hosted by the Federal Reserve Board in Washington, D.C. The event was organized at the initiative of Vice Chair for Supervision Michelle W. Bowman, with thanks also extended to Federal Reserve staff for their work in putting the conference together.\n\n- **FOMC's Informal Name for Its Framework Document:**  \n  The FOMC informally refers to its framework document as the \"consensus statement.\" The full formal title is \"Statement on Longer-Run Goals and Monetary Policy Strategy.\"\n\n- **ELB (Effective Lower Bound) Emphasis in Revisions:**  \n  In the most recent revisions, the FOMC decreased the emphasis on the ELB. The revised consensus statement removed language that had previously highlighted the ELB as a defining feature of the economic landscape, instead broadening the framework to address a wider range of economic conditions.\n\nThese details reflect both the institutional context of the framework review and the key policy changes regarding the FOMC's approach.",
         "relevant": "relevant",
    },
    {
        "reference": "Chair Powell announced the changes in his Jackson Hole speech in August. The ECB's strategy statement dates to 1998; the Fed's was first issued in 2012 under Chair Ben Bernanke. Both target 2 percent inflation.",
        "input": "State where and when the Fed's changes were announced, then report the issuance years of the ECB and Fed statements along with their shared inflation target.",
        "response": "Here are the key details based on your request:\n\n**Where and when were the Fed's changes announced?**\n- The Federal Reserve announced its changes to its monetary policy framework in a speech by Chair Jerome Powell at the annual economic symposium sponsored by the Federal Reserve Bank of Kansas City, held in Jackson Hole, Wyoming, on August 22, 2025.\n\n**Issuance years of the ECB and Fed statements:**\n- The European Central Bank (ECB) first issued its Monetary Policy Strategy Statement in 1998 (at its founding).\n- The Federal Reserve (Fed) first issued its Statement on Longer-Run Goals and Monetary Policy Strategy in 2012.\n- Both the ECB and the Fed published revisions to their respective framework documents in 2025.\n\n**Shared inflation target:**\n- Both the ECB and the Fed have a shared inflation target of 2 percent.\n\nI will now hand off this information for synthesis.",
        "relevant": "relevant",
    }
])


print(RAG_RELEVANCY_PROMPT_TEMPLATE)

model = OpenAIModel(
    model_name="gpt-4.1",
    temperature=0.0,
)

# Will ensure the binary value expected from the template is returned
rails = list(RAG_RELEVANCY_PROMPT_RAILS_MAP.values())
relevance_classifications = llm_classify(
    data=df_sample,
    template=RAG_RELEVANCY_PROMPT_TEMPLATE,
    model=model,
    rails=rails,
    concurrency=20,
)["label"].tolist()

true_labels = df_sample["relevant"].map(RAG_RELEVANCY_PROMPT_RAILS_MAP)
if true_labels.isnull().any():
    print("Unmapped values in 'relevant':", df_sample["relevant"][true_labels.isnull()].unique())
    raise ValueError("Some values in 'relevant' are not in RAG_RELEVANCY_PROMPT_RAILS_MAP")
true_labels = true_labels.tolist()

print(classification_report(true_labels, relevance_classifications, labels=rails))
confusion_matrix = ConfusionMatrix(
    actual_vector=true_labels, predict_vector=relevance_classifications, classes=rails
)
confusion_matrix.plot(
    cmap=plt.colormaps["Blues"],
    number_label=True,
    normalized=True,
)
# confusion_matrix = ConfusionMatrix(
#     actual_vector=true_labels, predict_vector=relevance_classifications, classes=rails
# )
# confusion_matrix.plot(
#     cmap=plt.colormaps["Blues"],
#     number_label=True,
#     normalized=True,
# )
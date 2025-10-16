from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace
from tqdm import tqdm
import os
from phoenix.otel import register
from phoenix.client import AsyncClient
import asyncio

tracer_provider = register(protocol="http/protobuf", project_identifier=os.getenv("PHOENIX_PROJECT_NAME"))
tracer = tracer_provider.get_tracer(__name__)

px_client = AsyncClient()

async def fetch_primary_df():
    return await px_client.spans.get_spans_dataframe(
        project_identifier=os.getenv("PHOENIX_PROJECT_NAME")
    )

primary_df = asyncio.run(fetch_primary_df())

print(primary_df.head())

# tracer_provider = register(
#         project_name=os.getenv("PHOENIX_PROJECT_NAME"),
#         endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT"),
# )

# tracer = trace.get_tracer(__name__)

# queries = [
#     "How can I query for a monitor's status using GraphQL?",
#     "How do I delete a model?",
#     "How much does an enterprise license of Arize cost?",
#     "How do I log a prediction using the python SDK?",
# ]

# for query in tqdm(queries):
#     with tracer.start_as_current_span("Query") as span:
#         span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "chain")
#         span.set_attribute(SpanAttributes.INPUT_VALUE, query)
#         response = query_engine.query(query)
#         span.set_attribute(SpanAttributes.OUTPUT_VALUE, response)
#         print(f"Query: {query}")
#         print(f"Response: {response}")

# tracer_provider = register(
#         project_name=os.getenv("PHOENIX_PROJECT_NAME"),
#         endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT"),
# )

# tracer = trace.get_tracer(__name__)

# queries = [
#     "How can I query for a monitor's status using GraphQL?",
#     "How do I delete a model?",
#     "How much does an enterprise license of Arize cost?",
#     "How do I log a prediction using the python SDK?",
# ]

# for query in tqdm(queries):
#     with tracer.start_as_current_span("Query") as span:
#         span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "chain")
#         span.set_attribute(SpanAttributes.INPUT_VALUE, query)
#         response = query_engine.query(query)
#         span.set_attribute(SpanAttributes.OUTPUT_VALUE, response)
#         print(f"Query: {query}")
#         print(f"Response: {response}")
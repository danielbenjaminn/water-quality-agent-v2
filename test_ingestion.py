import os
from pprint import pprint

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from water_quality_agent.ingestion.service import ingest_dataset


load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

result = ingest_dataset("data/dados.csv", llm=llm,)

print("\n=== RELATÓRIO ===")
print(result["report_text"])

print("\n=== MAPEAMENTO SEMÂNTICO ===")
pprint(result["mapping"])

print("\n=== CAPACIDADES LIBERADAS ===")
pprint(result["capabilities"])

print("\n=== METADADOS ===")
pprint(result["metadata"])
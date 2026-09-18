from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from water_quality_agent.ingestion.service import ingest_dataset
from water_quality_agent.tools.regulatory_tools import (
    regulatory_compliance,
)

load_dotenv(".env")


CSV_PATH = "data/dados_igam.csv"


# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------

llm = ChatOpenAI(
    model="gpt-5.6-luna",
    temperature=0,
    reasoning_effort="none",
)


# ---------------------------------------------------------
# INGESTÃO
# ---------------------------------------------------------

print("=" * 70)
print("INGESTÃO")
print("=" * 70)

ingestion = ingest_dataset(
    CSV_PATH,
    llm=llm,
)

assert ingestion["valid"] is True

print("Dataset válido.")


# ---------------------------------------------------------
# CONFORMIDADE REGULATÓRIA
# ---------------------------------------------------------

print()
print("=" * 70)
print("CONFORMIDADE REGULATÓRIA")
print("=" * 70)

result = regulatory_compliance.invoke(
    {
        "parameter": "OXIGENIO DISSOLVIDO",
        "point": "PV180",
    }
)

print(result)
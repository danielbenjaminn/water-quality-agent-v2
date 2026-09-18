import os
from pprint import pprint

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from water_quality_agent.ingestion.service import ingest_dataset
from water_quality_agent.core.session import SESSION


load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

result = ingest_dataset("data/dados_.csv", llm=llm,)

print("\n=== RELATÓRIO ===")
print(result["report_text"])

print("\n=== MAPEAMENTO SEMÂNTICO ===")
pprint(result["mapping"])

print("\n=== CAPACIDADES LIBERADAS ===")
pprint(result["capabilities"])

print("\n=== METADADOS ===")
pprint(result["metadata"])

print("\n=== DATASET CANÔNICO ===")

if SESSION.canonical is not None:
    print(SESSION.canonical.head(10))

    print("\nColunas:")
    print(SESSION.canonical.columns.tolist())

    print("\nTipos:")
    print(SESSION.canonical.dtypes)

    print("\nShape:")
    print(SESSION.canonical.shape)
else:
    print("Nenhum dataset canônico ativo.")
    print("Ingestão válida:", result["valid"])
    print("Campos ausentes:", result["report"]["required_missing"])
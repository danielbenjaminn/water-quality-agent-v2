from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from water_quality_agent.ingestion.service import ingest_dataset
from water_quality_agent.core.parameter_resolution import ParameterResolution
from water_quality_agent.core.hitl import review_pending_parameters


# ============================================================
# LLM
# ============================================================

load_dotenv(".env")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)


# ============================================================
# 1. INGESTÃO
# ============================================================

print("\n" + "=" * 70)
print("1. INGESTÃO")
print("=" * 70)

result = ingest_dataset(
    "data/dados_.csv",
    llm=llm,
)

print("Base válida:", result["valid"])


# ============================================================
# 2. RECUPERA PENDÊNCIAS
# ============================================================

print("\n" + "=" * 70)
print("2. PENDÊNCIAS")
print("=" * 70)

pending_data = result["metadata"].get(
    "pending_parameters",
    [],
)

pending = [
    ParameterResolution.model_validate(item)
    for item in pending_data
]

print(f"Total: {len(pending)}")

for item in pending:
    print(
        "-",
        item.original,
        "->",
        (
            item.llm_suggestion.suggested_canonical
            if item.llm_suggestion
            else None
        ),
    )


# ============================================================
# 3. HITL
# ============================================================

print("\n" + "=" * 70)
print("3. REVISÃO HUMANA")
print("=" * 70)

approved = review_pending_parameters(
    pending
)


# ============================================================
# 4. RESULTADO
# ============================================================

print("\n" + "=" * 70)
print("4. DECISÕES REGISTRADAS")
print("=" * 70)

if not approved:
    print("Nenhuma decisão registrada.")

else:
    for item in approved:
        print(
            f"{item.original}"
            f" -> {item.canonical}"
            f" | unidade={item.unit}"
            f" | source={item.source}"
        )
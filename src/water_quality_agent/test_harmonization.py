from __future__ import annotations

import os

import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from water_quality_agent.core.session import SESSION
from water_quality_agent.ingestion.service import ingest_dataset


load_dotenv(".env")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)


print("\n" + "=" * 70)
print("1. INGESTÃO + HARMONIZAÇÃO")
print("=" * 70)

result = ingest_dataset(
    "data/dados_.csv",
    llm=llm,
)

print("Base válida:", result["valid"])
print(
    "Pendências:",
    len(
        result["metadata"].get(
            "pending_parameters",
            [],
        )
    ),
)


df = SESSION.require_data()


print("\n" + "=" * 70)
print("2. SCHEMA ANALÍTICO")
print("=" * 70)

print(df.columns.tolist())


print("\n" + "=" * 70)
print("3. AMOSTRA")
print("=" * 70)

cols = [
    "parameter_original",
    "parameter",
    "result_original",
    "qualifier",
    "result",
    "unit_original",
    "unit",
    "unit_source",
    "conversion_factor",
    "conversion_status",
]

print(
    df[cols]
    .head(30)
    .to_string(index=False)
)


print("\n" + "=" * 70)
print("4. TIPOS")
print("=" * 70)

print("result dtype:", df["result"].dtype)


print("\n" + "=" * 70)
print("5. STATUS DAS CONVERSÕES")
print("=" * 70)

print(
    df["conversion_status"]
    .value_counts(dropna=False)
)


print("\n" + "=" * 70)
print("6. CONVERSÕES REALIZADAS")
print("=" * 70)

converted = df[
    df["conversion_status"] == "converted"
]

if converted.empty:
    print("Nenhuma conversão.")

else:
    print(
        converted[
            [
                "parameter",
                "result_original",
                "unit_original",
                "qualifier",
                "result",
                "unit",
                "conversion_factor",
            ]
        ]
        .to_string(index=False)
    )


print("\n" + "=" * 70)
print("7. CONVERSÕES NÃO RESOLVIDAS")
print("=" * 70)

unresolved = df[
    df["conversion_status"] == "unresolved"
]

if unresolved.empty:
    print("Nenhuma conversão não resolvida.")

else:
    print(
        unresolved[
            [
                "parameter",
                "result_original",
                "unit_original",
                "result",
                "unit",
                "unit_source",
            ]
        ]
        .drop_duplicates()
        .to_string(index=False)
    )


print("\n" + "=" * 70)
print("8. RESULTADOS CENSURADOS")
print("=" * 70)

censored = df[
    df["qualifier"].isin(["<", ">"])
]

print(
    censored[
        [
            "parameter",
            "result_original",
            "qualifier",
            "result",
            "unit",
        ]
    ]
    .head(30)
    .to_string(index=False)
)


print("\n" + "=" * 70)
print("9. TESTES BÁSICOS")
print("=" * 70)

assert pd.api.types.is_numeric_dtype(
    df["result"]
)

assert "qualifier" in df.columns

assert "result_original" in df.columns
assert "unit_original" in df.columns
assert "parameter_original" in df.columns

assert "target_unit" not in df.columns

assert "conversion_status" in df.columns

print("OK - result é numérico")
print("OK - qualifier existe")
print("OK - originais preservados")
print("OK - target_unit removida")
print("OK - conversion_status existe")


print("\n" + "=" * 70)
print("FIM")
print("=" * 70)
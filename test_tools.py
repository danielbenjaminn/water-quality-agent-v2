from __future__ import annotations

import os
from pprint import pprint

import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from water_quality_agent.core.session import SESSION
from water_quality_agent.ingestion.service import ingest_dataset

from water_quality_agent.tools.data_tools import (
    dataset_capabilities,
    list_parameters,
    list_points,
    get_series,
    resolve_water_parameter,
)

from water_quality_agent.tools.analysis_tools import (
    descriptive_statistics,
)

from water_quality_agent.tools.regulatory_tools import (
    get_conama_class2_limits,
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

CSV_PATH = (
    "/home/danielbenjamin/Documentos/Scripts Python/"
    "Agentic water quality v2/"
    "water-quality-agent-agentic-v2/"
    "data/dados_.csv"
)

PARAMETER = "Chumbo total"
POINT = "P01"


# ============================================================
# 1. INGESTÃO
# ============================================================

print("\n" + "=" * 70)
print("1. INGESTÃO")
print("=" * 70)

ingestion = ingest_dataset(
    CSV_PATH,
    llm=llm,
)

print(f"Base válida: {ingestion['valid']}")

if not ingestion["valid"]:
    print("\nA base foi bloqueada pela ingestão:")
    print(ingestion["report_text"])
    raise SystemExit(1)


# ============================================================
# 2. CAPACIDADES
# ============================================================

print("\n" + "=" * 70)
print("2. CAPACIDADES DO DATASET")
print("=" * 70)

capabilities = dataset_capabilities.invoke({})

pprint(capabilities["capabilities"])


# ============================================================
# 3. PARÂMETROS
# ============================================================

print("\n" + "=" * 70)
print("3. PARÂMETROS DISPONÍVEIS")
print("=" * 70)

parameters = list_parameters.invoke({})

print(f"Quantidade de parâmetros: {len(parameters)}")

for parameter in parameters[:20]:
    print(f"- {parameter}")

if len(parameters) > 20:
    print(
        f"... + {len(parameters) - 20} parâmetros"
    )


# ============================================================
# 4. PONTOS
# ============================================================

print("\n" + "=" * 70)
print("4. PONTOS DISPONÍVEIS")
print("=" * 70)

points = list_points.invoke({})

print(f"Quantidade de pontos: {len(points)}")

for point in points[:20]:
    print(f"- {point}")

if len(points) > 20:
    print(
        f"... + {len(points) - 20} pontos"
    )


# ============================================================
# 5. RECUPERAÇÃO DA SÉRIE
# ============================================================

print("\n" + "=" * 70)
print("5. GET_SERIES")
print("=" * 70)

print(f"Parâmetro solicitado: {PARAMETER}")
print(f"Ponto solicitado: {POINT}")

observations = get_series.invoke(
    {
        "parameter": PARAMETER,
        "point": POINT,
    }
)

print(
    f"\nObservações recuperadas: "
    f"{len(observations)}"
)

print("\nPrimeiras observações:")

for obs in observations[:10]:
    pprint(obs)


# ============================================================
# 6. VALIDAÇÃO BÁSICA DA SÉRIE
# ============================================================

print("\n" + "=" * 70)
print("6. VALIDAÇÃO DA SÉRIE")
print("=" * 70)

if not observations:
    print(
        "ERRO: nenhuma observação foi recuperada."
    )
    raise SystemExit(1)

required_fields = {
    "date",
    "parameter",
    "result",
    "unit",
}

missing_fields = (
    required_fields
    - set(observations[0])
)

if missing_fields:
    print(
        f"ERRO: campos ausentes: "
        f"{missing_fields}"
    )
    raise SystemExit(1)

print("Estrutura da série: OK")


# ============================================================
# 7. ESTATÍSTICA DESCRITIVA
# ============================================================

print("\n" + "=" * 70)
print("7. ESTATÍSTICA DESCRITIVA")
print("=" * 70)

statistics = descriptive_statistics.invoke(
    {
        "observations": observations,
    }
)

pprint(statistics)


# ============================================================
# 8. RESOLUÇÃO DO PARÂMETRO
# ============================================================

print("\n" + "=" * 70)
print("8. RESOLUÇÃO DO PARÂMETRO")
print("=" * 70)

resolved = resolve_water_parameter.invoke(
    {
        "name": PARAMETER,
    }
)

pprint(resolved)


# ============================================================
# 9. LIMITES CONAMA 357/2005 - CLASSE 2
# ============================================================

print("\n" + "=" * 70)
print("9. LIMITES CONAMA - CLASSE 2")
print("=" * 70)

limits = get_conama_class2_limits.invoke(
    {
        "parameter": PARAMETER,
    }
)

pprint(limits)


# ============================================================
# 10. INSPEÇÃO DA HARMONIZAÇÃO
# ============================================================

print("\n" + "=" * 70)
print("10. HARMONIZAÇÃO DO DATASET")
print("=" * 70)

df = SESSION.require_data()

print(f"\nShape: {df.shape}")

print("\nColunas:")
print(df.columns.tolist())

cols = [
    "parameter_original",
    "parameter",
    "result_original",
    "result",
    "unit_original",
    "unit",
    "target_unit",
    "unit_source",
]

cols = [
    col
    for col in cols
    if col in df.columns
]

print("\nAmostra harmonizada:")

print(
    df[cols]
    .drop_duplicates()
    .head(30)
    .to_string(index=False)
)


# ============================================================
# 11. PARÂMETROS ALTERADOS PELA HARMONIZAÇÃO
# ============================================================

print("\n" + "=" * 70)
print("11. PARÂMETROS HARMONIZADOS")
print("=" * 70)

if {
    "parameter_original",
    "parameter",
}.issubset(df.columns):

    changed_parameters = (
        df.loc[
            df["parameter_original"]
            .astype(str)
            .str.casefold()
            !=
            df["parameter"]
            .astype(str)
            .str.casefold(),
            [
                "parameter_original",
                "parameter",
            ],
        ]
        .drop_duplicates()
        .sort_values(
            "parameter_original"
        )
    )

    print(
        f"\nParâmetros alterados: "
        f"{len(changed_parameters)}"
    )

    if changed_parameters.empty:
        print("Nenhum.")
    else:
        print(
            changed_parameters.to_string(
                index=False
            )
        )


# ============================================================
# 12. FONTES DAS UNIDADES CANÔNICAS
# ============================================================

print("\n" + "=" * 70)
print("12. FONTES DAS UNIDADES CANÔNICAS")
print("=" * 70)

if "unit_source" in df.columns:

    print(
        df["unit_source"]
        .value_counts(dropna=False)
    )


# ============================================================
# 13. UNIDADES DIFERENTES DA UNIDADE ALVO
#
# IMPORTANTE:
# Aqui apenas INSPECIONAMOS.
# Ainda NÃO convertemos os resultados.
# ============================================================

print("\n" + "=" * 70)
print("13. UNIDADES DIFERENTES DA UNIDADE ALVO")
print("=" * 70)

if {
    "unit",
    "target_unit",
}.issubset(df.columns):

    different = df[
        df["target_unit"].notna()
        & df["unit"].notna()
        & (
            df["unit"].astype(str)
            !=
            df["target_unit"].astype(str)
        )
    ]

    different_cols = [
        col
        for col in [
            "parameter_original",
            "parameter",
            "unit_original",
            "unit",
            "target_unit",
            "unit_source",
        ]
        if col in different.columns
    ]

    different = (
        different[different_cols]
        .drop_duplicates()
    )

    print(
        f"\nTotal de combinações diferentes: "
        f"{len(different)}"
    )

    if different.empty:
        print("Nenhuma.")
    else:
        print(
            different.to_string(
                index=False
            )
        )


# ============================================================
# 14. PARÂMETROS SEM UNIDADE ALVO
# ============================================================

print("\n" + "=" * 70)
print("14. PARÂMETROS SEM UNIDADE ALVO")
print("=" * 70)

if "target_unit" in df.columns:

    unresolved_units = (
        df.loc[
            df["target_unit"].isna(),
            [
                "parameter_original",
                "parameter",
                "unit",
            ],
        ]
        .drop_duplicates()
    )

    print(
        f"\nTotal sem unidade alvo: "
        f"{len(unresolved_units)}"
    )

    if unresolved_units.empty:
        print("Nenhum.")
    else:
        print(
            unresolved_units.to_string(
                index=False
            )
        )


# ============================================================
# 15. FALLBACK LLM / PENDÊNCIAS HITL
# ============================================================

print("\n" + "=" * 70)
print("15. FALLBACK LLM / PENDÊNCIAS HITL")
print("=" * 70)

pending = (
    ingestion["metadata"]
    .get(
        "pending_parameters",
        []
    )
)

print(
    f"\nTotal de parâmetros pendentes: "
    f"{len(pending)}"
)

if not pending:
    print(
        "\nNenhum parâmetro exige revisão humana."
    )

else:

    for item in pending:

        print("\n" + "-" * 60)

        print(
            f"Original: "
            f"{item.get('original')}"
        )

        print(
            f"Status: "
            f"{item.get('status')}"
        )

        print(
            f"Fonte: "
            f"{item.get('source')}"
        )

        suggestion = (
            item.get("llm_suggestion")
        )

        if suggestion:

            print("\nSugestão da LLM:")

            print(
                "  Canonical:",
                suggestion.get(
                    "suggested_canonical"
                ),
            )

            print(
                "  Unidade:",
                suggestion.get(
                    "suggested_unit"
                ),
            )

            print(
                "  Confiança:",
                suggestion.get(
                    "confidence"
                ),
            )

            print(
                "  Motivo:",
                suggestion.get(
                    "reason"
                ),
            )

        else:

            print(
                "\nSem sugestão da LLM."
            )


# ============================================================
# 16. VERIFICAR SE PENDÊNCIAS NÃO FORAM APLICADAS
# ============================================================

print("\n" + "=" * 70)
print("16. VALIDAÇÃO DAS PENDÊNCIAS")
print("=" * 70)

if not pending:

    print(
        "Não existem pendências para validar."
    )

else:

    for item in pending:

        original = item.get("original")

        rows = df[
            df["parameter_original"]
            .astype(str)
            .eq(str(original))
        ]

        if rows.empty:

            print(
                f"\n{original}: "
                "não encontrado no DataFrame."
            )

            continue

        current_parameters = (
            rows["parameter"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        print(
            f"\nOriginal: {original}"
        )

        print(
            "Parâmetro atualmente armazenado:",
            current_parameters,
        )

        suggestion = (
            item.get("llm_suggestion")
        )

        if suggestion:

            suggested = suggestion.get(
                "suggested_canonical"
            )

            print(
                "Sugestão ainda não aprovada:",
                suggested,
            )

            if (
                suggested is not None
                and suggested
                in current_parameters
            ):

                print(
                    "ERRO: a sugestão da LLM "
                    "foi aplicada antes do HITL."
                )

            else:

                print(
                    "OK: sugestão da LLM "
                    "não foi aplicada."
                )


# ============================================================
# RESULTADO FINAL
# ============================================================

print("\n" + "=" * 70)
print("TESTE FINALIZADO")
print("=" * 70)

print(
    f"""
Fluxo testado:

CSV
 ↓
ingest_dataset(llm)
 ↓
validação
 ↓
harmonize_dataset(llm)
 ↓
resolve_dataset_parameters()
 ├─ resolução determinística
 └─ fallback LLM
       ↓
pending_parameters
       ↓
aguarda HITL

Dataset:
- linhas: {len(df)}
- parâmetros: {df["parameter"].nunique()}
- pontos: {df["point"].nunique()}
- pendências HITL: {len(pending)}

Nenhuma conversão física de resultados
foi executada neste teste.
"""
)
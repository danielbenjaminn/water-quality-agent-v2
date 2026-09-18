from __future__ import annotations

import pandas as pd

from water_quality_agent.core.session import SESSION

from pprint import pprint
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from water_quality_agent.tools.data_tools import resolve_water_parameter
from water_quality_agent.tools.regulatory_tools import get_conama_class2_limits
from water_quality_agent.tools.unit_tools import normalize_observation_units
from water_quality_agent.ingestion.service import ingest_dataset
from water_quality_agent.tools.data_tools import (
    dataset_capabilities,
    list_parameters,
    list_points,
    get_series,
)
from water_quality_agent.tools.analysis_tools import (
    descriptive_statistics,
)

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)
# ============================================================
# CONFIGURAÇÃO
# ============================================================

CSV_PATH = "/home/danielbenjamin/Documentos/Scripts Python/Agentic water quality v2/water-quality-agent-agentic-v2/data/dados_.csv"

# Use um parâmetro e ponto que existam na sua base.
# Depois podemos tornar essa escolha automática.
PARAMETER = "Chumbo total"
POINT = "P01"


# ============================================================
# 1. INGESTÃO
# ============================================================

print("\n" + "=" * 70)
print("1. INGESTÃO")
print("=" * 70)

ingestion = ingest_dataset(CSV_PATH, llm=llm)

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
    print(f"... + {len(parameters) - 20} parâmetros")


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
    print(f"... + {len(points) - 20} pontos")


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

print(f"\nObservações recuperadas: {len(observations)}")

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
    print("ERRO: nenhuma observação foi recuperada.")
    raise SystemExit(1)

required_fields = {
    "date",
    "parameter",
    "result",
    "unit",
}

missing_fields = required_fields - set(observations[0])

if missing_fields:
    print(f"ERRO: campos ausentes: {missing_fields}")
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
# RESULTADO
# ============================================================

print("\n" + "=" * 70)
print("TESTE FINALIZADO")
print("=" * 70)

print(
    f"""
Fluxo executado:

CSV
 ↓
ingest_dataset()
 ↓
SESSION.canonical
 ↓
get_series()
 ↓
descriptive_statistics()

Parâmetro: {PARAMETER}
Ponto: {POINT}
Observações: {len(observations)}
"""
)
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
# 10. NORMALIZAÇÃO DE UNIDADES
# ============================================================

print("\n" + "=" * 70)
print("10. NORMALIZAÇÃO DE UNIDADES")
print("=" * 70)

if not limits:
    print(
        "Nenhum limite regulatório encontrado. "
        "Teste de conversão de unidade não executado."
    )

else:
    # Primeiro apenas inspecionamos o retorno da legislação.
    # Ainda não assumimos o nome da chave que contém a unidade.
    print("\nEstrutura do primeiro limite retornado:")
    pprint(limits[0])

    target_unit = (
        limits[0].get("unit")
        or limits[0].get("Unidade")
        or limits[0].get("unidade")
    )

    if target_unit is None:
        print(
            "\nNão foi possível identificar automaticamente "
            "a unidade do limite regulatório."
        )

    else:
        print(f"\nUnidade regulatória: {target_unit}")

        normalized = normalize_observation_units.invoke(
            {
                "observations": observations,
                "target_unit": target_unit,
            }
        )

        print(
            f"Observações normalizadas: {len(normalized)}"
        )

        print("\nPrimeiras observações normalizadas:")

        for obs in normalized[:10]:
            pprint(obs)


print("\n" + "=" * 70)
print("SEGUNDO TESTE FINALIZADO")
print("=" * 70)

from water_quality_agent.core.conversao_unidades import padronizar_unidades


print("\n" + "=" * 70)
print("TESTE - PADRONIZAÇÃO GLOBAL DAS UNIDADES")
print("=" * 70)

df = SESSION.require_data()

unidades_originais = (
    df["unit"]
    .dropna()
    .astype(str)
    .drop_duplicates()
    .sort_values()
)

resultado = pd.DataFrame({
    "Original": unidades_originais,
})

resultado["Padronizada"] = padronizar_unidades(
    resultado["Original"]
)

print(resultado.to_string(index=False))

print("\nTotal de unidades originais:", resultado["Original"].nunique())
print(
    "Total após padronização:",
    resultado["Padronizada"].nunique()
)
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from water_quality_agent.ingestion.service import ingest_dataset
from water_quality_agent.tools.analysis_tools import kendall_correlations
from water_quality_agent.tools.data_tools import list_parameters, list_points


load_dotenv(".env")


# ============================================================
# LLM
# ============================================================

llm = ChatOpenAI(
    model="gpt-5.6-luna",
    temperature=0,
    reasoning_effort="none",
)


# ============================================================
# INGESTÃO
# ============================================================

print("\n" + "=" * 70)
print("INGESTÃO")
print("=" * 70)

ingestion = ingest_dataset(
    "data/dados_igam.csv",
    llm=llm,
)

print("Dataset válido:", ingestion["valid"])

if not ingestion["valid"]:
    print(ingestion)
    raise RuntimeError("Falha na ingestão.")


# ============================================================
# INSPEÇÃO
# ============================================================

print("\n" + "=" * 70)
print("PONTOS DISPONÍVEIS")
print("=" * 70)

points = list_points.invoke({})

print(points)


print("\n" + "=" * 70)
print("PARÂMETROS DISPONÍVEIS")
print("=" * 70)

parameters = list_parameters.invoke({})

print(parameters)


# ============================================================
# TESTE DE CORRELAÇÃO
# ============================================================

print("\n" + "=" * 70)
print("CORRELAÇÃO DE KENDALL")
print("=" * 70)

result = kendall_correlations.invoke(
    {
        "parameters": [
            "OXIGENIO DISSOLVIDO",
            "TURBIDEZ",
        ],
        "point": "PV180",
    }
)

print(result)


# ============================================================
# VALIDAÇÕES MÍNIMAS
# ============================================================

print("\n" + "=" * 70)
print("VALIDAÇÃO")
print("=" * 70)

assert isinstance(result, list)

if result:
    first = result[0]

    assert "point" in first
    assert "parameter_x" in first
    assert "parameter_y" in first
    assert "n_pairs" in first
    assert "kendall_tau" in first

    kendall = first["kendall_tau"]

    assert "value" in kendall
    assert "p_value" in kendall
    assert "valid" in kendall
    assert "reason" in kendall

    print("✓ Estrutura do resultado válida")
    print("✓ n_pairs:", first["n_pairs"])
    print("✓ Kendall tau:", kendall["value"])
    print("✓ p-value:", kendall["p_value"])
    print("✓ valid:", kendall["valid"])
    print("✓ reason:", kendall["reason"])

else:
    print(
        "⚠ A tool executou corretamente, mas não encontrou "
        "observações pareadas para os parâmetros selecionados."
    )


print("\nTESTE CONCLUÍDO.")
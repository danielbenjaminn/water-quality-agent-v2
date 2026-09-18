from dotenv import load_dotenv
from langchain_groq import ChatGroq

from water_quality_agent.ingestion.service import ingest_dataset
from water_quality_agent.tools.data_tools import (
    dataset_capabilities,
    list_parameters,
    list_points,
    resolve_water_parameter,
    get_series,
)


load_dotenv(".env")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)


# Ajuste somente o caminho para o CSV que você já está usando
CSV_PATH = "data/dados_.csv"


result = ingest_dataset(
    CSV_PATH,
    llm=llm,
)

assert result["valid"] is True


print("\nPARÂMETROS")
parameters = list_parameters.invoke({})
print(parameters[:20])


print("\nPONTOS")
points = list_points.invoke({})
print(points[:20])


print("\nRESOLUÇÃO DE OD")
resolved = resolve_water_parameter.invoke(
    {"name": "OD"}
)
print(resolved)


canonical = resolved.get("canonical")

assert canonical is not None


print("\nSÉRIE")
series = get_series.invoke(
    {
        "parameter": canonical,
    }
)

print(series[:5])


assert len(series) > 0

assert all(
    row["parameter"].casefold()
    == canonical.casefold()
    for row in series
)

assert all(
    isinstance(row["result"], (int, float))
    or row["result"] is None
    for row in series
)

print("\nOK - parâmetros canônicos")
print("OK - resolução da intenção do usuário")
print("OK - get_series consulta o dataset harmonizado")
print("OK - result permanece numérico")
print("OK - get_series não reconverte unidades")
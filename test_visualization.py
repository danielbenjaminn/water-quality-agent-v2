from dotenv import load_dotenv
from langchain_groq import ChatGroq

from water_quality_agent.ingestion.service import ingest_dataset
from water_quality_agent.tools.data_tools import (
    resolve_water_parameter,
    get_series,
)
from water_quality_agent.tools.regulatory_tools import (
    get_conama_class2_limits,
)
from water_quality_agent.tools.visualization_tools import (
    plot_time_series,
)


load_dotenv(".env")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)


CSV_PATH = "data/dados_igam.csv"


# ---------------------------------------------------------
# Ingestão
# ---------------------------------------------------------

result = ingest_dataset(
    CSV_PATH,
    llm=llm,
)

assert result["valid"] is True


# ---------------------------------------------------------
# Resolve OD
# ---------------------------------------------------------

resolved = resolve_water_parameter.invoke(
    {
        "name": "Oxigênio dissolvido",
    }
)

canonical = resolved["canonical"]

print("\nPARÂMETRO")
print(canonical)


# ---------------------------------------------------------
# Série
# ---------------------------------------------------------

series = get_series.invoke(
    {
        "parameter": canonical,
        "point": "PV180",
    }
)

print("\nSÉRIE")
print(series)

assert len(series) > 0


# ---------------------------------------------------------
# CONAMA
# ---------------------------------------------------------

limits = get_conama_class2_limits.invoke(
    {
        "parameter": canonical,
    }
)

print("\nLIMITES")
print(limits)

assert len(limits) > 0


minimum = limits[0].get("Min")
maximum = limits[0].get("Max")


# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

path = plot_time_series.invoke(
    {
        "observations": series,
        "title": "Oxigênio dissolvido - P01",
        "minimum": minimum,
        "maximum": maximum,
    }
)

print("\nGRÁFICO")
print(path)

print("\nOK - série harmonizada")
print("OK - limite regulatório recuperado")
print("OK - gráfico gerado")
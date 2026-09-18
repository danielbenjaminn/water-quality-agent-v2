from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from water_quality_agent.ingestion.service import ingest_dataset
from water_quality_agent.agent.graph import build_agent


load_dotenv(".env")


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)


CSV_PATH = "data/dados_igam.csv"


# ---------------------------------------------------------
# Ingestão
# ---------------------------------------------------------

ingestion = ingest_dataset(
    CSV_PATH,
    llm=llm,
)

assert ingestion["valid"] is True


# ---------------------------------------------------------
# Agente
# ---------------------------------------------------------

agent = build_agent(llm)


result = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content=(
                    "Plote a série temporal de oxigênio dissolvido "
                    "no ponto PV180 e mostre no gráfico os limites "
                    "aplicáveis da CONAMA 357/2005 Classe 2."
                )
            )
        ]
    }
)


# ---------------------------------------------------------
# Trace
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("EXECUÇÃO DO AGENTE")
print("=" * 70)


for i, message in enumerate(
    result["messages"],
    start=1,
):
    print(
        f"\n--- {i} | {message.__class__.__name__} ---"
    )

    if getattr(message, "content", None):
        print(message.content)

    tool_calls = getattr(
        message,
        "tool_calls",
        None,
    )

    if tool_calls:
        print("\nTOOL CALLS:")

        for call in tool_calls:
            print(f"\n  {call['name']}")
            print(f"  args = {call['args']}")


print("\n" + "=" * 70)
print("RESPOSTA FINAL")
print("=" * 70)

print(result["messages"][-1].content)
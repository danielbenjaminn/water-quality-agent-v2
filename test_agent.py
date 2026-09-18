from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from water_quality_agent.ingestion.service import ingest_dataset
from water_quality_agent.agent.graph import build_agent


load_dotenv(".env")


# =========================================================
# Configuração
# =========================================================

# llm = ChatGroq(
#     model="openai/gpt-oss-20b",
#     temperature=0,
# )

llm = ChatOpenAI(
    model="gpt-5.6-luna",
    temperature=0,
    reasoning_effort="none",
)

CSV_PATH = "data/dados_igam.csv"


# =========================================================
# Ingestão
# =========================================================

print("\n" + "=" * 70)
print("INGESTÃO")
print("=" * 70)

ingestion = ingest_dataset(
    CSV_PATH,
    llm=llm,
)

assert ingestion["valid"] is True

print("Dataset válido.")


# =========================================================
# Agente
# =========================================================

agent = build_agent(llm)

prompt = (
    "Analise o comportamento do oxigênio dissolvido no ponto PV180 "
        "ao longo da série histórica. Considere a estatística descritiva, "
        "a tendência temporal, os limites aplicáveis da CONAMA 357/2005 "
        "Classe 2 e utilize referências técnicas para fundamentar a "
        "interpretação."
)

print("\n" + "=" * 70)
print("PROMPT")
print("=" * 70)
print(prompt)


result = agent.invoke(
    {
        "messages": [
            HumanMessage(content=prompt)
        ]
    }
)


# =========================================================
# Trace
# =========================================================

print("\n" + "=" * 70)
print("TRACE DO AGENTE")
print("=" * 70)


for i, message in enumerate(result["messages"], start=1):

    print(
        f"\n{'-' * 70}\n"
        f"{i} | {message.__class__.__name__}\n"
        f"{'-' * 70}"
    )

    # -----------------------------------------------------
    # Human
    # -----------------------------------------------------

    if isinstance(message, HumanMessage):
        print(message.content)
        continue

    # -----------------------------------------------------
    # AI
    # -----------------------------------------------------

    if isinstance(message, AIMessage):

        if message.content:
            print(message.content)

        if message.tool_calls:

            print("\nTOOL CALLS:")

            for call in message.tool_calls:

                print(f"\n  Tool : {call['name']}")
                print(f"  Args : {call['args']}")

        continue

    # -----------------------------------------------------
    # Tool
    # -----------------------------------------------------

    if isinstance(message, ToolMessage):

        print(f"Tool : {message.name}")
        print("Retorno:")

        content = message.content

        # Evita despejar respostas gigantes no terminal
        if isinstance(content, str) and len(content) > 3000:
            print(content[:3000])
            print("\n...[retorno truncado no trace]...")
        else:
            print(content)

        continue

    # -----------------------------------------------------
    # Outros tipos
    # -----------------------------------------------------

    if getattr(message, "content", None):
        print(message.content)


# =========================================================
# Resposta final
# =========================================================

print("\n" + "=" * 70)
print("RESPOSTA FINAL")
print("=" * 70)

print(result["messages"][-1].content)
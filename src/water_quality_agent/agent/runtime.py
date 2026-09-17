from __future__ import annotations
from water_quality_agent.ingestion import ingest_dataset
from water_quality_agent.core.session import SESSION
from water_quality_agent.agent.graph import build_agent


class WaterQualityAgent:
    def __init__(self, llm):
        self.llm = llm
        self.graph = build_agent(llm)

    def load_csv(self, path: str) -> dict:
        """Lê, interpreta semanticamente e valida o dataset antes de habilitar análises."""
        return ingest_dataset(path, llm=self.llm)

    def ask(self, question: str):
        if not SESSION.metadata.get("ingestion_valid", False) or SESSION.canonical is None:
            missing = SESSION.metadata.get("ingestion_report", {}).get("required_missing", [])
            detail = ", ".join(missing) if missing else "schema obrigatório não validado"
            raise RuntimeError(
                "As análises estão bloqueadas porque o dataset não passou pela validação de ingestão. "
                f"Campos pendentes: {detail}."
            )
        return self.graph.invoke({"messages": [("user", question)]})

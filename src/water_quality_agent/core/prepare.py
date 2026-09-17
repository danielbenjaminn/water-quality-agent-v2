"""Compatibilidade temporária com chamadas antigas de `prepare_dataset`."""
from water_quality_agent.ingestion.service import ingest_dataset


def prepare_dataset(path, llm=None):
    return ingest_dataset(path, llm=llm)

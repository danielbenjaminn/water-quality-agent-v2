from __future__ import annotations
import pandas as pd
from langchain_core.tools import tool
from water_quality_agent.core.session import SESSION
from water_quality_agent.core.domain import resolve_parameter

@tool
def dataset_capabilities() -> dict:
    """Retorna campos semânticos identificados e análises possíveis no dataset ativo."""
    return {"schema_map": SESSION.schema_map, "capabilities": SESSION.capabilities, "metadata": SESSION.metadata}

@tool
def list_parameters() -> list[str]:
    """Lista os parâmetros presentes no dataset ativo, sem executar análise."""
    df = SESSION.require_data()
    return sorted(df["parameter"].dropna().astype(str).unique().tolist())

@tool
def list_points() -> list[str]:
    """Lista os pontos de monitoramento disponíveis, se houver coluna semântica de ponto."""
    df = SESSION.require_data()
    if "point" not in df: return []
    return sorted(df["point"].dropna().astype(str).unique().tolist())

@tool
def resolve_water_parameter(name: str) -> dict:
    """Resolve abreviação/nome recebido para o parâmetro canônico da base CONAMA do projeto."""
    return resolve_parameter(name)

@tool
def get_series(parameter: str, point: str | None = None, start_date: str | None = None, end_date: str | None = None) -> list[dict]:
    """Recupera observações do dataset ativo para um parâmetro/ponto/período. Não calcula estatísticas."""
    df = SESSION.require_data().copy()
    # Aceita nome original ou canônico resolvido.
    mask = df["parameter"].astype(str).str.casefold().eq(str(parameter).casefold())
    if not mask.any():
        resolved = resolve_parameter(parameter).get("canonical")
        if resolved:
            # resolve cada nome presente uma vez, evitando enviar dados ao LLM
            names = df["parameter"].dropna().astype(str).unique()
            accepted = [n for n in names if resolve_parameter(n).get("canonical") == resolved]
            mask = df["parameter"].isin(accepted)
    out = df.loc[mask].copy()
    if point is not None and "point" in out:
        out = out[out["point"].astype(str).str.casefold().eq(str(point).casefold())]
    if "date" in out:
        out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True)
        if start_date: out = out[out["date"] >= pd.Timestamp(start_date)]
        if end_date: out = out[out["date"] <= pd.Timestamp(end_date)]
    cols = [c for c in ["date","point","parameter","result","unit","qualifier"] if c in out.columns]
    return out[cols].where(pd.notna(out[cols]), None).to_dict("records")

from __future__ import annotations

import pandas as pd
from langchain_core.tools import tool

from water_quality_agent.core.session import SESSION
from water_quality_agent.core.domain import resolve_parameter


@tool
def dataset_capabilities() -> dict:
    """
    Retorna os campos semânticos identificados, metadados da ingestão
    e análises disponíveis no dataset ativo.
    """
    return {
        "schema_map": SESSION.schema_map,
        "capabilities": SESSION.capabilities,
        "metadata": SESSION.metadata,
    }


@tool
def list_parameters() -> list[str]:
    """
    Lista os parâmetros canônicos presentes no dataset ativo.
    """
    df = SESSION.require_data()

    return sorted(
        df["parameter"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


@tool
def list_points() -> list[str]:
    """
    Lista os pontos de monitoramento presentes no dataset ativo.
    """
    df = SESSION.require_data()

    if "point" not in df.columns:
        return []

    points = (
        df["point"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    # O sentinel representa ausência de coluna de ponto no dataset
    return sorted(
        point
        for point in points
        if point != "__SINGLE_POINT__"
    )


@tool
def resolve_water_parameter(name: str) -> dict:
    """
    Resolve abreviações ou nomes fornecidos pelo usuário para a
    identidade canônica conhecida pelo projeto.

    Esta tool resolve a intenção do usuário. Ela não modifica o
    dataset, que já foi harmonizado durante a ingestão.
    """
    return resolve_parameter(name)


@tool
def get_series(
    parameter: str,
    point: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict]:
    """
    Recupera observações já harmonizadas do dataset ativo.

    `parameter` deve ser o nome canônico do parâmetro.

    Não realiza resolução de parâmetros, limpeza de resultados,
    extração de qualifiers ou conversão de unidades.
    """
    df = SESSION.require_data()

    mask = (
        df["parameter"]
        .astype("string")
        .str.casefold()
        .eq(str(parameter).casefold())
    )

    out = df.loc[mask].copy()

    if point is not None and "point" in out.columns:
        out = out[
            out["point"]
            .astype("string")
            .str.casefold()
            .eq(str(point).casefold())
        ]

    if "date" in out.columns:
        # date já deve estar harmonizada, mas garantimos dtype
        # temporal para filtro e ordenação.
        out["date"] = pd.to_datetime(
            out["date"],
            errors="coerce",
        )

        if start_date is not None:
            start = pd.to_datetime(start_date)
            out = out[out["date"] >= start]

        if end_date is not None:
            end = pd.to_datetime(end_date)
            out = out[out["date"] <= end]

        out = out.sort_values("date")

    columns = [
        column
        for column in [
            "date",
            "point",
            "parameter",
            "result",
            "unit",
            "qualifier",
        ]
        if column in out.columns
    ]

    result = out[columns].copy()

    # Datas precisam ser serializáveis para retorno da tool.
    if "date" in result.columns:
        result["date"] = result["date"].apply(
            lambda value: (
                value.isoformat()
                if pd.notna(value)
                else None
            )
        )

    result = result.astype(object).where(
        pd.notna(result),
        None,
    )

    return result.to_dict("records")
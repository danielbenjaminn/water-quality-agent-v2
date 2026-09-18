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

def select_series(
    parameter: str,
    point: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Seleciona uma série diretamente do dataset harmonizado no backend.

    Esta é uma função Python interna. Não é uma tool e, portanto,
    os dados selecionados não trafegam pelo LLM.

    `parameter` deve ser o nome canônico previamente resolvido.
    """
    df = SESSION.require_data().copy()

    if "parameter" not in df.columns:
        raise ValueError(
            "Dataset harmonizado não possui a coluna 'parameter'."
        )

    mask = (
        df["parameter"]
        .astype(str)
        .str.casefold()
        .eq(parameter.casefold())
    )

    if point is not None:
        if "point" not in df.columns:
            raise ValueError(
                "Dataset harmonizado não possui a coluna 'point'."
            )

        mask &= (
            df["point"]
            .astype(str)
            .str.casefold()
            .eq(point.casefold())
        )

    selected = df.loc[mask].copy()

    if "date" in selected.columns:
        selected["date"] = pd.to_datetime(
            selected["date"],
            errors="coerce",
        )

        if start_date is not None:
            selected = selected[
                selected["date"] >= pd.Timestamp(start_date)
            ]

        if end_date is not None:
            selected = selected[
                selected["date"] <= pd.Timestamp(end_date)
            ]

        selected = selected.sort_values("date")

    return selected

@tool
def get_series(
    parameter: str,
    point: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict]:
    """
    Retorna observações de uma série do dataset harmonizado.

    `parameter` deve ser o nome canônico previamente resolvido.

    Use esta tool quando for necessário inspecionar os valores da série.
    Para análises e gráficos, prefira as tools específicas, que acessam
    os dados diretamente no backend.
    """
    df = select_series(
        parameter=parameter,
        point=point,
        start_date=start_date,
        end_date=end_date,
    )

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
        if column in df.columns
    ]

    result = df[columns].copy()

    if "date" in result.columns:
        result["date"] = result["date"].apply(
            lambda value: (
                value.isoformat()
                if pd.notna(value)
                else None
            )
        )

    return (
        result
        .where(pd.notna(result), None)
        .to_dict("records")
    )
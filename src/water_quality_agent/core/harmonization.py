from __future__ import annotations

import pandas as pd

from .conversao_unidades import padronizar_unidades
from .domain import resolve_parameter, canonical_unit_for

from .parameter_resolution import (
    resolve_dataset_parameters,
)

def harmonize_dataset(
    df: pd.DataFrame,
    llm=None,
) -> tuple[pd.DataFrame, list]:
    """
    Harmoniza deterministicamente o dataset após a ingestão.

    Etapas:
    1. preserva campos originais;
    2. normaliza grafia das unidades;
    3. resolve nomes de parâmetros;
    4. determina unidade canônica de cada parâmetro.

    A conversão física dos valores será feita posteriormente.
    """

    out = df.copy()

    # =========================================================
    # PRESERVAR ORIGINAIS
    # =========================================================

    out["parameter_original"] = out["parameter"]
    out["unit_original"] = out["unit"]
    out["result_original"] = out["result"]

    # =========================================================
    # NORMALIZAÇÃO GLOBAL DAS UNIDADES
    # =========================================================

    out["unit"] = padronizar_unidades(
        out["unit"]
    )

    # =========================================================
    # RESOLUÇÃO DOS PARÂMETROS
    # =========================================================

    resolutions, pending = resolve_dataset_parameters(
        out,
        llm=llm,
    )

    parameter_map = {
        original: resolution.canonical
        for original, resolution in resolutions.items()
        if resolution.status == "resolved"
        and resolution.canonical is not None
    }

    out["parameter"] = (
        out["parameter"]
        .map(parameter_map)
        .fillna(out["parameter"])
    )

    # =========================================================
    # UNIDADE CANÔNICA POR PARÂMETRO
    # =========================================================

    unit_map = {}
    unit_source_map = {}

    canonical_parameters = (
        out["parameter"]
        .dropna()
        .astype(str)
        .unique()
    )

    for parameter in canonical_parameters:

        observations = out[
            out["parameter"] == parameter
        ]

        resolved_unit = canonical_unit_for(
            parameter=parameter,
            observations=observations,
        )

        unit_map[parameter] = resolved_unit["unit"]
        unit_source_map[parameter] = resolved_unit["source"]

    out["target_unit"] = (
        out["parameter"]
        .map(unit_map)
    )

    out["unit_source"] = (
        out["parameter"]
        .map(unit_source_map)
    )

    return out, pending
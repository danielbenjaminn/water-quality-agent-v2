from __future__ import annotations

import numpy as np
import pandas as pd

from .conversao_unidades import (
    convert_analytical_units,
    padronizar_unidades,
)
from .domain import canonical_unit_for
from .extracao_qualifier import extrair_qualif
from .parameter_resolution import (
    resolve_dataset_parameters,
)


def _parse_results(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Separa o resultado analítico em:

        result_original
        qualifier
        result

    `result` passa a ser numérico.
    """

    out = df.copy()

    qualifiers = []
    numeric_results = []

    for index, value in out["result"].items():

        # extrair_qualif trabalha com ponto decimal
        if isinstance(value, str):
            parsed_value = value.strip().replace(",", ".")
        else:
            parsed_value = value

        qualifier, numeric = extrair_qualif(
            parsed_value,
            index,
        )

        qualifiers.append(qualifier)
        numeric_results.append(numeric)

    out["qualifier"] = qualifiers

    out["result"] = pd.to_numeric(
        pd.Series(
            numeric_results,
            index=out.index,
        ),
        errors="coerce",
    )

    return out


def harmonize_dataset(
    df: pd.DataFrame,
    llm=None,
) -> tuple[pd.DataFrame, list]:
    """
    Prepara o dataset canônico para uso analítico.

    Etapas:
    1. preserva valores originais;
    2. normaliza a representação das unidades;
    3. resolve nomes de parâmetros;
    4. extrai qualifier e resultado numérico;
    5. determina unidade canônica por parâmetro;
    6. converte resultados para a unidade canônica;
    7. mantém rastreabilidade da conversão.
    """

    out = df.copy()

    # ========================================================
    # 1. PRESERVAR ORIGINAIS
    # ========================================================

    out["parameter_original"] = out["parameter"]
    out["unit_original"] = out["unit"]
    out["result_original"] = out["result"]

    # ========================================================
    # 2. NORMALIZAÇÃO TEXTUAL GLOBAL DAS UNIDADES
    # ========================================================

    out["unit"] = padronizar_unidades(
        out["unit"]
    )

    # ========================================================
    # 3. RESOLUÇÃO DOS PARÂMETROS
    # ========================================================

    resolutions, pending = resolve_dataset_parameters(
        out,
        llm=llm,
    )

    parameter_map = {
        original: resolution.canonical
        for original, resolution in resolutions.items()
        if (
            resolution.status == "resolved"
            and resolution.canonical is not None
        )
    }

    out["parameter"] = (
        out["parameter"]
        .map(parameter_map)
        .fillna(out["parameter"])
    )

    # ========================================================
    # 4. QUALIFIER + RESULTADO NUMÉRICO
    # ========================================================

    out = _parse_results(out)

    # ========================================================
    # 5. UNIDADE CANÔNICA POR PARÂMETRO
    # ========================================================

    unit_map: dict[str, str | None] = {}
    unit_source_map: dict[str, str] = {}

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

        unit_map[parameter] = (
            resolved_unit["unit"]
        )

        unit_source_map[parameter] = (
            resolved_unit["source"]
        )

    out["target_unit"] = (
        out["parameter"]
        .map(unit_map)
    )

    out["unit_source"] = (
        out["parameter"]
        .map(unit_source_map)
    )

    # ========================================================
    # 6. CONVERSÃO FÍSICA DOS RESULTADOS
    # ========================================================

    out = convert_analytical_units(out)

    # ========================================================
    # 7. TARGET_UNIT NÃO É MAIS NECESSÁRIO
    #
    # Após a conversão:
    #
    #   unit = unidade analítica final
    #
    # unit_original continua preservando a entrada.
    # ========================================================

    out.drop(
        columns=["target_unit"],
        inplace=True,
    )

    # ========================================================
    # 8. ORDEM DAS COLUNAS
    # ========================================================

    preferred_order = [
        "date",
        "point",

        "parameter_original",
        "parameter",

        "result_original",
        "qualifier",
        "result",

        "unit_original",
        "unit",

        "unit_source",
        "conversion_factor",
        "conversion_status",
    ]

    existing_preferred = [
        column
        for column in preferred_order
        if column in out.columns
    ]

    remaining = [
        column
        for column in out.columns
        if column not in existing_preferred
    ]

    out = out[
        existing_preferred + remaining
    ]

    return out, pending
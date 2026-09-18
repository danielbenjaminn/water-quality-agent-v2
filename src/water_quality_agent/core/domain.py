from __future__ import annotations

from ast import literal_eval
from pathlib import Path

import pandas as pd

from .csv_loader import carregar_csv
from .padronizar_parametros import padronizar_param, trat_string
from .conversao_unidades import padronizar_unidades


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "config"


def load_legal_limits() -> pd.DataFrame:
    return carregar_csv(
        CONFIG / "limites_legais.csv"
    ).dataframe.copy()


def load_parameter_catalog() -> pd.DataFrame:
    path = CONFIG / "parametros_catalogo.csv"

    if not path.exists():
        return pd.DataFrame(
            columns=[
                "Parâmetro",
                "Parâmetro B.D.",
                "Unidade",
            ]
        )

    return carregar_csv(path).dataframe.copy()


def parameter_catalog() -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Cria o vocabulário conhecido de parâmetros.

    Fontes:
    1. limites_legais.csv
    2. parametros_catalogo.csv
    """

    limits = load_legal_limits()
    catalog = load_parameter_catalog()

    known: dict[str, str] = {}

    # ---------------------------------------------------------
    # Legislação
    # ---------------------------------------------------------

    if "Parâmetro B.D." in limits.columns:
        for value in (
            limits["Parâmetro B.D."]
            .dropna()
            .astype(str)
            .unique()
        ):
            known[trat_string(value)] = value

    # ---------------------------------------------------------
    # Catálogo complementar
    # ---------------------------------------------------------

    if "Parâmetro B.D." in catalog.columns:
        for value in (
            catalog["Parâmetro B.D."]
            .dropna()
            .astype(str)
            .unique()
        ):
            key = trat_string(value)

            # legislação tem prioridade
            known.setdefault(key, value)

    return catalog, known


def resolve_parameter(name: str) -> dict:
    """
    Resolve um nome/alias para um parâmetro canônico.
    """

    _, catalog = parameter_catalog()

    rules_text = (
        CONFIG / "regras_parametros.txt"
    ).read_text(encoding="utf-8")

    rules = literal_eval(f"[{rules_text}]")

    canonical, score, method = padronizar_param(
        p=name,
        param_dict=catalog,
        regras_parametros=rules,
    )

    return {
        "input": name,
        "canonical": canonical,
        "score": (
            float(score)
            if score is not None
            else None
        ),
        "method": method,
    }


def legal_limits_for(parameter: str) -> list[dict]:
    """
    Retorna limites CONAMA Classe 2 do parâmetro.
    """

    limits = load_legal_limits()

    subset = limits[
        limits["Parâmetro B.D."]
        .astype(str)
        .str.casefold()
        == str(parameter).casefold()
    ].copy()

    if "Classe" in subset.columns:
        subset = subset[
            subset["Classe"]
            .astype(str)
            .str.contains(
                "Classe 2",
                case=False,
                na=False,
            )
        ]

    cols = [
        c
        for c in [
            "Legislação",
            "Classe",
            "Parâmetro B.D.",
            "Parâmetro",
            "Unidade",
            "Min",
            "Max",
            "Obs",
        ]
        if c in subset.columns
    ]

    return (
        subset[cols]
        .where(pd.notna(subset[cols]), None)
        .to_dict("records")
    )


def canonical_unit_for(
    parameter: str,
    observations: pd.DataFrame | None = None,
) -> dict:
    """
    Determina a unidade canônica de um parâmetro.

    Prioridade:
        1. limites legais
        2. catálogo complementar
        3. unidade predominante no dataset
        4. unresolved -> futuramente HITL
    """

    # ---------------------------------------------------------
    # 1. Legislação
    # ---------------------------------------------------------

    limits = legal_limits_for(parameter)

    legal_units = [
        row.get("Unidade")
        for row in limits
        if row.get("Unidade")
    ]

    if legal_units:
        unit = padronizar_unidades(
            pd.Series([legal_units[0]])
        ).iloc[0]

        return {
            "unit": unit,
            "source": "legal",
        }

    # ---------------------------------------------------------
    # 2. Catálogo complementar
    # ---------------------------------------------------------

    catalog = load_parameter_catalog()

    if not catalog.empty:
        subset = catalog[
            catalog["Parâmetro B.D."]
            .astype(str)
            .str.casefold()
            == str(parameter).casefold()
        ]

        if (
            not subset.empty
            and "Unidade" in subset.columns
        ):
            units = (
                subset["Unidade"]
                .dropna()
                .astype(str)
            )

            units = units[
                units.str.strip() != ""
            ]

            if not units.empty:
                unit = padronizar_unidades(
                    pd.Series([units.iloc[0]])
                ).iloc[0]

                return {
                    "unit": unit,
                    "source": "catalog",
                }

    # ---------------------------------------------------------
    # 3. Unidade predominante no dataset
    # ---------------------------------------------------------

    if (
        observations is not None
        and not observations.empty
        and "unit" in observations.columns
    ):
        units = padronizar_unidades(
            observations["unit"]
        ).dropna()

        units = units[
            units.astype(str).str.strip() != ""
        ]

        if not units.empty:
            predominant = units.value_counts().idxmax()

            return {
                "unit": predominant,
                "source": "dataset_predominant",
            }

    # ---------------------------------------------------------
    # 4. Futuro HITL
    # ---------------------------------------------------------

    return {
        "unit": None,
        "source": "unresolved",
    }
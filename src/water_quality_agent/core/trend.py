"""
Análise de tendência temporal para dados de qualidade da água.

Implementa o teste não paramétrico de Mann-Kendall por:

    Ponto × Parâmetro

A análise suporta dados censurados.

Regras adotadas
---------------
1. A tendência é calculada separadamente para cada
   Ponto × Parâmetro.

2. Caso não exista coluna de ponto, ou ela esteja completamente
   vazia, todo o dataset é tratado como um único ponto lógico.

3. Censura à esquerda (<):

   - Um único reporting limit:
       todos os valores censurados são considerados empatados
       abaixo dos valores quantificados.

   - Múltiplos reporting limits:
       os dados são recensurados no MAIOR reporting limit.

       Exemplo:

           <1, <1, 3, <5, 7

       torna-se:

           <5, <5, <5, <5, 7

4. Censura à direita (>) é tratada de forma análoga:

   - Um único limite:
       valores censurados ficam empatados acima dos valores
       quantificados.

   - Múltiplos limites:
       recensura no MENOR reporting limit.

5. Séries contendo simultaneamente censura à esquerda e à direita
   são consideradas não suportadas nesta primeira implementação.

6. Mudanças nos reporting limits ao longo da série geram alerta,
   pois podem indicar mudança de metodologia / capacidade analítica
   ao longo do tempo.

7. Nenhuma observação é removida do DataFrame original.

Base metodológica
-----------------
Helsel & Hirsch
Statistical Methods in Water Resources
Capítulo 12 - Trend Analysis
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm


SINGLE_POINT_LABEL = "__single_point__"

MIN_OBSERVATIONS = 4


# ============================================================
# HELPERS
# ============================================================

def _invalid_mann_kendall(
    reason: str,
) -> dict:
    """
    Estrutura padrão para um teste Mann-Kendall inválido.
    """

    return {
        "s": None,
        "variance_s": None,
        "z": None,
        "tau": None,
        "p_value": None,
        "valid": False,
        "reason": reason,
    }


def _normalize_qualifier(
    series: pd.Series,
) -> pd.Series:
    """
    Normaliza a coluna Qualifier.

    Valores vazios são tratados como ausência de qualifier.
    """

    qualifier = (
        series
        .astype("string")
        .str.strip()
    )

    qualifier = qualifier.replace(
        "",
        pd.NA,
    )

    return qualifier


def _calculate_s(
    values: np.ndarray,
) -> int:
    """
    Calcula a estatística S do teste Mann-Kendall.

    Como os dados já estão ordenados cronologicamente:

        S = soma sign(x_j - x_i), para j > i

    Empates contribuem com zero.
    """

    n = len(values)

    s = 0

    for i in range(n - 1):

        differences = (
            values[i + 1:] - values[i]
        )

        s += int(
            np.sign(differences).sum()
        )

    return s


def _calculate_variance_s(
    values: np.ndarray,
) -> float:
    """
    Calcula a variância de S com correção para empates.

    A correção é particularmente importante para dados
    censurados, pois valores abaixo do mesmo reporting limit
    são tratados como empates.
    """

    n = len(values)

    _, counts = np.unique(
        values,
        return_counts=True,
    )

    tie_correction = sum(
        count
        * (count - 1)
        * (2 * count + 5)
        for count in counts
        if count > 1
    )

    variance_s = (
        n
        * (n - 1)
        * (2 * n + 5)
        - tie_correction
    ) / 18.0

    return float(variance_s)


def _calculate_z(
    s: int,
    variance_s: float,
) -> float:
    """
    Calcula o Z padronizado do Mann-Kendall
    utilizando correção de continuidade.
    """

    if variance_s <= 0:
        return 0.0

    denominator = math.sqrt(
        variance_s
    )

    if s > 0:

        return (
            s - 1
        ) / denominator

    if s < 0:

        return (
            s + 1
        ) / denominator

    return 0.0


# ============================================================
# CENSURA
# ============================================================

def _prepare_censored_series(
    group: pd.DataFrame,
    value_col: str,
    qualifier_col: str,
) -> tuple[pd.Series, dict]:
    """
    Prepara uma série para o Mann-Kendall respeitando censura.

    Returns
    -------
    pd.Series
        Série numérica utilizada no teste.

    dict
        Metadados sobre censura e recensura.
    """

    values = pd.to_numeric(
        group[value_col],
        errors="coerce",
    ).copy()

    # --------------------------------------------------------
    # Ausência de coluna Qualifier
    # --------------------------------------------------------

    if qualifier_col not in group.columns:

        metadata = {
            "n_censored": 0,
            "censored_fraction": 0.0,

            "n_left_censored": 0,
            "n_right_censored": 0,

            "censoring_type": None,

            "n_reporting_limits": 0,
            "reporting_limits": [],

            "recensored": False,
            "recensoring_limit": None,

            "n_recensored": 0,
            "recensored_fraction": 0.0,

            "n_newly_censored": 0,

            "reporting_limit_changes_over_time": False,
            "methodology_change_warning": False,
        }

        return values, metadata

    qualifier = _normalize_qualifier(
        group[qualifier_col]
    )

    # --------------------------------------------------------
    # Máscaras
    # --------------------------------------------------------

    left_mask = qualifier.eq("<")
    right_mask = qualifier.eq(">")

    censored_mask = (
        left_mask
        | right_mask
    )

    n_total = int(
        values.notna().sum()
    )

    n_left = int(
        left_mask.sum()
    )

    n_right = int(
        right_mask.sum()
    )

    n_censored = int(
        censored_mask.sum()
    )

    # --------------------------------------------------------
    # Censura mista
    # --------------------------------------------------------

    if (
        n_left > 0
        and n_right > 0
    ):

        metadata = {
            "n_censored": n_censored,

            "censored_fraction": (
                n_censored / n_total
                if n_total
                else 0.0
            ),

            "n_left_censored": n_left,
            "n_right_censored": n_right,

            "censoring_type": "mixed",

            "n_reporting_limits": None,
            "reporting_limits": None,

            "recensored": False,
            "recensoring_limit": None,

            "n_recensored": 0,
            "recensored_fraction": 0.0,

            "n_newly_censored": 0,

            "reporting_limit_changes_over_time": False,
            "methodology_change_warning": False,

            "unsupported": True,
            "unsupported_reason": (
                "mixed_left_and_right_censoring"
            ),
        }

        return values, metadata

    # ========================================================
    # SEM CENSURA
    # ========================================================

    if n_censored == 0:

        metadata = {
            "n_censored": 0,
            "censored_fraction": 0.0,

            "n_left_censored": 0,
            "n_right_censored": 0,

            "censoring_type": None,

            "n_reporting_limits": 0,
            "reporting_limits": [],

            "recensored": False,
            "recensoring_limit": None,

            "n_recensored": 0,
            "recensored_fraction": 0.0,

            "n_newly_censored": 0,

            "reporting_limit_changes_over_time": False,
            "methodology_change_warning": False,

            "unsupported": False,
            "unsupported_reason": None,
        }

        return values, metadata

    # ========================================================
    # CENSURA À ESQUERDA
    # ========================================================

    if n_left > 0:

        reporting_limits = sorted(
            values.loc[
                left_mask
                & values.notna()
            ]
            .unique()
            .tolist()
        )

        n_limits = len(
            reporting_limits
        )

        reporting_limit_changes = (
            n_limits > 1
        )

        # ----------------------------------------------------
        # Um único reporting limit
        # ----------------------------------------------------

        if n_limits == 1:

            limit = float(
                reporting_limits[0]
            )

            # Todos os censurados recebem o mesmo valor
            # operacional para representar o empate.
            #
            # Isso NÃO significa que a concentração seja
            # igual ao reporting limit.
            values.loc[left_mask] = limit

            metadata = {
                "n_censored": n_censored,

                "censored_fraction": (
                    n_censored / n_total
                    if n_total
                    else 0.0
                ),

                "n_left_censored": n_left,
                "n_right_censored": 0,

                "censoring_type": "left",

                "n_reporting_limits": 1,
                "reporting_limits": [
                    limit
                ],

                "recensored": False,
                "recensoring_limit": None,

                "n_recensored": 0,
                "recensored_fraction": 0.0,

                "n_newly_censored": 0,

                "reporting_limit_changes_over_time": False,
                "methodology_change_warning": False,

                "unsupported": False,
                "unsupported_reason": None,
            }

            return values, metadata

        # ----------------------------------------------------
        # Múltiplos reporting limits
        # ----------------------------------------------------

        recensoring_limit = float(
            max(reporting_limits)
        )

        original_values = values.copy()

        # Observações originalmente censuradas que estavam
        # em outro reporting limit.
        censored_limit_changed_mask = (
            left_mask
            & values.notna()
            & (
                values
                != recensoring_limit
            )
        )

        # Valores quantificados abaixo do maior reporting
        # limit deixam de poder ser distinguidos dos censurados
        # naquele limite.
        newly_censored_mask = (
            ~censored_mask
            & values.notna()
            & (
                values
                < recensoring_limit
            )
        )

        # Todos passam a representar o grupo < maior limite.
        recensor_mask = (
            left_mask
            | newly_censored_mask
        )

        values.loc[
            recensor_mask
        ] = recensoring_limit

        # Quantas observações tiveram sua representação
        # efetivamente modificada?
        changed_mask = (
            recensor_mask
            & original_values.notna()
            & (
                original_values
                != values
            )
        )

        n_recensored = int(
            changed_mask.sum()
        )

        n_newly_censored = int(
            newly_censored_mask.sum()
        )

        metadata = {
            "n_censored": n_censored,

            "censored_fraction": (
                n_censored / n_total
                if n_total
                else 0.0
            ),

            "n_left_censored": n_left,
            "n_right_censored": 0,

            "censoring_type": "left",

            "n_reporting_limits": (
                n_limits
            ),

            "reporting_limits": [
                float(x)
                for x in reporting_limits
            ],

            "recensored": True,

            "recensoring_limit": (
                recensoring_limit
            ),

            # Observações cuja representação numérica
            # mudou durante a recensura.
            "n_recensored": (
                n_recensored
            ),

            "recensored_fraction": (
                n_recensored / n_total
                if n_total
                else 0.0
            ),

            # Subconjunto especialmente importante:
            # valores originalmente quantificados que
            # passaram a integrar o grupo censurado.
            "n_newly_censored": (
                n_newly_censored
            ),

            "reporting_limit_changes_over_time": (
                reporting_limit_changes
            ),

            # Não afirmamos que houve necessariamente
            # mudança de metodologia.
            #
            # Apenas sinalizamos que múltiplos reporting
            # limits ocorreram ao longo da série e que isso
            # é compatível com uma possível mudança analítica.
            "methodology_change_warning": True,

            "unsupported": False,
            "unsupported_reason": None,
        }

        return values, metadata

    # ========================================================
    # CENSURA À DIREITA
    # ========================================================

    reporting_limits = sorted(
        values.loc[
            right_mask
            & values.notna()
        ]
        .unique()
        .tolist()
    )

    n_limits = len(
        reporting_limits
    )

    reporting_limit_changes = (
        n_limits > 1
    )

    # --------------------------------------------------------
    # Um único limite
    # --------------------------------------------------------

    if n_limits == 1:

        limit = float(
            reporting_limits[0]
        )

        values.loc[
            right_mask
        ] = limit

        metadata = {
            "n_censored": n_censored,

            "censored_fraction": (
                n_censored / n_total
                if n_total
                else 0.0
            ),

            "n_left_censored": 0,
            "n_right_censored": n_right,

            "censoring_type": "right",

            "n_reporting_limits": 1,
            "reporting_limits": [
                limit
            ],

            "recensored": False,
            "recensoring_limit": None,

            "n_recensored": 0,
            "recensored_fraction": 0.0,

            "n_newly_censored": 0,

            "reporting_limit_changes_over_time": False,
            "methodology_change_warning": False,

            "unsupported": False,
            "unsupported_reason": None,
        }

        return values, metadata

    # --------------------------------------------------------
    # Múltiplos limites à direita
    # --------------------------------------------------------
    #
    # Para >10, >20 etc., o limite comum conservador é
    # o MENOR reporting limit.
    # --------------------------------------------------------

    recensoring_limit = float(
        min(reporting_limits)
    )

    original_values = values.copy()

    newly_censored_mask = (
        ~censored_mask
        & values.notna()
        & (
            values
            > recensoring_limit
        )
    )

    recensor_mask = (
        right_mask
        | newly_censored_mask
    )

    values.loc[
        recensor_mask
    ] = recensoring_limit

    changed_mask = (
        recensor_mask
        & original_values.notna()
        & (
            original_values
            != values
        )
    )

    n_recensored = int(
        changed_mask.sum()
    )

    n_newly_censored = int(
        newly_censored_mask.sum()
    )

    metadata = {
        "n_censored": n_censored,

        "censored_fraction": (
            n_censored / n_total
            if n_total
            else 0.0
        ),

        "n_left_censored": 0,
        "n_right_censored": n_right,

        "censoring_type": "right",

        "n_reporting_limits": (
            n_limits
        ),

        "reporting_limits": [
            float(x)
            for x in reporting_limits
        ],

        "recensored": True,

        "recensoring_limit": (
            recensoring_limit
        ),

        "n_recensored": (
            n_recensored
        ),

        "recensored_fraction": (
            n_recensored / n_total
            if n_total
            else 0.0
        ),

        "n_newly_censored": (
            n_newly_censored
        ),

        "reporting_limit_changes_over_time": (
            reporting_limit_changes
        ),

        "methodology_change_warning": True,

        "unsupported": False,
        "unsupported_reason": None,
    }

    return values, metadata


# ============================================================
# MANN-KENDALL
# ============================================================

def _mann_kendall(
    dates: pd.Series,
    values: pd.Series,
) -> dict:
    """
    Executa o teste Mann-Kendall.

    Parameters
    ----------
    dates:
        Datas das observações.

    values:
        Valores preparados para o teste, incluindo eventual
        tratamento ordinal da censura.

    Returns
    -------
    dict
        Estatística S, variância, Z, tau e p-value.
    """

    temp = pd.DataFrame(
        {
            "date": dates,
            "value": values,
        }
    )

    temp = temp.dropna(
        subset=[
            "date",
            "value",
        ]
    )

    temp = temp.sort_values(
        "date"
    )

    n = len(temp)

    if n < MIN_OBSERVATIONS:

        return _invalid_mann_kendall(
            "insufficient_observations"
        )

    # --------------------------------------------------------
    # Datas duplicadas
    # --------------------------------------------------------
    #
    # O Mann-Kendall assume uma ordenação temporal.
    # Duas observações do mesmo parâmetro no mesmo instante
    # não possuem relação temporal entre si.
    #
    # Não escolhemos arbitrariamente uma ordem.
    # --------------------------------------------------------

    if temp["date"].duplicated().any():

        return _invalid_mann_kendall(
            "duplicate_dates"
        )

    y = (
        temp["value"]
        .astype(float)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Série constante
    # --------------------------------------------------------

    if len(
        np.unique(y)
    ) < 2:

        return _invalid_mann_kendall(
            "constant_values"
        )

    # --------------------------------------------------------
    # S
    # --------------------------------------------------------

    s = _calculate_s(
        y
    )

    # --------------------------------------------------------
    # Variância corrigida para empates
    # --------------------------------------------------------

    variance_s = (
        _calculate_variance_s(
            y
        )
    )

    if variance_s <= 0:

        return _invalid_mann_kendall(
            "zero_variance"
        )

    # --------------------------------------------------------
    # Z
    # --------------------------------------------------------

    z = _calculate_z(
        s=s,
        variance_s=variance_s,
    )

    # --------------------------------------------------------
    # p-value bilateral
    # --------------------------------------------------------

    p_value = float(
        2.0
        * (
            1.0
            - norm.cdf(
                abs(z)
            )
        )
    )

    # --------------------------------------------------------
    # Kendall Tau
    # --------------------------------------------------------
    #
    # O tau é calculado contra o tempo cronológico.
    #
    # scipy.stats.kendalltau utiliza correção para empates.
    # --------------------------------------------------------

    time_numeric = (
        temp["date"]
        .map(pd.Timestamp.toordinal)
        .to_numpy()
    )

    tau_result = kendalltau(
        time_numeric,
        y,
        nan_policy="omit",
    )

    tau = (
        float(
            tau_result.statistic
        )
        if not pd.isna(
            tau_result.statistic
        )
        else None
    )

    return {
        "s": int(s),
        "variance_s": float(
            variance_s
        ),
        "z": float(z),
        "tau": tau,
        "p_value": p_value,
        "valid": True,
        "reason": None,
    }


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def mann_kendall_analysis(
    df: pd.DataFrame,
    date_col: str = "Data",
    point_col: str | None = "Ponto",
    parameter_col: str = "Parâmetro",
    value_col: str = "Resultado Num",
    qualifier_col: str = "Qualifier",
) -> list[dict]:
    """
    Executa análise de tendência Mann-Kendall para cada
    combinação Ponto × Parâmetro.

    Parameters
    ----------
    df:
        DataFrame normalizado pela pipeline.

    date_col:
        Coluna canônica de data.

        Padrão:
            "Data"

    point_col:
        Coluna canônica do ponto.

        Padrão:
            "Ponto"

        Se ausente, None ou completamente vazia,
        todo o dataset é tratado como um único ponto.

    parameter_col:
        Coluna canônica do parâmetro analítico.

        Padrão:
            "Parâmetro"

    value_col:
        Coluna numérica contendo o resultado analítico.

        Padrão:
            "Resultado Num"

    qualifier_col:
        Coluna contendo os qualificadores de censura.

        Padrão:
            "Qualifier"

    Returns
    -------
    list[dict]

        Um resultado para cada:

            Ponto × Parâmetro
    """

    # ========================================================
    # VALIDAÇÃO
    # ========================================================

    required_columns = [
        date_col,
        parameter_col,
        value_col,
    ]

    for col in required_columns:

        if col not in df.columns:

            raise ValueError(
                f"Coluna obrigatória não encontrada: {col}"
            )

    # ========================================================
    # CÓPIA
    # ========================================================

    work_df = df.copy()

    # ========================================================
    # DATA
    # ========================================================

    work_df[date_col] = pd.to_datetime(
        work_df[date_col],
        errors="coerce",
        dayfirst=True,
    )

    # ========================================================
    # VALORES
    # ========================================================

    work_df[value_col] = pd.to_numeric(
        work_df[value_col],
        errors="coerce",
    )

    # ========================================================
    # PONTO
    # ========================================================

    point_inferred = False

    if (
        point_col is None
        or point_col not in work_df.columns
        or work_df[point_col].isna().all()
    ):

        internal_point_col = (
            "__analysis_point__"
        )

        work_df[
            internal_point_col
        ] = SINGLE_POINT_LABEL

        point_inferred = True

    else:

        internal_point_col = (
            point_col
        )

    # ========================================================
    # RESULTADOS
    # ========================================================

    results: list[dict] = []

    # ========================================================
    # PONTO × PARÂMETRO
    # ========================================================

    grouped = work_df.groupby(
        [
            internal_point_col,
            parameter_col,
        ],
        dropna=False,
        sort=False,
    )

    for (
        point,
        parameter,
    ), group in grouped:

        group = group.copy()

        # ----------------------------------------------------
        # Registros totais
        # ----------------------------------------------------

        n_total = len(
            group
        )

        # ----------------------------------------------------
        # Registros numericamente utilizáveis
        # ----------------------------------------------------

        valid_mask = (
            group[date_col].notna()
            & group[value_col].notna()
        )

        analysis_group = (
            group.loc[
                valid_mask
            ]
            .copy()
        )

        n_valid = len(
            analysis_group
        )

        n_missing = (
            n_total
            - n_valid
        )

        # ----------------------------------------------------
        # Prepara censura
        # ----------------------------------------------------

        prepared_values, censoring = (
            _prepare_censored_series(
                group=analysis_group,
                value_col=value_col,
                qualifier_col=qualifier_col,
            )
        )

        # ----------------------------------------------------
        # Caso não suportado
        # ----------------------------------------------------

        if censoring.get(
            "unsupported",
            False,
        ):

            mann_kendall = (
                _invalid_mann_kendall(
                    censoring[
                        "unsupported_reason"
                    ]
                )
            )

        else:

            mann_kendall = (
                _mann_kendall(
                    dates=analysis_group[
                        date_col
                    ],
                    values=prepared_values,
                )
            )

        # ----------------------------------------------------
        # Resultado agregado
        # ----------------------------------------------------

        results.append(
            {
                "point": point,
                "point_inferred": (
                    point_inferred
                ),

                "parameter": parameter,

                "n_total": int(
                    n_total
                ),

                "n_valid": int(
                    n_valid
                ),

                "n_missing": int(
                    n_missing
                ),

                "censoring": censoring,

                "mann_kendall": (
                    mann_kendall
                ),
            }
        )

    return results
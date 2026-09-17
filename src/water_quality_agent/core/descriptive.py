"""
Estatísticas descritivas para dados de qualidade da água.

Base metodológica:
    Helsel, D.R. & Hirsch, R.M.
    Statistical Methods in Water Resources.

O módulo calcula estatísticas clássicas e resistentes por
parâmetro e ponto de monitoramento.

Princípios:
- não remove outliers;
- não substitui valores censurados por valores arbitrários;
- preserva o limite de reporte presente em Resultado Num;
- contabiliza censura;
- calcula estatísticas mesmo quando há censura;
- informa se cada estatística pode ser interpretada diretamente;
- não realiza interpretação ambiental;
- não realiza testes de hipótese.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import gmean, skew, trim_mean


SINGLE_POINT_LABEL = "__single_point__"


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def _calculate_quartile_skew(
    q25: float,
    median: float,
    q75: float,
) -> float | None:
    """
    Calcula o coeficiente resistente de assimetria baseado
    nos quartis.

    Retorna None quando IQR = 0.

    Fórmula:
        ((Q3 - mediana) - (mediana - Q1)) / IQR

    Fonte:
        Helsel & Hirsch - Statistical Methods in Water Resources.
    """

    iqr = q75 - q25

    if iqr == 0:
        return None

    return (
        (q75 - median)
        - (median - q25)
    ) / iqr


def _metric(
    value,
    valid: bool,
    reason: str | None = None,
) -> dict:
    """
    Padroniza a saída de uma estatística.

    Parameters
    ----------
    value:
        Valor calculado da estatística.

    valid:
        Indica se o valor pode ser interpretado diretamente
        como estimativa da estatística.

    reason:
        Motivo pelo qual a estatística não deve ser
        interpretada diretamente.

    Returns
    -------
    dict
        Estrutura padronizada:

        {
            "value": ...,
            "valid": True/False,
            "reason": ...
        }
    """

    return {
        "value": value,
        "valid": bool(valid),
        "reason": reason,
    }


# ============================================================
# CÁLCULO DAS ESTATÍSTICAS
# ============================================================

def _calculate_statistics(
    values: pd.Series,
    qualifiers: pd.Series | None = None,
    trim_proportion: float = 0.25,
) -> dict:
    """
    Calcula estatísticas descritivas clássicas e resistentes.

    Quando existem observações censuradas, os valores
    numéricos associados aos qualificadores representam
    limites de reporte e não necessariamente valores medidos.

    As estatísticas continuam sendo calculadas para manter
    rastreabilidade, mas recebem indicadores de validade
    metodológica.

    Parameters
    ----------
    values:
        Série contendo os valores numéricos.

        Para observações censuradas, o valor representa
        o limite associado à censura.

        Exemplo:

            Resultado original: <5
            Resultado Num:       5
            Qualifier:           <

    qualifiers:
        Série contendo os qualificadores associados.

        Exemplos:

            "<"  -> censura à esquerda
            "<=" -> censura à esquerda
            "≤"  -> censura à esquerda

            ">"  -> censura à direita
            ">=" -> censura à direita
            "≥"  -> censura à direita

        NaN, None ou string vazia representam observações
        não censuradas.

    trim_proportion:
        Proporção removida de CADA extremidade para cálculo
        da média aparada.

        O padrão 0.25 remove 25% de cada extremidade,
        utilizando os 50% centrais.

    Returns
    -------
    dict
        Estatísticas descritivas e metadados de censura.
    """

    # --------------------------------------------------------
    # Preparação
    # --------------------------------------------------------

    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    n_total = len(numeric)

    n_missing = int(
        numeric.isna().sum()
    )

    clean = numeric.dropna()

    n_valid = len(clean)

    missing_fraction = (
        n_missing / n_total
        if n_total > 0
        else None
    )

    # --------------------------------------------------------
    # Censura
    # --------------------------------------------------------

    if qualifiers is not None:

        # Garante que qualifier e values mantenham
        # exatamente os mesmos índices.
        qualifier = qualifiers.reindex(values.index)

        qualifier = (
            qualifier
            .fillna("")
            .astype(str)
            .str.strip()
        )

        left_censored = qualifier.isin([
            "<",
            "<=",
            "≤",
        ])

        right_censored = qualifier.isin([
            ">",
            ">=",
            "≥",
        ])

        n_left_censored = int(
            left_censored.sum()
        )

        n_right_censored = int(
            right_censored.sum()
        )

    else:

        n_left_censored = 0
        n_right_censored = 0

    n_censored = (
        n_left_censored
        + n_right_censored
    )

    # --------------------------------------------------------
    # Proporções de censura
    # --------------------------------------------------------

    censored_fraction = (
        n_censored / n_total
        if n_total > 0
        else None
    )

    left_censored_fraction = (
        n_left_censored / n_total
        if n_total > 0
        else None
    )

    right_censored_fraction = (
        n_right_censored / n_total
        if n_total > 0
        else None
    )

    # --------------------------------------------------------
    # Caracterização da censura
    # --------------------------------------------------------

    has_censoring = (
        n_censored > 0
    )

    has_left_censoring = (
        n_left_censored > 0
    )

    has_right_censoring = (
        n_right_censored > 0
    )

    mixed_censoring = (
        has_left_censoring
        and has_right_censoring
    )

    # --------------------------------------------------------
    # Motivo geral para estatísticas baseadas diretamente
    # na magnitude dos valores
    # --------------------------------------------------------

    if not has_censoring:

        censoring_reason = None

    elif mixed_censoring:

        censoring_reason = (
            "left_and_right_censored_values_"
            "treated_as_reporting_limits"
        )

    elif has_left_censoring:

        censoring_reason = (
            "left_censored_values_"
            "treated_as_reporting_limits"
        )

    else:

        censoring_reason = (
            "right_censored_values_"
            "treated_as_reporting_limits"
        )

    # --------------------------------------------------------
    # Validade das estatísticas dependentes da magnitude
    # --------------------------------------------------------
    #
    # Média, desvio-padrão, variância etc. dependem dos
    # valores numéricos propriamente ditos.
    #
    # Portanto, se <5 estiver representado numericamente
    # como 5, esses valores podem ser calculados, mas não
    # devem ser interpretados como estimativas convencionais.
    # --------------------------------------------------------

    magnitude_stats_valid = (
        not has_censoring
    )

    # --------------------------------------------------------
    # Validade da mediana
    # --------------------------------------------------------
    #
    # Para censura em uma única direção:
    #
    # < 50% censurado:
    #     a mediana amostral pode permanecer conhecida.
    #
    # >= 50%:
    #     não consideramos a estimativa diretamente válida.
    #
    # Para censura simultânea à esquerda e à direita,
    # adotamos uma postura conservadora no MVP.
    # --------------------------------------------------------

    if not has_censoring:

        median_valid = True
        median_reason = None

    elif mixed_censoring:

        median_valid = False

        median_reason = (
            "mixed_left_and_right_censoring"
        )

    elif has_left_censoring:

        median_valid = (
            left_censored_fraction < 0.50
        )

        median_reason = (
            None
            if median_valid
            else "left_censoring_at_or_above_50_percent"
        )

    else:

        median_valid = (
            right_censored_fraction < 0.50
        )

        median_reason = (
            None
            if median_valid
            else "right_censoring_at_or_above_50_percent"
        )

    # --------------------------------------------------------
    # Validade dos quartis / IQR
    # --------------------------------------------------------
    #
    # Regra conservadora para o MVP:
    #
    # < 25% de censura em uma única direção:
    #     quartis e IQR podem ser utilizados.
    #
    # >= 25%:
    #     mantemos os cálculos, mas sinalizamos que não
    #     devem ser interpretados diretamente.
    #
    # Censura mista:
    #     marcada como não válida.
    # --------------------------------------------------------

    if not has_censoring:

        quartiles_valid = True
        quartiles_reason = None

    elif mixed_censoring:

        quartiles_valid = False

        quartiles_reason = (
            "mixed_left_and_right_censoring"
        )

    elif has_left_censoring:

        quartiles_valid = (
            left_censored_fraction < 0.25
        )

        quartiles_reason = (
            None
            if quartiles_valid
            else "left_censoring_at_or_above_25_percent"
        )

    else:

        quartiles_valid = (
            right_censored_fraction < 0.25
        )

        quartiles_reason = (
            None
            if quartiles_valid
            else "right_censoring_at_or_above_25_percent"
        )

    # --------------------------------------------------------
    # Sem observações numéricas válidas
    # --------------------------------------------------------

    if n_valid == 0:

        no_data_reason = (
            "no_valid_numeric_values"
        )

        return {
            # Amostragem
            "n_total": int(n_total),
            "n_valid": 0,
            "n_missing": n_missing,
            "missing_fraction": missing_fraction,

            # Censura
            "n_censored": n_censored,
            "censored_fraction": censored_fraction,

            "n_left_censored": n_left_censored,
            "left_censored_fraction":
                left_censored_fraction,

            "n_right_censored": n_right_censored,
            "right_censored_fraction":
                right_censored_fraction,

            # Extremos
            "min": _metric(
                None,
                False,
                no_data_reason,
            ),

            "max": _metric(
                None,
                False,
                no_data_reason,
            ),

            "range": _metric(
                None,
                False,
                no_data_reason,
            ),

            # Posição
            "mean": _metric(
                None,
                False,
                no_data_reason,
            ),

            "median": _metric(
                None,
                False,
                no_data_reason,
            ),

            "geometric_mean": _metric(
                None,
                False,
                no_data_reason,
            ),

            "trimmed_mean": _metric(
                None,
                False,
                no_data_reason,
            ),

            # Percentis
            "q25": _metric(
                None,
                False,
                no_data_reason,
            ),

            "q50": _metric(
                None,
                False,
                no_data_reason,
            ),

            "q75": _metric(
                None,
                False,
                no_data_reason,
            ),

            # Dispersão
            "variance": _metric(
                None,
                False,
                no_data_reason,
            ),

            "std": _metric(
                None,
                False,
                no_data_reason,
            ),

            "iqr": _metric(
                None,
                False,
                no_data_reason,
            ),

            "mad": _metric(
                None,
                False,
                no_data_reason,
            ),

            # Assimetria
            "skewness": _metric(
                None,
                False,
                no_data_reason,
            ),

            "quartile_skew": _metric(
                None,
                False,
                no_data_reason,
            ),
        }

    # ========================================================
    # CÁLCULOS
    # ========================================================

    # --------------------------------------------------------
    # Valores básicos
    # --------------------------------------------------------

    minimum = float(
        clean.min()
    )

    maximum = float(
        clean.max()
    )

    # --------------------------------------------------------
    # Percentis
    # --------------------------------------------------------

    q25 = float(
        clean.quantile(0.25)
    )

    median = float(
        clean.median()
    )

    q75 = float(
        clean.quantile(0.75)
    )

    iqr = (
        q75 - q25
    )

    # --------------------------------------------------------
    # Medidas de posição
    # --------------------------------------------------------

    mean = float(
        clean.mean()
    )

    trimmed_mean = float(
        trim_mean(
            clean.to_numpy(),
            proportiontocut=trim_proportion,
        )
    )

    # --------------------------------------------------------
    # Média geométrica
    # --------------------------------------------------------

    if (clean > 0).all():

        geometric_mean = float(
            gmean(
                clean.to_numpy()
            )
        )

        geometric_mean_computable = True

    else:

        geometric_mean = None
        geometric_mean_computable = False

    # --------------------------------------------------------
    # Medidas de dispersão
    # --------------------------------------------------------

    data_range = (
        maximum - minimum
    )

    if n_valid >= 2:

        variance = float(
            clean.var(
                ddof=1
            )
        )

        std = float(
            clean.std(
                ddof=1
            )
        )

    else:

        variance = None
        std = None

    # --------------------------------------------------------
    # Median Absolute Deviation (MAD)
    # --------------------------------------------------------
    #
    # Definição adotada por Helsel & Hirsch:
    #
    # MAD = median(|Xi - median(X)|)
    #
    # Não aplicamos fator 1.4826.
    # --------------------------------------------------------

    mad = float(
        np.median(
            np.abs(
                clean - median
            )
        )
    )

    # --------------------------------------------------------
    # Assimetria clássica
    # --------------------------------------------------------

    if (
        n_valid >= 3
        and clean.nunique() > 1
    ):

        skewness = float(
            skew(
                clean.to_numpy(),
                bias=False,
            )
        )

    else:

        skewness = None

    # --------------------------------------------------------
    # Assimetria resistente
    # --------------------------------------------------------

    quartile_skew = (
        _calculate_quartile_skew(
            q25=q25,
            median=median,
            q75=q75,
        )
    )

    # ========================================================
    # RESULTADO
    # ========================================================

    return {
        # ----------------------------------------------------
        # Amostragem
        # ----------------------------------------------------

        "n_total": int(
            n_total
        ),

        "n_valid": int(
            n_valid
        ),

        "n_missing": n_missing,

        "missing_fraction": (
            float(missing_fraction)
            if missing_fraction is not None
            else None
        ),

        # ----------------------------------------------------
        # Censura
        # ----------------------------------------------------

        "n_censored": n_censored,

        "censored_fraction": (
            float(censored_fraction)
            if censored_fraction is not None
            else None
        ),

        "n_left_censored":
            n_left_censored,

        "left_censored_fraction": (
            float(left_censored_fraction)
            if left_censored_fraction is not None
            else None
        ),

        "n_right_censored":
            n_right_censored,

        "right_censored_fraction": (
            float(right_censored_fraction)
            if right_censored_fraction is not None
            else None
        ),

        # ----------------------------------------------------
        # Extremos
        # ----------------------------------------------------

        # Se houver censura à esquerda, o menor valor
        # calculado pode ser apenas um limite de reporte.

        "min": _metric(
            minimum,
            not has_left_censoring,
            (
                None
                if not has_left_censoring
                else "left_censored_values_present"
            ),
        ),

        # Se houver censura à direita, o maior valor
        # calculado pode ser apenas um limite.

        "max": _metric(
            maximum,
            not has_right_censoring,
            (
                None
                if not has_right_censoring
                else "right_censored_values_present"
            ),
        ),

        "range": _metric(
            float(data_range),
            not has_censoring,
            censoring_reason,
        ),

        # ----------------------------------------------------
        # Medidas de posição
        # ----------------------------------------------------

        "mean": _metric(
            mean,
            magnitude_stats_valid,
            censoring_reason,
        ),

        "median": _metric(
            median,
            median_valid,
            median_reason,
        ),

        "geometric_mean": _metric(
            geometric_mean,
            (
                geometric_mean_computable
                and magnitude_stats_valid
            ),
            (
                "non_positive_values_present"
                if not geometric_mean_computable
                else censoring_reason
            ),
        ),

        "trimmed_mean": _metric(
            trimmed_mean,
            magnitude_stats_valid,
            censoring_reason,
        ),

        # ----------------------------------------------------
        # Percentis
        # ----------------------------------------------------

        "q25": _metric(
            q25,
            quartiles_valid,
            quartiles_reason,
        ),

        "q50": _metric(
            median,
            median_valid,
            median_reason,
        ),

        "q75": _metric(
            q75,
            quartiles_valid,
            quartiles_reason,
        ),

        # ----------------------------------------------------
        # Dispersão
        # ----------------------------------------------------

        "variance": _metric(
            variance,
            (
                variance is not None
                and magnitude_stats_valid
            ),
            (
                "insufficient_sample_size"
                if variance is None
                else censoring_reason
            ),
        ),

        "std": _metric(
            std,
            (
                std is not None
                and magnitude_stats_valid
            ),
            (
                "insufficient_sample_size"
                if std is None
                else censoring_reason
            ),
        ),

        "iqr": _metric(
            float(iqr),
            quartiles_valid,
            quartiles_reason,
        ),

        "mad": _metric(
            mad,
            magnitude_stats_valid,
            censoring_reason,
        ),

        # ----------------------------------------------------
        # Assimetria
        # ----------------------------------------------------

        "skewness": _metric(
            skewness,
            (
                skewness is not None
                and magnitude_stats_valid
            ),
            (
                "insufficient_sample_size_or_constant_values"
                if skewness is None
                else censoring_reason
            ),
        ),

        "quartile_skew": _metric(
            quartile_skew,
            (
                quartile_skew is not None
                and quartiles_valid
            ),
            (
                "iqr_is_zero"
                if quartile_skew is None
                else quartiles_reason
            ),
        ),
    }


# ============================================================
# ANÁLISE POR PONTO × PARÂMETRO
# ============================================================

def descriptive_analysis(
    df: pd.DataFrame,
    parameter_col: str,
    value_col: str,
    point_col: str | None = None,
    qualifier_col: str | None = "Qualifier",
    trim_proportion: float = 0.25,
) -> list[dict]:
    """
    Executa estatística descritiva por ponto e parâmetro.

    Se não houver coluna de ponto, todos os registros são
    considerados pertencentes a um único ponto lógico.

    Parameters
    ----------
    df:
        DataFrame normalizado.

    parameter_col:
        Nome da coluna contendo o parâmetro.

    value_col:
        Nome da coluna contendo o valor numérico.

    point_col:
        Nome da coluna contendo o ponto de monitoramento.

        Se None, inexistente ou totalmente vazia, o dataset
        será tratado como pertencente a um único ponto.

    qualifier_col:
        Nome da coluna contendo os qualificadores de censura.

        O padrão é "Qualifier".

    trim_proportion:
        Proporção removida de cada extremidade para cálculo
        da média aparada.

    Returns
    -------
    list[dict]
        Uma entrada para cada combinação:

            ponto × parâmetro
    """

    # --------------------------------------------------------
    # Validação das colunas obrigatórias
    # --------------------------------------------------------

    if parameter_col not in df.columns:
        raise ValueError(
            f"Coluna de parâmetro não encontrada: "
            f"{parameter_col}"
        )

    if value_col not in df.columns:
        raise ValueError(
            f"Coluna de valor não encontrada: "
            f"{value_col}"
        )

    work_df = df.copy()

    # --------------------------------------------------------
    # Tratamento da dimensão "ponto"
    # --------------------------------------------------------

    point_inferred = False

    if (
        point_col is None
        or point_col not in work_df.columns
    ):

        internal_point_col = (
            "__analysis_point__"
        )

        work_df[internal_point_col] = (
            SINGLE_POINT_LABEL
        )

        point_inferred = True

    elif work_df[point_col].isna().all():

        internal_point_col = (
            "__analysis_point__"
        )

        work_df[internal_point_col] = (
            SINGLE_POINT_LABEL
        )

        point_inferred = True

    else:

        internal_point_col = (
            point_col
        )

    # --------------------------------------------------------
    # Agrupamento
    # --------------------------------------------------------

    grouped = work_df.groupby(
        [
            internal_point_col,
            parameter_col,
        ],
        dropna=False,
        sort=False,
    )

    results = []

    # --------------------------------------------------------
    # Análise grupo a grupo
    # --------------------------------------------------------

    for (
        point,
        parameter,
    ), group in grouped:

        # Qualificadores são opcionais.
        #
        # Se a coluna não existir, assume-se que não há
        # informação explícita sobre censura.

        qualifiers = (
            group[qualifier_col]
            if (
                qualifier_col is not None
                and qualifier_col in group.columns
            )
            else None
        )

        statistics = (
            _calculate_statistics(
                values=group[value_col],
                qualifiers=qualifiers,
                trim_proportion=trim_proportion,
            )
        )

        result = {
            "point": point,
            "point_inferred": point_inferred,
            "parameter": parameter,
            **statistics,
        }

        results.append(
            result
        )

    return results
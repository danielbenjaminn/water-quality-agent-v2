"""
Identificação de valores potencialmente atípicos em dados de
qualidade da água utilizando a regra de Tukey baseada no IQR.

O módulo NÃO remove observações.

Ele apenas adiciona ao DataFrame uma coluna categórica:

    Outlier_IQR

Classificações possíveis:

    none
    lower_outside
    lower_far_outside
    upper_outside
    upper_far_outside
    insufficient_data

Base metodológica:
    Tukey boxplot / Helsel & Hirsch
    Statistical Methods in Water Resources.

Observações censuradas permanecem na análise utilizando o
valor numérico associado ao limite de reporte.

Exemplo:

    <0.05

é representado numericamente como:

    0.05

para fins do cálculo operacional dos quartis e IQR.

O qualifier original permanece preservado no DataFrame.
"""

from __future__ import annotations

import pandas as pd


SINGLE_POINT_LABEL = "__single_point__"


# ============================================================
# CONSTANTES
# ============================================================

OUTLIER_COLUMN = "Outlier_IQR"

NO_OUTLIER = "none"

LOWER_OUTSIDE = "lower_outside"
LOWER_FAR_OUTSIDE = "lower_far_outside"

UPPER_OUTSIDE = "upper_outside"
UPPER_FAR_OUTSIDE = "upper_far_outside"

INSUFFICIENT_DATA = "insufficient_data"


# Número mínimo de observações numéricas para tentar aplicar
# a regra de Tukey.
#
# Com menos observações, os quartis/IQR tornam-se pouco
# informativos para identificação de observações incomuns.
MIN_SAMPLE_SIZE = 4


# ============================================================
# FUNÇÃO INTERNA
# ============================================================

def _flag_group_outliers(
    group: pd.DataFrame,
    value_col: str,
) -> pd.Series:
    """
    Identifica observações potencialmente atípicas dentro
    de um único grupo Ponto × Parâmetro.

    Utiliza a regra de Tukey:

        IQR = Q3 - Q1

        lower_fence = Q1 - 1.5 * IQR
        upper_fence = Q3 + 1.5 * IQR

        lower_far_fence = Q1 - 3 * IQR
        upper_far_fence = Q3 + 3 * IQR

    Parameters
    ----------
    group:
        DataFrame correspondente a um único
        Ponto × Parâmetro.

    value_col:
        Coluna numérica utilizada para cálculo.

    Returns
    -------
    pd.Series
        Série com a classificação de cada observação.
    """

    values = pd.to_numeric(
        group[value_col],
        errors="coerce",
    )

    result = pd.Series(
        NO_OUTLIER,
        index=group.index,
        dtype="object",
    )

    # --------------------------------------------------------
    # Valores ausentes
    # --------------------------------------------------------

    valid = values.dropna()

    n_valid = len(valid)

    # --------------------------------------------------------
    # Amostra insuficiente
    # --------------------------------------------------------

    if n_valid < MIN_SAMPLE_SIZE:

        result.loc[values.notna()] = (
            INSUFFICIENT_DATA
        )

        return result

    # --------------------------------------------------------
    # Quartis
    # --------------------------------------------------------

    q1 = float(
        valid.quantile(0.25)
    )

    q3 = float(
        valid.quantile(0.75)
    )

    iqr = (
        q3 - q1
    )

    # --------------------------------------------------------
    # IQR = 0
    # --------------------------------------------------------
    #
    # Não utilizamos a regra automaticamente quando não
    # existe dispersão interquartil.
    #
    # Isso evita situações como:
    #
    #     0.0003
    #     0.0003
    #     0.0003
    #     0.0004
    #
    # onde Q1 = Q3 e qualquer pequena diferença poderia ser
    # classificada artificialmente como outlier.
    # --------------------------------------------------------

    if iqr == 0:

        result.loc[values.notna()] = (
            INSUFFICIENT_DATA
        )

        return result

    # --------------------------------------------------------
    # Limites de Tukey
    # --------------------------------------------------------

    lower_fence = (
        q1 - 1.5 * iqr
    )

    upper_fence = (
        q3 + 1.5 * iqr
    )

    lower_far_fence = (
        q1 - 3.0 * iqr
    )

    upper_far_fence = (
        q3 + 3.0 * iqr
    )

    # --------------------------------------------------------
    # Outside
    # --------------------------------------------------------

    lower_outside_mask = (
        (values < lower_fence)
        & (values >= lower_far_fence)
    )

    upper_outside_mask = (
        (values > upper_fence)
        & (values <= upper_far_fence)
    )

    # --------------------------------------------------------
    # Far outside
    # --------------------------------------------------------

    lower_far_mask = (
        values < lower_far_fence
    )

    upper_far_mask = (
        values > upper_far_fence
    )

    # --------------------------------------------------------
    # Classificação
    # --------------------------------------------------------

    result.loc[
        lower_outside_mask
    ] = LOWER_OUTSIDE

    result.loc[
        upper_outside_mask
    ] = UPPER_OUTSIDE

    result.loc[
        lower_far_mask
    ] = LOWER_FAR_OUTSIDE

    result.loc[
        upper_far_mask
    ] = UPPER_FAR_OUTSIDE

    return result


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def flag_iqr_outliers(
    df: pd.DataFrame,
    parameter_col: str = "Parâmetro B.D.",
    value_col: str = "Resultado",
    point_col: str | None = "Ponto",
) -> pd.DataFrame:
    """
    Adiciona ao DataFrame a coluna Outlier_IQR utilizando
    a regra de Tukey por Ponto × Parâmetro.

    A função NÃO remove observações.

    Parameters
    ----------
    df:
        DataFrame normalizado pela pipeline.

    parameter_col:
        Nome da coluna contendo o parâmetro canônico.

        Padrão:
            "Parâmetro B.D."

    value_col:
        Nome da coluna contendo o resultado convertido para
        a unidade canônica.

        Padrão:
            "Resultado"

    point_col:
        Nome da coluna contendo o ponto de monitoramento.

        Padrão:
            "Ponto"

        Se a coluna estiver ausente, for None ou estiver
        completamente vazia, todo o dataset será tratado
        como pertencente a um único ponto lógico.

    Returns
    -------
    pd.DataFrame
        Cópia do DataFrame original contendo a nova coluna:

            Outlier_IQR
    """

    # --------------------------------------------------------
    # Validação
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

    # Inicializa explicitamente a coluna.
    work_df[OUTLIER_COLUMN] = NO_OUTLIER

    # --------------------------------------------------------
    # Tratamento da dimensão ponto
    # --------------------------------------------------------

    temporary_point_col = False

    if (
        point_col is None
        or point_col not in work_df.columns
        or work_df[point_col].isna().all()
    ):

        internal_point_col = (
            "__analysis_point__"
        )

        work_df[internal_point_col] = (
            SINGLE_POINT_LABEL
        )

        temporary_point_col = True

    else:

        internal_point_col = (
            point_col
        )

    # --------------------------------------------------------
    # Agrupamento
    # --------------------------------------------------------
    #
    # Cada parâmetro precisa ser avaliado dentro do seu
    # próprio ponto.
    #
    # Nunca devemos calcular um IQR misturando, por exemplo:
    #
    #     P01 / Fósforo
    #     P02 / Fósforo
    #
    # ou:
    #
    #     P01 / Fósforo
    #     P01 / Turbidez
    #
    # --------------------------------------------------------

    grouped = work_df.groupby(
        [
            internal_point_col,
            parameter_col,
        ],
        dropna=False,
        sort=False,
    )

    # --------------------------------------------------------
    # Aplicação grupo a grupo
    # --------------------------------------------------------

    for _, group in grouped:

        flags = _flag_group_outliers(
            group=group,
            value_col=value_col,
        )

        work_df.loc[
            flags.index,
            OUTLIER_COLUMN,
        ] = flags

    # --------------------------------------------------------
    # Remoção da coluna temporária
    # --------------------------------------------------------

    if temporary_point_col:

        work_df.drop(
            columns=[
                internal_point_col,
            ],
            inplace=True,
        )

    return work_df
"""
Análise de correlação entre parâmetros de qualidade da água.

A análise utiliza Kendall Tau para avaliar associações monotônicas
entre pares de parâmetros.

Regras adotadas
---------------
1. Apenas resultados efetivamente quantificados participam da
   correlação.

2. Observações censuradas são excluídas da análise de correlação.
   Portanto, qualquer linha onde Qualifier não seja nulo é removida
   da cópia temporária utilizada neste módulo.

   Exemplos excluídos:
       <0.05
       >10

3. O DataFrame original não é modificado.

4. A coluna Outlier_IQR não interfere no cálculo. Observações
   sinalizadas como potenciais outliers continuam participando
   normalmente, desde que não sejam censuradas.

5. As observações são pareadas pela identidade da amostragem:
       Data + Ponto

6. A correlação é calculada separadamente para cada ponto.

7. Para cada par de parâmetros são utilizadas somente as datas
   em que ambos possuem resultado numérico simultaneamente.

8. Se não houver coluna de ponto, todo o dataset é tratado como
   pertencente a um único ponto lógico.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


SINGLE_POINT_LABEL = "__single_point__"

MIN_PAIRED_OBSERVATIONS = 4


def _metric(
    value,
    p_value,
    valid: bool,
    reason: str | None = None,
) -> dict:
    """
    Estrutura padronizada para o resultado de uma correlação.
    """

    return {
        "value": value,
        "p_value": p_value,
        "valid": bool(valid),
        "reason": reason,
    }


def _calculate_kendall(
    x: pd.Series,
    y: pd.Series,
) -> dict:
    """
    Calcula Kendall Tau para duas séries já pareadas.
    """

    n_pairs = len(x)

    # --------------------------------------------------------
    # Amostra insuficiente
    # --------------------------------------------------------

    if n_pairs < MIN_PAIRED_OBSERVATIONS:

        return _metric(
            value=None,
            p_value=None,
            valid=False,
            reason="insufficient_paired_observations",
        )

    # --------------------------------------------------------
    # Série constante
    # --------------------------------------------------------
    #
    # Kendall Tau não fornece uma associação útil quando
    # uma das variáveis não apresenta variação.
    # --------------------------------------------------------

    if x.nunique() < 2 or y.nunique() < 2:

        return _metric(
            value=None,
            p_value=None,
            valid=False,
            reason="constant_values",
        )

    # --------------------------------------------------------
    # Kendall Tau
    # --------------------------------------------------------

    result = kendalltau(
        x,
        y,
        nan_policy="omit",
    )

    tau = result.statistic
    p_value = result.pvalue

    # Segurança adicional contra resultados NaN.
    if pd.isna(tau) or pd.isna(p_value):

        return _metric(
            value=None,
            p_value=None,
            valid=False,
            reason="kendall_tau_not_computable",
        )

    return _metric(
        value=float(tau),
        p_value=float(p_value),
        valid=True,
        reason=None,
    )


def correlation_analysis(
    df: pd.DataFrame,
    date_col: str = "Data",
    point_col: str | None = "Ponto",
    parameter_col: str = "Parâmetro",
    value_col: str = "Resultado Num",
    qualifier_col: str = "Qualifier",
) -> list[dict]:
    """
    Calcula correlações de Kendall entre pares de parâmetros.

    Parameters
    ----------
    df:
        DataFrame normalizado pela pipeline.

    date_col:
        Coluna contendo a data da observação.

        Padrão:
            "Data"

    point_col:
        Coluna contendo o ponto de monitoramento.

        Padrão:
            "Ponto"

        Caso não exista, seja None ou esteja completamente vazia,
        todo o conjunto será tratado como um único ponto lógico.

    parameter_col:
        Coluna contendo o parâmetro.

        Padrão:
            "Parâmetro"

        Utilizamos o parâmetro analítico neste estágio para que
        parâmetros ainda não resolvidos pelo matching regulatório
        não sejam agrupados incorretamente.

    value_col:
        Coluna numérica utilizada na análise.

        Padrão:
            "Resultado Num"

    qualifier_col:
        Coluna contendo os qualificadores de censura.

        Padrão:
            "Qualifier"

    Returns
    -------
    list[dict]

        Um registro para cada combinação:

            Ponto × Parâmetro X × Parâmetro Y

        Exemplo:

        {
            "point": "P01",
            "point_inferred": False,
            "parameter_x": "OD",
            "parameter_y": "Turbidez",
            "n_pairs": 30,
            "kendall_tau": {
                "value": -0.45,
                "p_value": 0.002,
                "valid": True,
                "reason": None
            }
        }
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
    # CÓPIA TEMPORÁRIA
    # ========================================================

    work_df = df.copy()

    # ========================================================
    # REMOVE RESULTADOS CENSURADOS
    # ========================================================
    #
    # Qualquer qualifier significa que não conhecemos a
    # concentração efetivamente observada.
    #
    # Portanto:
    #
    #     <0.05
    #     >10
    #
    # não entram na correlação.
    #
    # O DataFrame original permanece intacto.
    # ========================================================

    if qualifier_col in work_df.columns:

        qualifier = (
            work_df[qualifier_col]
            .astype("string")
            .str.strip()
        )

        quantified_mask = (
            qualifier.isna()
            | qualifier.eq("")
        )

        work_df = work_df.loc[
            quantified_mask
        ].copy()

    # ========================================================
    # CONVERSÃO NUMÉRICA
    # ========================================================

    work_df[value_col] = pd.to_numeric(
        work_df[value_col],
        errors="coerce",
    )

    # Não há motivo para carregar valores sem resultado
    # numérico para a tabela wide.

    work_df = work_df.dropna(
        subset=[
            date_col,
            parameter_col,
            value_col,
        ]
    )

    # ========================================================
    # DIMENSÃO PONTO
    # ========================================================

    point_inferred = False

    if (
        point_col is None
        or point_col not in work_df.columns
        or work_df[point_col].isna().all()
    ):

        internal_point_col = "__analysis_point__"

        work_df[internal_point_col] = (
            SINGLE_POINT_LABEL
        )

        point_inferred = True

    else:

        internal_point_col = point_col

    # ========================================================
    # RESULTADOS
    # ========================================================

    results: list[dict] = []

    # ========================================================
    # PROCESSAMENTO POR PONTO
    # ========================================================

    for point, point_df in work_df.groupby(
        internal_point_col,
        dropna=False,
        sort=False,
    ):

        # ====================================================
        # LONG -> WIDE
        # ====================================================
        #
        # Antes:
        #
        # Data       Parâmetro      Resultado
        # 01/01      OD             6.2
        # 01/01      Turbidez       12.5
        # 02/01      OD             5.8
        # 02/01      Turbidez       18.2
        #
        # Depois:
        #
        # Data         OD     Turbidez
        # 01/01       6.2       12.5
        # 02/01       5.8       18.2
        #
        # Não incluímos Outlier_IQR nem quaisquer outras
        # colunas na identidade da observação.
        # ====================================================

        wide = point_df.pivot_table(
            index=date_col,
            columns=parameter_col,
            values=value_col,
            aggfunc="first",
        )

        parameters = list(
            wide.columns
        )

        # ====================================================
        # COMBINAÇÕES DE PARÂMETROS
        # ====================================================

        for parameter_x, parameter_y in combinations(
            parameters,
            2,
        ):

            # ================================================
            # PAREAMENTO
            # ================================================
            #
            # Aqui ocorre o ponto central da estratégia.
            #
            # Não fazemos dropna() no DataFrame wide inteiro.
            # Apenas selecionamos as duas variáveis que estão
            # sendo comparadas e removemos linhas onde uma
            # delas esteja ausente.
            # ================================================

            paired = (
                wide[
                    [
                        parameter_x,
                        parameter_y,
                    ]
                ]
                .dropna()
            )

            n_pairs = len(paired)

            x = paired[
                parameter_x
            ]

            y = paired[
                parameter_y
            ]

            # ================================================
            # KENDALL
            # ================================================

            kendall_result = _calculate_kendall(
                x=x,
                y=y,
            )

            # ================================================
            # RESULTADO
            # ================================================

            results.append(
                {
                    "point": point,
                    "point_inferred": point_inferred,
                    "parameter_x": parameter_x,
                    "parameter_y": parameter_y,
                    "n_pairs": int(n_pairs),
                    "kendall_tau": kendall_result,
                }
            )

    return results
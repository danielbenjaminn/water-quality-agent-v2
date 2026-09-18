from typing import Tuple

import pandas as pd
import numpy as np


def padronizar_unidades(s: pd.Series) -> pd.Series:
    """
    Padroniza a representação textual das unidades de medida.

    Esta função NÃO converte valores numéricos.
    Ela apenas normaliza diferentes grafias da mesma unidade.

    Exemplos
    --------
    mg/l        -> mg/L
    mg / L      -> mg/L
    μg/L        -> µg/L
    ug/L        -> µg/L
    µg / l      -> µg/L
    ºC          -> °C
    Col/100mL   -> UFC/100mL
    NOUNIT      -> -

    Parameters
    ----------
    s : pd.Series
        Série contendo as unidades originais.

    Returns
    -------
    pd.Series
        Série com as unidades textualmente padronizadas.
    """

    subs_unidades = {
        # -----------------------------------------------------
        # Símbolo micro
        # μ (grego) -> µ (micro sign)
        # -----------------------------------------------------
        r"μ": "µ",

        # -----------------------------------------------------
        # Concentração massa / volume
        # -----------------------------------------------------
        r"(?i)^mg\s*[./]?\s*l(?:itro)?(?:\^-?1|-1)?$": "mg/L",

        r"(?i)^(?:ug|µg)\s*[./]?\s*l(?:itro)?(?:\^-?1|-1)?$": "µg/L",

        r"(?i)^g\s*[./]?\s*l(?:itro)?(?:\^-?1|-1)?$": "g/L",

        # -----------------------------------------------------
        # Concentração massa / massa
        # -----------------------------------------------------
        r"(?i)^mg\s*[./]?\s*kg(?:\^-?1|-1)?$": "mg/kg",

        r"(?i)^(?:ug|µg)\s*[./]?\s*kg(?:\^-?1|-1)?$": "µg/kg",

        # -----------------------------------------------------
        # Microbiologia
        # -----------------------------------------------------
        r"(?i)^col\s*/\s*100\s*ml$": "UFC/100mL",

        r"(?i)^ufc\s*/\s*100\s*ml$": "UFC/100mL",

        r"(?i)^nmp\s*/\s*100\s*ml$": "NMP/100mL",

        r"(?i)^c[ée]l(?:ulas?)?\.?\s*/\s*ml$": "cel/mL",

        # -----------------------------------------------------
        # Condutividade
        # -----------------------------------------------------
        r"(?i)^(?:us|µs)\s*/\s*cm$": "µS/cm",

        # -----------------------------------------------------
        # Temperatura
        # -----------------------------------------------------
        r"(?i)^[º°]\s*c$": "°C",

        # -----------------------------------------------------
        # Volume / volume
        # -----------------------------------------------------
        r"(?i)^ml\s*/\s*l.*$": "mL/L",

        # -----------------------------------------------------
        # Sem unidade / presença-ausência
        # -----------------------------------------------------
        r"(?i)^nounit$": "-",

        r"(?i)^ausente$": "P/A",
    }

    # Mantém NA como NA e permite operações .str.
    result = s.astype("string").str.strip()

    # Remove espaços duplicados.
    result = result.str.replace(
        r"\s+",
        " ",
        regex=True,
    )

    # Primeiro unifica o caractere Unicode de micro.
    result = result.str.replace(
        "μ",
        "µ",
        regex=False,
    )

    # Aplica as regras de normalização.
    result = result.replace(
        to_replace=subs_unidades,
        regex=True,
    )

    return result


def contar_casas_decimais(valor):
    """Conta o número de casas decimais de um valor numérico."""
    valor_str = f"{valor:.16f}".rstrip('0')

    if '.' in valor_str:
        return len(valor_str.split('.')[1])

    return 0


def formatar_multiplicacao(row):
    """Aplica a multiplicação com lógica condicional baseada no valor de fc."""

    # Conversão não conhecida
    if pd.isna(row['fc']):
        return np.nan

    # Resultado inexistente
    if pd.isna(row['Resultado Num']):
        return np.nan

    # Multiplicação direta para fc >= 1
    if row['fc'] >= 1:
        return row['Resultado Num'] * row['fc']

    # Para fc < 1, preserva as casas decimais
    resultado = row['Resultado Num'] * row['fc']

    casas_decimais_total = (
        contar_casas_decimais(row['Resultado Num'])
        + contar_casas_decimais(row['fc'])
    )

    return f"{resultado:.{casas_decimais_total}f}"


def convert_units(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Recebe um DataFrame em processo de tratamento e adiciona
    as informações referentes à conversão de unidades.

    Conversões conhecidas recebem seu respectivo fator.

    Quando Unidade Original == Unidade, o fator é 1.

    Quando as unidades são diferentes e não existe conversão
    mapeada, o fator permanece NaN. Dessa forma, uma conversão
    desconhecida não é tratada silenciosamente como equivalência.

    Args:
        df (pd.DataFrame): DataFrame inicial.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - DataFrame após o processo de conversão;
            - Tabela de conversões mapeadas utilizada.
    """

    # Mapear conversões:
    # DE | PARA | FATOR DE CONVERSÃO
    mConversao = np.array(
        [
            ('mg/L', 'µg/L', 1000),
            ('mS/cm', 'µS/cm', 1000),
            ('µg/L', 'mg/L', 0.001),
            ('µS/cm', 'mS/cm', 0.001),
            ('NTU', 'UNT', 1),
            ('CU', 'mg Pt/L', 1),
            ('UFC/100mL', 'NMP/100mL', 1),
        ],
        dtype=object
    )

    # Criar DataFrame a partir das conversões mapeadas
    tConversao = pd.DataFrame.from_records(
        mConversao,
        columns=['De', 'Para', 'fc'],
        coerce_float=True
    )

    # Mesclar fator de conversão à tabela principal
    df_merged = pd.merge(
        left=df,
        left_on=['Unidade Original', 'Unidade'],
        right=tConversao,
        right_on=['De', 'Para'],
        how='left'
    )

    # Fator 1 SOMENTE quando a unidade original
    # já é igual à unidade esperada.
    mesma_unidade = (
        df_merged['Unidade Original']
        == df_merged['Unidade']
    )

    df_merged.loc[mesma_unidade, 'fc'] = 1

    # Resultado formatado para apresentação
    df_merged['Resultado_string'] = df_merged.apply(
        formatar_multiplicacao,
        axis=1
    )

    df_merged['Resultado_string'] = (
        df_merged['Resultado_string']
        .astype('string')
        .str.replace('.', ',', regex=False)
    )

    # Resultado numérico convertido.
    # Quando fc é NaN, Resultado também será NaN.
    df_merged['Resultado'] = (
        df_merged['Resultado Num']
        * df_merged['fc']
    )

    # Remove colunas auxiliares do merge
    df_merged.drop(
        columns=['De', 'Para'],
        inplace=True
    )

    return df_merged, tConversao


def converted_units(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna as conversões efetivamente realizadas.

    São consideradas somente linhas em que a unidade original
    é diferente da unidade de destino e existe fator de conversão.

    Args:
        df (pd.DataFrame): DataFrame pós-conversão.

    Returns:
        pd.DataFrame: Conversões aplicadas.
    """

    check_units_cols = [
        'Parâmetro',
        'Parâmetro B.D',
        'Unidade Original',
        'Unidade',
        'fc'
    ]

    units = (
        df[
            (df['Unidade Original'] != df['Unidade'])
            & (df['fc'].notna())
        ]
        .reindex(columns=check_units_cols)
        .drop_duplicates()
    )

    return units


def unidades_nao_convertidas(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna conversões necessárias que não estão mapeadas.

    Uma linha é considerada não resolvida quando a unidade original
    difere da unidade esperada e não existe fator de conversão.

    Args:
        df (pd.DataFrame): DataFrame pós-conversão.

    Returns:
        pd.DataFrame: Conversões de unidade não resolvidas.
    """

    check_units_cols = [
        'Parâmetro',
        'Parâmetro B.D',
        'Unidade Original',
        'Unidade'
    ]

    units = (
        df[
            (df['Unidade Original'] != df['Unidade'])
            & (df['fc'].isna())
        ]
        .reindex(columns=check_units_cols)
        .drop_duplicates()
    )

    return units
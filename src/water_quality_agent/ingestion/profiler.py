from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _sample_values(
    series: pd.Series,
    sample_fraction: float = 0.01,
    min_examples: int = 6,
    max_examples: int = 30,
) -> list[str]:
    """
    Gera exemplos representativos dos valores de uma coluna para
    auxiliar a inferência semântica do schema pelo LLM.

    Estratégia
    ----------
    1. Valores nulos são ignorados.
    2. Se a coluna possuir baixa cardinalidade (número de valores
       únicos <= max_examples), todos os valores únicos são enviados.
    3. Para colunas de alta cardinalidade, é utilizada uma amostra
       proporcional dos registros não nulos.
    4. A amostra é limitada por min_examples e max_examples.
    5. Os registros são selecionados em posições distribuídas ao
       longo da coluna para evitar viés causado pela ordenação dos dados.
    6. Duplicatas são removidas após a amostragem.

    Parameters
    ----------
    series:
        Série original.

    sample_fraction:
        Fração dos registros não nulos usada para determinar
        o tamanho da amostra em colunas de alta cardinalidade.

    min_examples:
        Quantidade mínima desejada de exemplos.

    max_examples:
        Quantidade máxima de exemplos enviada ao LLM.

    Returns
    -------
    list[str]
        Exemplos convertidos para string.
    """

    valid = series.dropna()

    if valid.empty:
        return []

    unique = valid.drop_duplicates()
    unique_count = len(unique)

    # ---------------------------------------------------------
    # Baixa cardinalidade
    # ---------------------------------------------------------
    # Se conseguimos mostrar todos os valores distintos sem
    # ultrapassar o limite de contexto, isso fornece ao LLM
    # uma representação completa da coluna.
    # ---------------------------------------------------------

    if unique_count <= max_examples:
        return [str(value) for value in unique.tolist()]

    # ---------------------------------------------------------
    # Alta cardinalidade
    # ---------------------------------------------------------

    sample_size = int(np.ceil(len(valid) * sample_fraction))

    sample_size = max(min_examples, sample_size)
    sample_size = min(max_examples, sample_size)
    sample_size = min(sample_size, len(valid))

    # Seleciona posições distribuídas por toda a coluna.
    #
    # Isso é preferível a head(), porque bases ambientais podem
    # estar ordenadas por data, ponto, parâmetro ou campanha.
    positions = np.linspace(
        0,
        len(valid) - 1,
        num=sample_size,
        dtype=int,
    )

    sampled = valid.iloc[positions]

    # A mesma informação pode aparecer várias vezes na amostra.
    # Para o LLM, repetir valores não acrescenta informação.
    sampled = sampled.drop_duplicates()

    return [str(value) for value in sampled.tolist()]


def profile_dataframe(
    df: pd.DataFrame,
    sample_fraction: float = 0.01,
    min_examples: int = 6,
    max_examples: int = 30,
) -> dict[str, Any]:
    """
    Cria um perfil compacto do DataFrame para inferência semântica
    das colunas.

    O perfil não envia o dataset completo ao LLM. Para cada coluna
    são produzidas informações estruturais e uma pequena amostra
    representativa dos valores.

    Parameters
    ----------
    df:
        DataFrame original.

    sample_fraction:
        Fração utilizada para calcular a quantidade de exemplos
        em colunas de alta cardinalidade.

    min_examples:
        Quantidade mínima desejada de exemplos.

    max_examples:
        Quantidade máxima de exemplos por coluna.

    Returns
    -------
    dict[str, Any]
        Perfil estrutural do dataset.
    """

    columns: list[dict[str, Any]] = []

    for column in df.columns:
        series = df[column]

        non_null = series.dropna()

        non_null_count = int(non_null.size)
        unique_count = int(non_null.nunique())

        null_count = int(series.isna().sum())

        columns.append(
            {
                "name": str(column),
                "dtype": str(series.dtype),

                # Quantidade absoluta ajuda a interpretar bases
                # pequenas, onde uma porcentagem pode enganar.
                "null_count": null_count,

                "null_pct": round(
                    float(series.isna().mean() * 100),
                    2,
                ),

                "non_null_count": non_null_count,

                # Cardinalidade é uma pista semântica importante.
                #
                # Ex.:
                # unidade   -> geralmente baixa cardinalidade
                # parâmetro -> baixa/média cardinalidade
                # resultado -> geralmente alta cardinalidade
                "unique_count": unique_count,

                "examples": _sample_values(
                    series,
                    sample_fraction=sample_fraction,
                    min_examples=min_examples,
                    max_examples=max_examples,
                ),
            }
        )

    return {
        "rows": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": columns,
    }
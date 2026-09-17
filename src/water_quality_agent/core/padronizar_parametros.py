import re
from collections import Counter

import numpy as np
from unidecode import unidecode
from fuzzywuzzy import fuzz


def trat_string(x: str) -> str:
    """Aplica tratamentos de limpeza de texto sobre uma string.

    O texto é retornado sem acentos, em maiúsculas, sem espaços
    adicionais e sem caracteres especiais.

    Args:
        x (str): Sequência de texto de entrada.

    Returns:
        str: Sequência de texto após o tratamento.
    """
    x_trat = unidecode(x.upper().replace('-', ' '))

    return re.sub(
        r'\n',
        '',
        re.sub(
            r'\s{2,}',
            ' ',
            re.sub(r'[^\w\s]', '', x_trat)
        )
    ).strip()


# Implementação da função Jaro
def jaro_distance(s1, s2):
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)

    if len1 == 0 or len2 == 0:
        return 0.0

    max_dist = max(0, (max(len1, len2) // 2) - 1)

    match = 0

    hash_s1 = [0] * len1
    hash_s2 = [0] * len2

    # Verifica os caracteres correspondentes
    for i in range(len1):
        for j in range(
            max(0, i - max_dist),
            min(len2, i + max_dist + 1)
        ):
            if s1[i] == s2[j] and not hash_s2[j]:
                hash_s1[i] = 1
                hash_s2[j] = 1
                match += 1
                break

    if match == 0:
        return 0.0

    # Conta as transposições
    t = 0
    point = 0

    for i in range(len1):
        if hash_s1[i]:
            while point < len2 and not hash_s2[point]:
                point += 1

            if point < len2 and s1[i] != s2[point]:
                t += 1

            point += 1

    t /= 2

    # Calcula a distância de Jaro
    jaro = (
        match / len1
        + match / len2
        + (match - t) / match
    ) / 3

    return jaro


def jaro_winkler_distance(s1, s2, prefix_scale=0.1):
    jaro_dist = jaro_distance(s1, s2)

    # Calcula o comprimento do prefixo comum
    # até o máximo de 4 caracteres
    prefix_length = 0

    for i in range(min(len(s1), len(s2))):
        if s1[i] == s2[i]:
            prefix_length += 1
        else:
            break

    prefix_length = min(4, prefix_length)

    # Calcula a distância de Jaro-Winkler
    jaro_winkler = (
        jaro_dist
        + (
            prefix_length
            * prefix_scale
            * (1 - jaro_dist)
        )
    )

    return jaro_winkler


def score_function(string1, string2):
    """Calcula o score combinado entre WRatio e Jaro-Winkler."""

    w_fuzzy = 1
    w_lev = 100

    ratio = (
        w_fuzzy * fuzz.WRatio(string1, string2)
        + w_lev * jaro_winkler_distance(string1, string2)
    )

    return np.round(
        ratio / (100 * w_fuzzy + w_lev),
        3
    )


def get_similar(param, param_dict, n=5):
    """Retorna os parâmetros mais similares e seus respectivos scores."""

    scores = {
        p: score_function(param, p_trat)
        for p_trat, p in param_dict.items()
    }

    c = Counter(scores)

    return c.most_common(n)


def padronizar_param(
    p,
    param_dict,
    regras_parametros,
    limiar_similaridade=0.85,
    margem_ambiguidade=0.05
):
    """Padroniza o nome de um parâmetro.

    Ordem de resolução:
        1. Regras de parâmetros/regex;
        2. Normalização do nome recebido;
        3. Correspondência exata com Parâmetro B.D. normalizado;
        4. Similaridade com Parâmetro B.D. normalizado.

    Parâmetros com baixa similaridade ou correspondência ambígua
    ficam pendentes para avaliação por LLM.

    Args:
        p:
            Nome original do parâmetro recebido do dataset.

        param_dict:
            Dicionário em que as chaves são os valores normalizados
            de "Parâmetro B.D." de limites_legais.csv e os valores
            são os respectivos nomes originais.

        regras_parametros:
            Lista de tuplas (regex, parametro_padronizado).

        limiar_similaridade:
            Score mínimo para aceitar a correspondência fuzzy.

        margem_ambiguidade:
            Diferença mínima entre os dois melhores resultados
            para aceitar automaticamente o primeiro.

    Returns:
        tuple:
            (
                parametro_padronizado,
                score_similaridade,
                metodo_resolucao
            )
    """

    # Garante uma entrada textual válida
    if p is None:
        return (
            np.nan,
            np.nan,
            'Pendente LLM - parâmetro vazio'
        )

    p = str(p).strip()

    if not p:
        return (
            np.nan,
            np.nan,
            'Pendente LLM - parâmetro vazio'
        )

    # 1. Tenta primeiro as regras regex sobre o parâmetro original
    for regex, parametro_padronizado in regras_parametros:
        m = re.search(regex, p, re.IGNORECASE)

        if m:
            return (
                m.expand(parametro_padronizado),
                np.nan,
                'Encontrado em regras_parametros'
            )

    # 2. Normaliza o parâmetro apenas para comparação
    p_normalizado = trat_string(p)

    # 3. Verifica correspondência exata após a normalização
    if p_normalizado in param_dict:
        return (
            param_dict[p_normalizado],
            1.0,
            'Encontrado após normalização'
        )

    # 4. Busca os dois parâmetros mais similares
    similares = get_similar(
        param=p_normalizado,
        param_dict=param_dict,
        n=2
    )

    # Nenhum candidato disponível
    if not similares:
        return (
            np.nan,
            np.nan,
            'Pendente LLM - nenhum candidato'
        )

    melhor_parametro, melhor_score = similares[0]

    # Similaridade abaixo do limite
    if melhor_score < limiar_similaridade:
        return (
            np.nan,
            melhor_score,
            'Pendente LLM - baixa similaridade'
        )

    # Verifica ambiguidade entre os dois melhores candidatos
    if len(similares) > 1:
        segundo_score = similares[1][1]

        if (melhor_score - segundo_score) < margem_ambiguidade:
            return (
                np.nan,
                melhor_score,
                'Pendente LLM - similaridade ambígua'
            )

    # Correspondência aceita
    return (
        melhor_parametro,
        melhor_score,
        'Estimado por similaridade'
    )
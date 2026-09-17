from __future__ import annotations
import json
import re
import unicodedata
import pandas as pd

from .models import ALL_FIELDS, FIELD_DESCRIPTIONS, SemanticMapping
from .profiler import profile_dataframe
from water_quality_agent.llm.structured_output import invoke_structured

ALIASES = {
    "parameter": {"parametro", "analito", "ensaio", "determinacao", "variavel", "parameter", "analyte"},
    "result": {"resultado", "valor", "valor medido", "resultado analitico", "concentracao", "result", "value", "measurement"},
    "unit": {"unidade", "un", "unid", "unidade de medida", "unit", "units"},
    "date": {"data", "data coleta", "data de coleta", "data amostragem", "data de amostragem", "date", "sampling date", "collection date"},
    "point": {"ponto", "estacao", "estacao de monitoramento", "ponto coleta", "ponto de coleta", "station", "site", "point", "local"},
    "qualifier": {"qualifier", "qualificador", "sinal", "censor", "censura", "flag"},
    "report_id": {"laudo", "numero laudo", "numero do laudo", "n laudo", "id laudo", "report id", "report number"},
    "campaign": {"campanha", "campanha amostragem", "campanha de amostragem", "campaign", "sampling campaign"},
    "water_class": {"classe", "classe agua", "classe da agua", "enquadramento", "water class"},
    "matrix": {"matriz", "matriz amostra", "matriz da amostra", "matrix", "sample matrix"},
    "sub_basin": {"sub bacia", "subbacia", "sub basin", "subbasin"},
    "basin": {"bacia", "bacia hidrografica", "basin", "watershed"},
}


def _norm(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value).strip().lower())
    value = "".join(c for c in value if not unicodedata.combining(c))
    value = re.sub(r"[_\-.]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def deterministic_mapping(df: pd.DataFrame) -> dict[str, str]:
    mapped: dict[str, str] = {}
    used_columns: set[str] = set()
    normalized_aliases = {field: {_norm(a) for a in aliases} for field, aliases in ALIASES.items()}
    for column in df.columns:
        normalized = _norm(column)
        hits = [field for field, aliases in normalized_aliases.items() if normalized in aliases]
        if len(hits) == 1 and hits[0] not in mapped and str(column) not in used_columns:
            mapped[hits[0]] = str(column)
            used_columns.add(str(column))
    return mapped


def _sanitize_mapping(mapping: dict[str, str | None], columns: list[str], locked: dict[str, str]) -> dict[str, str | None]:
    """Aceita somente campos conhecidos, colunas reais e mapeamento 1:1.

    Mapeamentos determinísticos (`locked`) têm precedência sobre sugestões do LLM.
    """
    result = {field: None for field in ALL_FIELDS}
    used: set[str] = set()
    for field, column in locked.items():
        result[field] = column
        used.add(column)
    for field in ALL_FIELDS:
        if field in locked:
            continue
        column = mapping.get(field)
        if column is None or column not in columns or column in used:
            continue
        result[field] = column
        used.add(column)
    return result


def semantic_mapping(df: pd.DataFrame, llm=None) -> SemanticMapping:
    deterministic = deterministic_mapping(df)
    columns = [str(c) for c in df.columns]
    if llm is None or len(deterministic) == len(ALL_FIELDS):
        mapping = _sanitize_mapping({}, columns, deterministic)
        return SemanticMapping(mapping=mapping, confidence={k: 1.0 for k in deterministic})

    unresolved = [field for field in ALL_FIELDS if field not in deterministic]
    profile = profile_dataframe(df)
    prompt = f"""Você resolve semanticamente schemas de datasets de monitoramento de qualidade da água.

O dataset pode ter nomes de colunas completamente diferentes. Use em conjunto: nome da coluna, dtype, exemplos de valores e relação entre as colunas.

Mapeie SOMENTE os campos do domínio abaixo:
{json.dumps(FIELD_DESCRIPTIONS, ensure_ascii=False, indent=2)}

Regras obrigatórias:
- Nunca invente nomes de colunas: use exatamente um nome presente no perfil.
- Uma coluna original não pode representar dois campos semânticos.
- Se não houver evidência suficiente, retorne null. É melhor UNKNOWN do que um palpite.
- Não mapeie latitude, longitude, coordenadas, geolocalização, CRS ou outras colunas fora do domínio acima.
- Não force toda coluna a ter significado para o sistema.
- Os campos já resolvidos deterministicamente não devem ser alterados.

Já resolvidos: {json.dumps(deterministic, ensure_ascii=False)}
Campos ainda possíveis: {json.dumps(unresolved, ensure_ascii=False)}
Perfil do dataset: {json.dumps(profile, ensure_ascii=False)}

Retorne `mapping`, `confidence` entre 0 e 1 e `notes`. Inclua todos os campos semânticos em mapping, usando null quando desconhecido."""
    inferred = invoke_structured(
        llm=llm,
        prompt=prompt,
        schema=SemanticMapping,
    )
    clean = _sanitize_mapping(inferred.mapping, columns, deterministic)
    confidence = {field: float(inferred.confidence.get(field, 0.0)) for field in ALL_FIELDS if clean.get(field)}
    confidence.update({field: 1.0 for field in deterministic})
    return SemanticMapping(mapping=clean, confidence=confidence, notes=inferred.notes)

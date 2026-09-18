from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from water_quality_agent.llm.structured_output import invoke_structured

from .csv_loader import carregar_csv
from .domain import (
    CONFIG,
    load_legal_limits,
    load_parameter_catalog,
    resolve_parameter,
)
from .padronizar_parametros import trat_string


# ============================================================
# MODELOS
# ============================================================

class LLMParameterSuggestion(BaseModel):
    original: str

    suggested_canonical: str | None = None
    suggested_unit: str | None = None

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    reason: str


class ParameterResolution(BaseModel):
    original: str

    canonical: str | None = None
    unit: str | None = None

    status: str
    source: str

    score: float | None = None

    llm_suggestion: LLMParameterSuggestion | None = None


# ============================================================
# VOCABULÁRIO CONHECIDO
# ============================================================

def known_parameters() -> list[dict]:
    """
    Retorna parâmetros canônicos conhecidos pelas bases locais.

    Fontes:
    - limites_legais.csv
    - parametros_catalogo.csv
    """

    records: dict[str, dict] = {}

    # --------------------------------------------------------
    # Legislação
    # --------------------------------------------------------

    limits = load_legal_limits()

    if "Parâmetro B.D." in limits.columns:

        for _, row in limits.iterrows():

            canonical = row.get("Parâmetro B.D.")

            if pd.isna(canonical):
                continue

            canonical = str(canonical).strip()

            unit = row.get("Unidade")

            if pd.isna(unit):
                unit = None

            key = trat_string(canonical)

            records[key] = {
                "canonical": canonical,
                "unit": unit,
                "source": "legal",
            }

    # --------------------------------------------------------
    # Catálogo complementar
    # --------------------------------------------------------

    catalog = load_parameter_catalog()

    if not catalog.empty:

        for _, row in catalog.iterrows():

            canonical = row.get("Parâmetro B.D.")

            if pd.isna(canonical):
                continue

            canonical = str(canonical).strip()

            unit = row.get("Unidade")

            if pd.isna(unit):
                unit = None

            key = trat_string(canonical)

            # legislação tem prioridade
            records.setdefault(
                key,
                {
                    "canonical": canonical,
                    "unit": unit,
                    "source": "catalog",
                },
            )

    return list(records.values())


# ============================================================
# RESOLUÇÃO DETERMINÍSTICA
# ============================================================

def resolve_deterministically(
    parameter: str,
) -> ParameterResolution:
    """
    Tenta resolver o parâmetro usando apenas os mecanismos
    determinísticos já existentes.

    Valores ausentes retornados pelo resolver (NaN, None ou
    string vazia) são tratados como não resolvidos.
    """

    result = resolve_parameter(parameter)

    canonical = result.get("canonical")
    score = result.get("score")
    method = result.get("method")

    # =========================================================
    # NORMALIZAÇÃO DO RETORNO
    # =========================================================

    if pd.isna(canonical):
        canonical = None

    elif isinstance(canonical, str):
        canonical = canonical.strip()

        if not canonical:
            canonical = None

    if pd.isna(score):
        score = None

    # =========================================================
    # RESOLVIDO
    # =========================================================

    if canonical is not None:

        return ParameterResolution(
            original=parameter,
            canonical=str(canonical),
            unit=None,
            status="resolved",
            source=f"deterministic:{method}",
            score=(
                float(score)
                if score is not None
                else None
            ),
        )

    # =========================================================
    # NÃO RESOLVIDO
    # =========================================================

    return ParameterResolution(
        original=parameter,
        canonical=None,
        unit=None,
        status="unresolved",
        source="deterministic",
        score=(
            float(score)
            if score is not None
            else None
        ),
    )


# ============================================================
# PERSISTÊNCIA DO HITL
# ============================================================

def persist_parameter_mapping(
    original: str,
    canonical: str,
    unit: str | None,
) -> None:
    """
    Persiste uma decisão humana em parametros_catalogo.csv.

    Essa função só deve ser chamada DEPOIS da revisão humana.
    """

    path = CONFIG / "parametros_catalogo.csv"

    if path.exists():
        catalog = carregar_csv(path).dataframe.copy()
    else:
        catalog = pd.DataFrame(
            columns=[
                "Parâmetro",
                "Parâmetro B.D.",
                "Unidade",
            ]
        )

    new_row = pd.DataFrame(
        [
            {
                "Parâmetro": original,
                "Parâmetro B.D.": canonical,
                "Unidade": unit,
            }
        ]
    )

    # --------------------------------------------------------
    # Evita duplicar exatamente o mesmo alias
    # --------------------------------------------------------

    if not catalog.empty:

        existing = (
            catalog["Parâmetro"]
            .astype(str)
            .map(trat_string)
            .eq(trat_string(original))
        )

        if existing.any():

            catalog.loc[
                existing,
                "Parâmetro B.D.",
            ] = canonical

            catalog.loc[
                existing,
                "Unidade",
            ] = unit

        else:
            catalog = pd.concat(
                [catalog, new_row],
                ignore_index=True,
            )

    else:
        catalog = new_row

    catalog.to_csv(
        path,
        sep=";",
        encoding="utf-8",
        index=False,
    )

# ============================================================
# FALLBACK LLM
# ============================================================

def suggest_parameter_with_llm(
    parameter: str,
    observed_units: list[str],
    llm,
) -> LLMParameterSuggestion:
    """
    Usa a LLM somente como fallback para parâmetros que
    não puderam ser resolvidos deterministicamente.

    A sugestão NÃO é aplicada automaticamente.
    Ela deverá ser validada posteriormente via HITL.
    """

    known = known_parameters()

    vocabulary = "\n".join(
        (
            f"- {item['canonical']}"
            + (
                f" | unidade: {item['unit']}"
                if item["unit"]
                else ""
            )
        )
        for item in known
    )

    units_text = (
        ", ".join(observed_units)
        if observed_units
        else "não informada"
    )

    prompt = f"""
Você está auxiliando na harmonização de parâmetros de
qualidade da água.

Parâmetro recebido:
{parameter}

Unidades observadas no dataset:
{units_text}

Parâmetros canônicos conhecidos:
{vocabulary}

Sua tarefa é verificar se o parâmetro recebido corresponde
semanticamente a algum parâmetro canônico conhecido.

Regras:

1. Não invente correspondências.

2. Preserve diferenças semanticamente importantes, como:
   - total vs dissolvido;
   - nitrato vs nitrito;
   - nitrogênio vs nitrato;
   - fósforo total vs fosfato;
   - formas químicas diferentes.

3. Se houver correspondência plausível, use EXATAMENTE o
   nome canônico presente na lista fornecida.

4. Se não houver correspondência suficientemente segura,
   retorne:
       suggested_canonical = null

5. suggested_unit deve ser a unidade esperada para o
   parâmetro sugerido, quando conhecida.

6. confidence deve estar entre 0 e 1.

7. reason deve explicar brevemente a sugestão.

IMPORTANTE:
Sua resposta é apenas uma sugestão.
Um humano decidirá posteriormente se ela será aceita.
"""

    return invoke_structured(
        llm=llm,
        prompt=prompt,
        schema=LLMParameterSuggestion,
    )


# ============================================================
# RESOLUÇÃO DOS PARÂMETROS DO DATASET
# ============================================================

def resolve_dataset_parameters(
    df: pd.DataFrame,
    llm=None,
) -> tuple[
    dict[str, ParameterResolution],
    list[ParameterResolution],
]:
    """
    Resolve cada parâmetro único do dataset apenas uma vez.

    Fluxo:
        1. resolução determinística;
        2. se não resolver, fallback LLM;
        3. sugestão da LLM NÃO é aplicada;
        4. parâmetro entra em pending para HITL.

    Returns
    -------
    resolutions:
        Dicionário:
            parâmetro_original -> ParameterResolution

    pending:
        Lista dos parâmetros que precisam de revisão humana.
    """

    resolutions: dict[str, ParameterResolution] = {}
    pending: list[ParameterResolution] = []

    # --------------------------------------------------------
    # Trabalha somente com nomes únicos
    # --------------------------------------------------------

    parameters = (
        df["parameter"]
        .dropna()
        .astype(str)
        .unique()
    )

    for parameter in parameters:

        # ====================================================
        # 1. RESOLUÇÃO DETERMINÍSTICA
        # ====================================================

        resolution = resolve_deterministically(
            parameter
        )

        if resolution.status == "resolved":

            resolutions[parameter] = resolution
            continue

        # ====================================================
        # 2. NÃO RESOLVEU E NÃO TEMOS LLM
        # ====================================================

        if llm is None:

            resolutions[parameter] = resolution
            pending.append(resolution)

            continue

        # ====================================================
        # 3. UNIDADES OBSERVADAS PARA O PARÂMETRO
        # ====================================================

        mask = (
            df["parameter"]
            .astype(str)
            .eq(parameter)
        )

        observed_units = (
            df.loc[mask, "unit"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        # ====================================================
        # 4. FALLBACK LLM
        # ====================================================

        suggestion = suggest_parameter_with_llm(
            parameter=parameter,
            observed_units=observed_units,
            llm=llm,
        )

        # ====================================================
        # 5. GERA PENDÊNCIA HITL
        #
        # Não colocamos:
        #
        # canonical=suggestion.suggested_canonical
        #
        # porque isso aplicaria a decisão da LLM antes da
        # validação humana.
        # ====================================================

        resolution = ParameterResolution(
            original=parameter,
            canonical=None,
            unit=None,
            status="needs_review",
            source="llm",
            score=suggestion.confidence,
            llm_suggestion=suggestion,
        )

        resolutions[parameter] = resolution
        pending.append(resolution)

    return resolutions, pending

# ============================================================
# APLICAÇÃO DE DECISÃO HUMANA
# ============================================================

def apply_human_decision(
    original: str,
    canonical: str,
    unit: str | None,
) -> ParameterResolution:
    """
    Registra uma decisão humana e devolve a resolução final.

    Pode representar:

    - aceitação da sugestão da LLM;
    - correção da sugestão;
    - criação de parâmetro novo.
    """

    persist_parameter_mapping(
        original=original,
        canonical=canonical,
        unit=unit,
    )

    return ParameterResolution(
        original=original,
        canonical=canonical,
        unit=unit,
        status="resolved",
        source="human",
        score=1.0,
    )
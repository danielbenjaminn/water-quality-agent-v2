from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)


def _extract_json(text: str) -> dict:
    """
    Extrai o primeiro objeto JSON válido encontrado no texto.

    Tolera:
    - texto antes/depois do JSON;
    - markdown ```json ... ```;
    - conteúdo adicional após o primeiro objeto JSON.
    """

    text = text.strip()

    # ---------------------------------------------------------
    # Remove fence Markdown, quando houver
    # ---------------------------------------------------------

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # ---------------------------------------------------------
    # Primeiro tenta interpretar a resposta inteira
    # ---------------------------------------------------------

    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    # ---------------------------------------------------------
    # Procura o primeiro objeto JSON válido
    #
    # raw_decode lê somente UM objeto JSON e ignora o que
    # vier depois dele.
    # ---------------------------------------------------------

    decoder = json.JSONDecoder()

    for index, char in enumerate(text):

        if char != "{":
            continue

        try:
            parsed, _ = decoder.raw_decode(
                text[index:]
            )

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            continue

    raise ValueError(
        "Não foi possível extrair um objeto JSON válido "
        "da resposta da LLM."
    )


def invoke_structured(
    llm,
    prompt: str,
    schema: type[T],
) -> T:
    """
    Executa uma chamada ao LLM e garante retorno validado
    pelo schema Pydantic.

    Estratégia:
    1. tenta structured output nativo via JSON Schema;
    2. se o provider/modelo não suportar, usa JSON textual;
    3. valida sempre o resultado final com Pydantic.
    """

    # -----------------------------------------------------
    # Estratégia 1: structured output nativo
    # -----------------------------------------------------

    try:
        structured_llm = llm.with_structured_output(
            schema,
            method="json_schema",
        )

        result = structured_llm.invoke(prompt)

        if isinstance(result, schema):
            return result

        return schema.model_validate(result)

    except Exception:
        pass

    # -----------------------------------------------------
    # Estratégia 2: JSON textual
    # -----------------------------------------------------

    schema_json = json.dumps(
        schema.model_json_schema(),
        ensure_ascii=False,
        indent=2,
    )

    fallback_prompt = f"""
{prompt}

IMPORTANTE:

Retorne APENAS um objeto JSON válido.

O JSON deve obedecer exatamente ao seguinte schema:

{schema_json}

Não utilize markdown.
Não utilize ```json.
Não escreva explicações antes ou depois do JSON.
"""

    response = llm.invoke(fallback_prompt)

    if hasattr(response, "content"):
        content = response.content
    else:
        content = str(response)

    parsed = _extract_json(content)

    try:
        return schema.model_validate(parsed)

    except ValidationError as exc:
        raise ValueError(
            "O LLM retornou JSON, mas ele não corresponde "
            "ao schema esperado."
        ) from exc
from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)


def _extract_json(text: str) -> dict:
    """
    Extrai um objeto JSON mesmo quando o modelo retorna
    ```json ... ```
    ou texto adicional.
    """

    text = text.strip()

    # Remove markdown fences.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: procura primeiro objeto JSON completo aparente.
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("O LLM não retornou um objeto JSON válido.")

    return json.loads(text[start:end + 1])


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
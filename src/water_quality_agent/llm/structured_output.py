from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)


# ============================================================
# EXTRAÇÃO DE JSON
# ============================================================

def _extract_json(text: str) -> dict:
    """
    Extrai o primeiro objeto JSON válido encontrado no texto.

    Tolera:
    - texto antes/depois do JSON;
    - markdown ```json ... ```;
    - conteúdo adicional após o primeiro objeto JSON.
    """

    text = text.strip()

    # --------------------------------------------------------
    # Remove fences Markdown simples
    # --------------------------------------------------------

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # --------------------------------------------------------
    # Primeiro tenta o conteúdo inteiro
    # --------------------------------------------------------

    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # Depois procura o primeiro objeto JSON válido
    # --------------------------------------------------------

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


# ============================================================
# CONTEÚDO DA RESPOSTA
# ============================================================

def _response_content(response) -> str:
    """
    Obtém o conteúdo textual retornado pelo modelo.
    """

    if hasattr(response, "content"):
        content = response.content
    else:
        content = response

    if isinstance(content, str):
        return content

    return str(content)


# ============================================================
# EXEMPLO DE ESTRUTURA ESPERADA
# ============================================================

def _expected_fields(schema: type[T]) -> str:
    """
    Produz uma descrição simples dos campos esperados sem
    entregar o JSON Schema inteiro ao modelo.
    """

    fields = []

    for name, field in schema.model_fields.items():

        required = field.is_required()

        fields.append(
            f"- {name}: {field.annotation}"
            + (" [obrigatório]" if required else " [opcional]")
        )

    return "\n".join(fields)


# ============================================================
# FALLBACK TEXTUAL
# ============================================================

def _invoke_textual_json(
    llm,
    prompt: str,
    schema: type[T],
) -> tuple[T | None, str, Exception | None]:
    """
    Solicita explicitamente uma INSTÂNCIA JSON do schema.

    Retorna:
        objeto validado ou None,
        conteúdo bruto,
        erro de validação/extracão ou None.
    """

    fields = _expected_fields(schema)

    fallback_prompt = f"""
{prompt}

FORMATO DA RESPOSTA

Retorne SOMENTE uma INSTÂNCIA JSON contendo os valores da resposta.

Não retorne JSON Schema.
Não descreva propriedades, tipos ou definições.
Não utilize as chaves "properties", "$defs", "required" ou "title"
para descrever o formato.

Campos que o objeto final deve possuir:

{fields}

Exemplo conceitual do formato esperado:

{{
  "campo_1": "valor",
  "campo_2": null
}}

Esse exemplo mostra apenas o FORMATO.
Use os nomes reais dos campos listados acima.

Não utilize markdown.
Não utilize ```json.
Não escreva qualquer texto antes ou depois do objeto JSON.
"""

    response = llm.invoke(fallback_prompt)

    content = _response_content(response)

    try:
        parsed = _extract_json(content)
        validated = schema.model_validate(parsed)

        return validated, content, None

    except (ValueError, ValidationError) as exc:
        return None, content, exc


# ============================================================
# REPARO
# ============================================================

def _repair_structured_response(
    llm,
    original_prompt: str,
    invalid_content: str,
    schema: type[T],
) -> T:
    """
    Faz uma única tentativa de reparo quando a resposta textual
    não corresponde ao schema esperado.
    """

    fields = _expected_fields(schema)

    repair_prompt = f"""
Você respondeu anteriormente a uma solicitação estruturada,
mas a resposta não pôde ser validada.

SOLICITAÇÃO ORIGINAL:

{original_prompt}

RESPOSTA INVÁLIDA:

{invalid_content}

Gere novamente a resposta.

Retorne SOMENTE uma INSTÂNCIA JSON com os valores solicitados.

Campos esperados:

{fields}

IMPORTANTE:

- não retorne JSON Schema;
- não retorne "properties";
- não retorne "$defs";
- não descreva os tipos;
- não utilize markdown;
- não escreva explicações fora do JSON;
- retorne exatamente um objeto JSON.
"""

    response = llm.invoke(repair_prompt)

    content = _response_content(response)

    parsed = _extract_json(content)

    try:
        return schema.model_validate(parsed)

    except ValidationError as exc:
        raise ValueError(
            "O LLM não retornou uma resposta compatível "
            "com o schema mesmo após uma tentativa de reparo."
        ) from exc


# ============================================================
# INTERFACE PRINCIPAL
# ============================================================

def invoke_structured(
    llm,
    prompt: str,
    schema: type[T],
) -> T:
    """
    Executa uma chamada ao LLM e garante retorno validado
    pelo schema Pydantic.

    Estratégia:
    1. tenta structured output nativo;
    2. se falhar, solicita JSON textual;
    3. valida com Pydantic;
    4. se inválido, faz uma única tentativa de reparo.
    """

    native_error: Exception | None = None

    # --------------------------------------------------------
    # 1. Structured output nativo
    # --------------------------------------------------------

    try:
        structured_llm = llm.with_structured_output(
            schema,
            method="json_schema",
        )

        result = structured_llm.invoke(prompt)

        if isinstance(result, schema):
            return result

        return schema.model_validate(result)

    except Exception as exc:
        native_error = exc

    # --------------------------------------------------------
    # 2. Fallback JSON textual
    # --------------------------------------------------------

    result, raw_content, fallback_error = _invoke_textual_json(
        llm=llm,
        prompt=prompt,
        schema=schema,
    )

    if result is not None:
        return result

    # --------------------------------------------------------
    # 3. Uma tentativa de reparo
    # --------------------------------------------------------

    try:
        return _repair_structured_response(
            llm=llm,
            original_prompt=prompt,
            invalid_content=raw_content,
            schema=schema,
        )

    except Exception as repair_error:

        raise ValueError(
            "Falha ao obter structured output válido.\n"
            f"Erro do structured output nativo: {native_error}\n"
            f"Erro do fallback textual: {fallback_error}\n"
            f"Erro do reparo: {repair_error}"
        ) from repair_error
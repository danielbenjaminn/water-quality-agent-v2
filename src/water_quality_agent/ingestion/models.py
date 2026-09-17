from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

SemanticField = Literal[
    "date", "parameter", "result", "unit", "point", "qualifier",
    "report_id", "campaign", "water_class", "matrix", "sub_basin", "basin",
]

REQUIRED_FIELDS = ("date", "parameter", "result", "unit")
OPTIONAL_FIELDS = (
    "point", "qualifier", "report_id", "campaign", "water_class",
    "matrix", "sub_basin", "basin",
)
ALL_FIELDS = REQUIRED_FIELDS + OPTIONAL_FIELDS

FIELD_DESCRIPTIONS = {
    "date": "Data de coleta/amostragem da observação.",
    "parameter": "Parâmetro/analito medido, por exemplo OD, pH, turbidez.",
    "result": "Resultado analítico medido, inclusive resultados censurados como <0,01.",
    "unit": "Unidade de medida associada ao resultado.",
    "point": "Ponto, estação ou local de monitoramento.",
    "qualifier": "Qualificador/sinal de censura quando armazenado em coluna separada.",
    "report_id": "Número ou identificador de laudo/relatório analítico.",
    "campaign": "Campanha, rodada ou evento de amostragem.",
    "water_class": "Classe/enquadramento do corpo hídrico.",
    "matrix": "Matriz da amostra, por exemplo água superficial, água bruta.",
    "sub_basin": "Sub-bacia hidrográfica.",
    "basin": "Bacia hidrográfica.",
}


class SemanticMapping(BaseModel):
    mapping: dict[str, str | None] = Field(
        description="campo semântico -> nome exato da coluna original; null se desconhecido"
    )
    confidence: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class ColumnInterpretation(BaseModel):
    column: str
    semantic_field: str | None = None
    meaning: str = "Não utilizada / significado não necessário ao domínio"
    confidence: float | None = None


class IngestionReport(BaseModel):
    valid: bool
    rows: int
    columns: int
    interpretations: list[ColumnInterpretation]
    required_found: list[str]
    required_missing: list[str]
    point_inferred: bool = False
    notes: list[str] = Field(default_factory=list)
    message: str

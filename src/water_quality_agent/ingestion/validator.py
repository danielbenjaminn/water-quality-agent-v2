from __future__ import annotations
import pandas as pd

from .models import (
    ALL_FIELDS, FIELD_DESCRIPTIONS, REQUIRED_FIELDS,
    ColumnInterpretation, IngestionReport, SemanticMapping,
)

SINGLE_POINT_VALUE = "__SINGLE_POINT__"


def canonicalize(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    rename = {original: semantic for semantic, original in mapping.items() if original}
    return df.rename(columns=rename).copy()


def validate_and_canonicalize(df: pd.DataFrame, semantic: SemanticMapping) -> tuple[pd.DataFrame | None, IngestionReport]:
    mapping = semantic.mapping
    found = [field for field in REQUIRED_FIELDS if mapping.get(field)]
    missing = [field for field in REQUIRED_FIELDS if not mapping.get(field)]
    valid = not missing

    reverse = {column: field for field, column in mapping.items() if column}
    interpretations: list[ColumnInterpretation] = []
    for column in df.columns:
        field = reverse.get(str(column))
        interpretations.append(ColumnInterpretation(
            column=str(column),
            semantic_field=field,
            meaning=FIELD_DESCRIPTIONS[field] if field else "Não utilizada / significado não necessário ao domínio",
            confidence=semantic.confidence.get(field) if field else None,
        ))

    point_inferred = valid and not mapping.get("point")
    notes = list(semantic.notes)
    if point_inferred:
        notes.append("Nenhuma coluna de ponto foi identificada; o dataset será tratado como um único ponto não identificado.")

    if valid:
        canonical = canonicalize(df, mapping)
        if point_inferred:
            canonical["point"] = SINGLE_POINT_VALUE
        message = "Base apta para análise: data, parâmetro, resultado e unidade foram identificados."
    else:
        canonical = None
        labels = ", ".join(missing)
        message = f"Base bloqueada: não foi possível identificar os campos obrigatórios: {labels}."

    report = IngestionReport(
        valid=valid,
        rows=len(df),
        columns=len(df.columns),
        interpretations=interpretations,
        required_found=found,
        required_missing=missing,
        point_inferred=point_inferred,
        notes=notes,
        message=message,
    )
    return canonical, report


def format_ingestion_report(report: IngestionReport) -> str:
    status = "BASE APTA PARA ANÁLISE" if report.valid else "BASE NÃO APTA PARA ANÁLISE"
    lines = [status, "", f"Registros: {report.rows}", f"Colunas: {report.columns}", "", "Colunas identificadas:"]
    for item in report.interpretations:
        if item.semantic_field:
            confidence = f" ({item.confidence:.0%})" if item.confidence is not None else ""
            lines.append(f"- {item.column} -> {item.semantic_field}{confidence}: {item.meaning}")
        else:
            lines.append(f"- {item.column} -> unknown: não utilizada pelo domínio")
    lines.extend(["", report.message])
    if report.point_inferred:
        lines.append("Ponto: não identificado; todas as observações serão tratadas como pertencentes a um único ponto.")
    if report.notes:
        lines.append("")
        lines.append("Observações:")
        lines.extend(f"- {note}" for note in report.notes)
    return "\n".join(lines)

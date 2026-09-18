from __future__ import annotations
from pathlib import Path

from water_quality_agent.core.csv_loader import carregar_csv
from water_quality_agent.core.session import SESSION
from .profiler import profile_dataframe
from .semantic_mapper import semantic_mapping
from water_quality_agent.core.harmonization import harmonize_dataset
from .validator import format_ingestion_report, validate_and_canonicalize


def ingest_dataset(path: str | Path, llm=None) -> dict:
    loaded = carregar_csv(path)

    raw = loaded.dataframe.copy()

    semantic = semantic_mapping(
        raw,
        llm=llm,
    )

    canonical, report = validate_and_canonicalize(
        raw,
        semantic,
    )

    # =========================================================
    # PENDÊNCIAS DE RESOLUÇÃO
    # =========================================================
    #
    # Sempre existe, mesmo se a ingestão for inválida.
    #
    # Isso evita UnboundLocalError e mantém uma estrutura
    # consistente no metadata.
    # =========================================================

    pending_parameters = []

    # =========================================================
    # SESSION - DADOS BRUTOS
    # =========================================================

    SESSION.source = Path(path)
    SESSION.raw = raw

    SESSION.schema_map = {
        k: v
        for k, v in semantic.mapping.items()
        if v
    }

    # =========================================================
    # DATASET VÁLIDO
    # =========================================================

    if report.valid:

        canonical, pending_parameters = harmonize_dataset(
            canonical,
            llm=llm,
        )

        SESSION.canonical = canonical

        fields = set(canonical.columns)

        SESSION.capabilities = {
            "descriptive": True,
            "time_series": True,
            "regulatory": True,
            "trend": True,

            "compare_points": bool(
                semantic.mapping.get("point")
            ),

            "campaign_comparison": (
                "campaign" in fields
            ),

            "basin_context": (
                "basin" in fields
            ),

            "sub_basin_context": (
                "sub_basin" in fields
            ),
        }

    # =========================================================
    # DATASET INVÁLIDO
    # =========================================================

    else:

        SESSION.canonical = None
        SESSION.capabilities = {}

    # =========================================================
    # METADATA
    # =========================================================
    #
    # Só montamos DEPOIS da harmonização, pois agora
    # pending_parameters já existe.
    # =========================================================

    SESSION.metadata = {
        "encoding": loaded.encoding,
        "separator": loaded.separador,

        "schema_notes": semantic.notes,

        "ingestion_valid": report.valid,

        "point_inferred": report.point_inferred,

        "ingestion_report": (
            report.model_dump()
        ),

        "pending_parameters": [
            item.model_dump()
            for item in pending_parameters
        ],
    }

    # =========================================================
    # RESULTADO
    # =========================================================

    return {
        "valid": report.valid,

        "profile": profile_dataframe(raw),

        "mapping": semantic.model_dump(),

        "report": report.model_dump(),

        "report_text": format_ingestion_report(
            report
        ),

        "capabilities": SESSION.capabilities,

        "metadata": SESSION.metadata,
    }
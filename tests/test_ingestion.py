import pandas as pd

from water_quality_agent.ingestion.semantic_mapper import semantic_mapping
from water_quality_agent.ingestion.validator import validate_and_canonicalize, SINGLE_POINT_VALUE


def test_required_fields_and_optional_point():
    df = pd.DataFrame({
        "Data de coleta": ["01/01/2024", "02/01/2024"],
        "Analito": ["OD", "OD"],
        "Valor medido": [6.2, 5.8],
        "Unidade": ["mg/L", "mg/L"],
        "Observacao": ["ok", "ok"],
    })
    semantic = semantic_mapping(df, llm=None)
    canonical, report = validate_and_canonicalize(df, semantic)
    assert report.valid is True
    assert report.point_inferred is True
    assert (canonical["point"] == SINGLE_POINT_VALUE).all()
    assert next(i for i in report.interpretations if i.column == "Observacao").semantic_field is None


def test_missing_required_unit_blocks_dataset():
    df = pd.DataFrame({
        "Data": ["01/01/2024"],
        "Parametro": ["OD"],
        "Resultado": [6.2],
        "Ponto": ["AV007"],
    })
    semantic = semantic_mapping(df, llm=None)
    canonical, report = validate_and_canonicalize(df, semantic)
    assert report.valid is False
    assert canonical is None
    assert "unit" in report.required_missing


def test_optional_domain_columns_are_recognized_but_geo_is_unknown():
    df = pd.DataFrame({
        "Data": ["01/01/2024"], "Parametro": ["OD"], "Resultado": [6.2], "Unidade": ["mg/L"],
        "Numero do laudo": ["L-1"], "Campanha": ["C1"], "Matriz": ["Agua superficial"],
        "Sub-bacia": ["Velhas"], "Bacia": ["Sao Francisco"], "Latitude": [-19.9], "Longitude": [-43.9],
    })
    semantic = semantic_mapping(df, llm=None)
    canonical, report = validate_and_canonicalize(df, semantic)
    assert report.valid is True
    assert semantic.mapping["report_id"] == "Numero do laudo"
    assert semantic.mapping["campaign"] == "Campanha"
    assert semantic.mapping["matrix"] == "Matriz"
    assert semantic.mapping["sub_basin"] == "Sub-bacia"
    assert semantic.mapping["basin"] == "Bacia"
    geo = {i.column: i.semantic_field for i in report.interpretations if i.column in {"Latitude", "Longitude"}}
    assert geo == {"Latitude": None, "Longitude": None}

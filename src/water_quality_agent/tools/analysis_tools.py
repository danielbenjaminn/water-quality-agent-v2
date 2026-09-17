from __future__ import annotations
import pandas as pd
from langchain_core.tools import tool
from water_quality_agent.core.descriptive import descriptive_analysis
from water_quality_agent.core.trend import mann_kendall_analysis
from water_quality_agent.core.correlation import correlation_analysis
from water_quality_agent.core.outliers import flag_iqr_outliers
from water_quality_agent.core.extracao_qualifier import extrair_qualif


def _records_df(observations: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(observations)
    rename = {"parameter":"Parâmetro B.D.", "result":"Resultado", "point":"Ponto", "date":"Data", "qualifier":"Qualifier", "unit":"Unidade"}
    df = df.rename(columns=rename)
    if "Resultado" in df.columns:
        parsed = [extrair_qualif(str(v).replace(",", "."), i) for i, v in df["Resultado"].items()]
        if "Qualifier" not in df.columns:
            df["Qualifier"] = [x[0] for x in parsed]
        else:
            extracted = pd.Series([x[0] for x in parsed], index=df.index)
            df["Qualifier"] = df["Qualifier"].where(df["Qualifier"].notna() & df["Qualifier"].astype(str).str.strip().ne(""), extracted)
        df["Resultado Num"] = [x[1] for x in parsed]
    return df

@tool
def descriptive_statistics(observations: list[dict]) -> list[dict]:
    """Calcula estatísticas descritivas/censura somente sobre as observações fornecidas pelo agente."""
    if not observations: return []
    df = _records_df(observations)
    return descriptive_analysis(df=df, parameter_col="Parâmetro B.D.", value_col="Resultado Num", point_col="Ponto" if "Ponto" in df else None)

@tool
def mann_kendall_trend(observations: list[dict]) -> list[dict]:
    """Executa análise de tendência Mann-Kendall somente na série selecionada pelo agente."""
    if not observations: return []
    return mann_kendall_analysis(_records_df(observations), parameter_col="Parâmetro B.D.", value_col="Resultado Num")

@tool
def kendall_correlations(observations: list[dict]) -> list[dict]:
    """Calcula correlações Kendall sobre o subconjunto selecionado pelo agente."""
    if not observations: return []
    return correlation_analysis(_records_df(observations))

@tool
def iqr_outliers(observations: list[dict]) -> list[dict]:
    """Marca outliers IQR no subconjunto fornecido; não remove observações."""
    if not observations: return []
    df = flag_iqr_outliers(_records_df(observations))
    return df.where(pd.notna(df), None).to_dict("records")

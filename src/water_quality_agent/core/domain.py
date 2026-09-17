from __future__ import annotations
from pathlib import Path
from ast import literal_eval
import pandas as pd
from .csv_loader import carregar_csv
from .padronizar_parametros import padronizar_param, trat_string

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "config"


def load_legal_limits() -> pd.DataFrame:
    return carregar_csv(CONFIG / "limites_legais.csv").dataframe.copy()


def parameter_catalog() -> tuple[pd.DataFrame, dict[str, str]]:
    limits = load_legal_limits()
    col = "Parâmetro B.D."
    d = {trat_string(v): v for v in limits[col].dropna().astype(str).unique()}
    return limits, d


def resolve_parameter(name: str) -> dict:
    limits, catalog = parameter_catalog()
    # Reuse the existing fuzzy implementation, without pipeline orchestration.
    rules_text = (CONFIG / "regras_parametros.txt").read_text(encoding="utf-8")
    rules = literal_eval(f"[{rules_text}]")
    result = padronizar_param(p=name, param_dict=catalog, regras_parametros=rules)
    canonical, score, method = result
    return {"input": name, "canonical": canonical, "score": float(score) if score is not None else None, "method": method}


def legal_limits_for(parameter: str) -> list[dict]:
    limits = load_legal_limits()
    subset = limits[limits["Parâmetro B.D."].astype(str).str.casefold() == str(parameter).casefold()].copy()
    # Projeto deliberadamente restrito a água superficial doce Classe 2 / CONAMA 357.
    if "Classe" in subset.columns:
        subset = subset[subset["Classe"].astype(str).str.contains("Classe 2", case=False, na=False)]
    cols = [c for c in ["Legislação", "Classe", "Parâmetro B.D.", "Parâmetro", "Unidade", "Min", "Max", "Obs"] if c in subset.columns]
    return subset[cols].where(pd.notna(subset[cols]), None).to_dict("records")

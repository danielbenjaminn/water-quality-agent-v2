from __future__ import annotations

import math

import pandas as pd


def calculate_regulatory_compliance(
    df: pd.DataFrame,
    minimum: float | None = None,
    maximum: float | None = None,
) -> dict:
    """
    Calcula a conformidade regulatória de uma série harmonizada.

    Nesta primeira versão, apenas observações não censuradas são
    classificadas como conformes ou não conformes.

    Observações censuradas (< ou >) são reportadas separadamente e
    não entram no cálculo da proporção de conformidade.
    """

    if df.empty:
        return {
            "n_total": 0,
            "n_evaluable": 0,
            "n_censored": 0,
            "n_compliant": 0,
            "n_non_compliant": 0,
            "compliance_fraction": None,
            "non_compliance_fraction": None,
            "valid": False,
            "reason": "no_observations",
        }

    data = df.copy()

    data["result"] = pd.to_numeric(
        data["result"],
        errors="coerce",
    )

    valid_result = data["result"].notna()

    qualifiers = (
        data["qualifier"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    censored = qualifiers.isin(["<", ">"])

    evaluable = valid_result & ~censored

    evaluated = data.loc[evaluable].copy()

    n_total = len(data)
    n_censored = int((valid_result & censored).sum())
    n_evaluable = len(evaluated)

    if n_evaluable == 0:
        return {
            "n_total": n_total,
            "n_evaluable": 0,
            "n_censored": n_censored,
            "n_compliant": 0,
            "n_non_compliant": 0,
            "compliance_fraction": None,
            "non_compliance_fraction": None,
            "valid": False,
            "reason": "no_evaluable_observations",
        }

    compliant = pd.Series(
        True,
        index=evaluated.index,
        dtype=bool,
    )

    if minimum is not None:
        minimum = float(minimum)

        if not math.isnan(minimum):
            compliant &= evaluated["result"] >= minimum

    if maximum is not None:
        maximum = float(maximum)

        if not math.isnan(maximum):
            compliant &= evaluated["result"] <= maximum

    n_compliant = int(compliant.sum())
    n_non_compliant = int((~compliant).sum())

    return {
        "n_total": n_total,
        "n_evaluable": n_evaluable,
        "n_censored": n_censored,
        "n_compliant": n_compliant,
        "n_non_compliant": n_non_compliant,
        "compliance_fraction": n_compliant / n_evaluable,
        "non_compliance_fraction": n_non_compliant / n_evaluable,
        "valid": True,
        "reason": None,
    }
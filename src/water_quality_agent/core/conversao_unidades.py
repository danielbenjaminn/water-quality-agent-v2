from __future__ import annotations

import pandas as pd


# ============================================================
# NORMALIZAÇÃO TEXTUAL DAS UNIDADES
# ============================================================

def padronizar_unidades(s: pd.Series) -> pd.Series:
    """
    Limpa e padroniza unidades para sua representação física.

    Informações sobre espécie química, elemento, composto ou
    analito são removidas da unidade.

    Exemplos:
        mg/L N       -> mg/L
        mg/L P       -> mg/L
        mg/L SO4     -> mg/L
        mg/L C6H5OH  -> mg/L
        mgO/L O2     -> mg/L
        mgPb/L Pb    -> mg/L
        µg/L TBT     -> µg/L
        μg/L         -> µg/L
        ug/L         -> µg/L
    """

    result = s.astype("string").str.strip()

    # ========================================================
    # 1. LIMPEZA GERAL
    # ========================================================

    result = result.str.replace(
        r"\s+",
        " ",
        regex=True,
    )

    # Padroniza símbolo micro
    result = result.str.replace(
        "μ",
        "µ",
        regex=False,
    )

    # ========================================================
    # 2. CORREÇÕES EXPLÍCITAS
    # ========================================================

    # Erro conhecido da base
    result = result.replace(
        {
            r"(?i)^mg\s*/\s*fl.*$": "mg/L",
        },
        regex=True,
    )

    # ========================================================
    # 3. MASSA / VOLUME
    #
    # Tudo que representar mg por litro vira mg/L.
    #
    # Exemplos:
    # mg/L
    # mg / L
    # mg/L N
    # mg/L SO4
    # mg/L C6H5OH
    # mgPb/L Pb
    # mgO/L O2
    # ========================================================

    result = result.replace(
        {
            # mg/L + qualquer informação posterior
            r"(?i)^mg\s*/\s*l(?:\s+.*)?$": "mg/L",

            # mgX/L ou mgXYZ/L + informação posterior
            r"(?i)^mg[a-z0-9]+\s*/\s*l(?:\s+.*)?$": "mg/L",

            # outras grafias simples
            r"(?i)^mg\s*[.]\s*l(?:\s+.*)?$": "mg/L",
            r"(?i)^mg\s+l(?:itro)?(?:\^-?1|-1)?(?:\s+.*)?$": "mg/L",
        },
        regex=True,
    )

    # ========================================================
    # 4. MICROGRAMA / VOLUME
    #
    # Tudo que representar µg por litro vira µg/L.
    # ========================================================

    result = result.replace(
        {
            r"(?i)^(?:ug|µg)\s*/\s*l(?:\s+.*)?$": "µg/L",

            r"(?i)^(?:ug|µg)[a-z0-9]+\s*/\s*l(?:\s+.*)?$": "µg/L",

            r"(?i)^(?:ug|µg)\s*[.]\s*l(?:\s+.*)?$": "µg/L",

            r"(?i)^(?:ug|µg)\s+l(?:itro)?(?:\^-?1|-1)?(?:\s+.*)?$": "µg/L",
        },
        regex=True,
    )

    # ========================================================
    # 5. GRAMA / VOLUME
    # ========================================================

    result = result.replace(
        {
            r"(?i)^g\s*/\s*l(?:\s+.*)?$": "g/L",
            r"(?i)^g\s*[.]\s*l(?:\s+.*)?$": "g/L",
        },
        regex=True,
    )

    # ========================================================
    # 6. MASSA / MASSA
    # ========================================================

    result = result.replace(
        {
            r"(?i)^mg\s*/\s*kg(?:\s+.*)?$": "mg/kg",

            r"(?i)^(?:ug|µg)\s*/\s*kg(?:\s+.*)?$": "µg/kg",
        },
        regex=True,
    )

    # ========================================================
    # 7. MICROBIOLOGIA
    # ========================================================

    result = result.replace(
        {
            r"(?i)^nmp\s*/\s*100\s*ml.*$": "NMP/100mL",

            r"(?i)^ufc\s*/\s*100\s*ml.*$": "UFC/100mL",

            r"(?i)^col\s*/\s*100\s*ml.*$": "UFC/100mL",

            r"(?i)^c[ée]l(?:ulas?)?\.?\s*/\s*ml.*$": "cel/mL",
        },
        regex=True,
    )

    # ========================================================
    # 8. CONDUTIVIDADE
    # ========================================================

    result = result.replace(
        {
            r"(?i)^(?:us|µs)\s*/\s*cm.*$": "µS/cm",

            r"(?i)^ms\s*/\s*cm.*$": "mS/cm",
        },
        regex=True,
    )

    # ========================================================
    # 9. TURBIDEZ
    # ========================================================

    result = result.replace(
        {
            r"(?i)^ntu$": "NTU",
            r"(?i)^unt$": "UNT",
        },
        regex=True,
    )

    # ========================================================
    # 10. TEMPERATURA
    # ========================================================

    result = result.replace(
        {
            r"(?i)^[º°]\s*c.*$": "°C",
        },
        regex=True,
    )

    # ========================================================
    # 11. VOLUME / VOLUME
    # ========================================================

    result = result.replace(
        {
            r"(?i)^ml\s*/\s*l.*$": "mL/L",
        },
        regex=True,
    )

    # ========================================================
    # 12. SEM UNIDADE / QUALITATIVO
    # ========================================================

    result = result.replace(
        {
            r"(?i)^nounit$": "-",
            r"(?i)^ausente$": "P/A",
        },
        regex=True,
    )

    return result

# ============================================================
# TABELA ÚNICA DE CONVERSÕES
# ============================================================

CONVERSION_FACTORS = {
    # massa / volume
    ("mg/L", "µg/L"): 1000.0,
    ("µg/L", "mg/L"): 0.001,

    ("g/L", "mg/L"): 1000.0,
    ("mg/L", "g/L"): 0.001,

    # condutividade
    ("mS/cm", "µS/cm"): 1000.0,
    ("µS/cm", "mS/cm"): 0.001,

    # equivalências adotadas pelo projeto
    ("NTU", "UNT"): 1.0,
    ("UNT", "NTU"): 1.0,

    ("CU", "mg Pt/L"): 1.0,
    ("mg Pt/L", "CU"): 1.0,

    ("UFC/100mL", "NMP/100mL"): 1.0,
}


# ============================================================
# FATOR DE CONVERSÃO
# ============================================================

def conversion_factor(
    source_unit: str | None,
    target_unit: str | None,
) -> float | None:
    """
    Retorna o fator necessário para converter source_unit
    em target_unit.

    Retorna:
        1.0  -> unidades iguais
        fator -> conversão conhecida
        None -> conversão não conhecida / unidade ausente
    """

    if source_unit is None or target_unit is None:
        return None

    if pd.isna(source_unit) or pd.isna(target_unit):
        return None

    source = str(source_unit).strip()
    target = str(target_unit).strip()

    if not source or not target:
        return None

    if source == target:
        return 1.0

    return CONVERSION_FACTORS.get(
        (source, target)
    )


# ============================================================
# CONVERSÃO DO DATAFRAME ANALÍTICO
# ============================================================

def convert_analytical_units(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Converte a coluna `result` da unidade observada (`unit`)
    para a unidade canônica (`target_unit`).

    Espera as colunas:
        result
        unit
        target_unit

    Cria:
        conversion_factor
        conversion_status

    Ao final:
        - `result` contém o valor convertido;
        - `unit` contém a unidade final;
        - conversões desconhecidas NÃO alteram resultado/unidade.
    """

    out = df.copy()

    factors = []

    statuses = []

    converted_results = []

    final_units = []

    for _, row in out.iterrows():

        result = row.get("result")
        source_unit = row.get("unit")
        target_unit = row.get("target_unit")

        # ----------------------------------------------------
        # Unidade de destino não resolvida
        # ----------------------------------------------------

        if (
            target_unit is None
            or pd.isna(target_unit)
            or str(target_unit).strip() == ""
        ):
            factors.append(None)
            statuses.append("unresolved")
            converted_results.append(result)
            final_units.append(source_unit)
            continue

        # ----------------------------------------------------
        # Fator
        # ----------------------------------------------------

        factor = conversion_factor(
            source_unit=source_unit,
            target_unit=target_unit,
        )

        # ----------------------------------------------------
        # Conversão não conhecida
        # ----------------------------------------------------

        if factor is None:
            factors.append(None)
            statuses.append("unresolved")
            converted_results.append(result)
            final_units.append(source_unit)
            continue

        # ----------------------------------------------------
        # Unidade já correta
        # ----------------------------------------------------

        if factor == 1.0 and source_unit == target_unit:
            factors.append(1.0)
            statuses.append("identity")
            converted_results.append(result)
            final_units.append(target_unit)
            continue

        # ----------------------------------------------------
        # Resultado ausente
        #
        # A unidade ainda pode ser harmonizada porque a
        # conversão entre as unidades é conhecida.
        # ----------------------------------------------------

        if result is None or pd.isna(result):
            factors.append(factor)

            if source_unit == target_unit:
                statuses.append("identity")
            else:
                statuses.append("converted")

            converted_results.append(result)
            final_units.append(target_unit)
            continue

        # ----------------------------------------------------
        # Conversão física
        # ----------------------------------------------------

        converted_results.append(
            float(result) * factor
        )

        factors.append(factor)

        if source_unit == target_unit:
            statuses.append("identity")
        else:
            statuses.append("converted")

        final_units.append(target_unit)

    out["conversion_factor"] = factors
    out["conversion_status"] = statuses

    out["result"] = converted_results
    out["unit"] = final_units

    return out
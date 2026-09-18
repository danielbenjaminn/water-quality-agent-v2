import pandas as pd

from water_quality_agent.core.conversao_unidades import (
    convert_analytical_units,
)


df = pd.DataFrame(
    {
        "parameter": [
            "TESTE A",
            "TESTE B",
            "TESTE C",
            "TESTE D",
            "TESTE E",
        ],
        "qualifier": [
            None,
            None,
            "<",
            ">",
            None,
        ],
        "result": [
            0.002,
            2.0,
            0.002,
            2.0,
            10.0,
        ],
        "unit": [
            "mg/L",
            "µg/L",
            "mg/L",
            "µg/L",
            "mg/L",
        ],
        "target_unit": [
            "µg/L",
            "mg/L",
            "µg/L",
            "mg/L",
            "mg/L",
        ],
    }
)


result = convert_analytical_units(df)


print("\nRESULTADO")
print(
    result[
        [
            "parameter",
            "qualifier",
            "result",
            "unit",
            "conversion_factor",
            "conversion_status",
        ]
    ].to_string(index=False)
)


# ============================================================
# TESTES
# ============================================================

# 0.002 mg/L -> 2 µg/L
assert result.loc[0, "result"] == 2.0
assert result.loc[0, "unit"] == "µg/L"
assert result.loc[0, "conversion_status"] == "converted"


# 2 µg/L -> 0.002 mg/L
assert result.loc[1, "result"] == 0.002
assert result.loc[1, "unit"] == "mg/L"
assert result.loc[1, "conversion_status"] == "converted"


# <0.002 mg/L -> <2 µg/L
assert result.loc[2, "qualifier"] == "<"
assert result.loc[2, "result"] == 2.0
assert result.loc[2, "unit"] == "µg/L"


# >2 µg/L -> >0.002 mg/L
assert result.loc[3, "qualifier"] == ">"
assert result.loc[3, "result"] == 0.002
assert result.loc[3, "unit"] == "mg/L"


# Unidade já correta
assert result.loc[4, "result"] == 10.0
assert result.loc[4, "unit"] == "mg/L"
assert result.loc[4, "conversion_status"] == "identity"


print("\nOK - conversão mg/L -> µg/L")
print("OK - conversão µg/L -> mg/L")
print("OK - qualifier < preservado")
print("OK - qualifier > preservado")
print("OK - identidade preservada")
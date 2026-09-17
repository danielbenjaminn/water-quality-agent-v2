from __future__ import annotations
from langchain_core.tools import tool

FACTORS = {
    ("ug/l", "mg/l"): 1e-3, ("µg/l", "mg/l"): 1e-3,
    ("mg/l", "ug/l"): 1e3, ("mg/l", "µg/l"): 1e3,
    ("mg/l", "mg/l"): 1.0, ("ug/l", "ug/l"): 1.0, ("µg/l", "µg/l"): 1.0,
    ("g/l", "mg/l"): 1e3, ("mg/l", "g/l"): 1e-3,
}

def _u(x: str) -> str: return str(x).strip().lower().replace("μ", "µ")

@tool
def normalize_observation_units(observations: list[dict], target_unit: str) -> list[dict]:
    """Converte resultados numéricos para uma unidade comum. Falha explicitamente para conversões não cadastradas."""
    out=[]
    for obs in observations:
        row=dict(obs); src=row.get("unit")
        if src is None: raise ValueError("Observação sem unidade.")
        key=(_u(src), _u(target_unit))
        if key not in FACTORS: raise ValueError(f"Conversão não cadastrada: {src} -> {target_unit}")
        raw=row.get("result")
        # preserva qualificador textual; converte magnitude
        text=str(raw).strip().replace(",", ".")
        import re
        m=re.search(r"[-+]?\d+(?:\.\d+)?", text)
        if not m: raise ValueError(f"Resultado não numérico: {raw}")
        value=float(m.group())*FACTORS[key]
        prefix=text[:m.start()].strip()
        row["result"] = f"{prefix}{value}" if prefix else value
        row["unit"] = target_unit
        out.append(row)
    return out

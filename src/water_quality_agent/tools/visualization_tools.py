from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from langchain_core.tools import tool

@tool
def plot_time_series(observations: list[dict], title: str, minimum: float | None = None, maximum: float | None = None) -> str:
    """Plota série temporal já selecionada/normalizada e linhas regulatórias min/max quando fornecidas."""
    if not observations: raise ValueError("Sem observações para plotar.")
    df=pd.DataFrame(observations)
    if "date" not in df: raise ValueError("Série temporal requer data.")
    df["date"]=pd.to_datetime(df["date"], errors="coerce")
    df["value"]=df["result"].astype(str).str.replace(",",".",regex=False).str.extract(r"([-+]?\d+(?:\.\d+)?)")[0].astype(float)
    df=df.dropna(subset=["date","value"]).sort_values("date")
    unit=str(df["unit"].dropna().iloc[0]) if "unit" in df and df["unit"].notna().any() else ""
    fig,ax=plt.subplots(figsize=(11,5)); ax.plot(df["date"],df["value"],marker="o",label="Observações")
    if minimum is not None: ax.axhline(minimum,linestyle="--",label=f"Mínimo CONAMA: {minimum} {unit}")
    if maximum is not None: ax.axhline(maximum,linestyle="--",label=f"Máximo CONAMA: {maximum} {unit}")
    ax.set(title=title,xlabel="Data",ylabel=unit); ax.legend(); ax.grid(alpha=.25); fig.autofmt_xdate()
    out=Path("outputs"); out.mkdir(exist_ok=True); path=out/"time_series.png"; fig.savefig(path,dpi=150,bbox_inches="tight"); plt.close(fig)
    return str(path.resolve())

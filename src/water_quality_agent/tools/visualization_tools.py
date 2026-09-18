from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from langchain_core.tools import tool


@tool
def plot_time_series(
    observations: list[dict],
    title: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> str:
    """
    Plota uma série temporal previamente selecionada e harmonizada.

    Os valores de `result` devem ser numéricos e as unidades já devem
    estar harmonizadas.

    Valores censurados são plotados na magnitude reportada, com
    marcadores distintos para qualifiers "<" e ">".

    Linhas regulatórias mínima e máxima são adicionadas quando
    fornecidas.
    """
    if not observations:
        raise ValueError("Sem observações para plotar.")

    df = pd.DataFrame(observations)

    required = {"date", "result"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Colunas obrigatórias ausentes para série temporal: "
            f"{sorted(missing)}"
        )

    # ---------------------------------------------------------
    # Tipos
    # ---------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df["result"] = pd.to_numeric(
        df["result"],
        errors="coerce",
    )

    df = (
        df
        .dropna(subset=["date", "result"])
        .sort_values("date")
    )

    if df.empty:
        raise ValueError(
            "Nenhuma observação válida para plotar."
        )

    # ---------------------------------------------------------
    # Unidade
    # ---------------------------------------------------------

    unit = ""

    if "unit" in df.columns:
        units = (
            df["unit"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if len(units) > 1:
            raise ValueError(
                "A série contém mais de uma unidade analítica: "
                f"{units}. Os dados deveriam estar harmonizados."
            )

        if units:
            unit = units[0]

    # ---------------------------------------------------------
    # Qualifiers
    # ---------------------------------------------------------

    if "qualifier" not in df.columns:
        df["qualifier"] = None

    qualifiers = (
        df["qualifier"]
        .astype("string")
        .str.strip()
    )

    left_censored = df[
        qualifiers.eq("<").fillna(False)
    ]

    right_censored = df[
        qualifiers.eq(">").fillna(False)
    ]

    censored_mask = (
        qualifiers.isin(["<", ">"])
        .fillna(False)
    )

    regular = df[
        ~censored_mask
    ]

    # ---------------------------------------------------------
    # Gráfico
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    # Linha da série completa.
    #
    # Ela conecta as magnitudes reportadas, inclusive limites de
    # quantificação/detecção de observações censuradas.
    ax.plot(
        df["date"],
        df["result"],
        linewidth=1,
        alpha=0.7,
    )

    if not regular.empty:
        ax.scatter(
            regular["date"],
            regular["result"],
            marker="o",
            label="Observações",
        )

    if not left_censored.empty:
        ax.scatter(
            left_censored["date"],
            left_censored["result"],
            marker="v",
            label="< limite reportado",
        )

    if not right_censored.empty:
        ax.scatter(
            right_censored["date"],
            right_censored["result"],
            marker="^",
            label="> limite reportado",
        )

    # ---------------------------------------------------------
    # Limites regulatórios
    # ---------------------------------------------------------

    if minimum is not None:
        ax.axhline(
            minimum,
            linestyle="--",
            label=(
                f"Mínimo CONAMA: {minimum}"
                + (f" {unit}" if unit else "")
            ),
        )

    if maximum is not None:
        ax.axhline(
            maximum,
            linestyle="--",
            label=(
                f"Máximo CONAMA: {maximum}"
                + (f" {unit}" if unit else "")
            ),
        )

    # ---------------------------------------------------------
    # Aparência
    # ---------------------------------------------------------

    ax.set_title(title)
    ax.set_xlabel("Data")
    ax.set_ylabel(unit or "Resultado")

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.autofmt_xdate()

    # ---------------------------------------------------------
    # Saída
    # ---------------------------------------------------------

    output_dir = Path("outputs")

    output_dir.mkdir(
        exist_ok=True
    )

    path = output_dir / "time_series.png"

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    return str(path.resolve())
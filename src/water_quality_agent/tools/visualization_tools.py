from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from langchain_core.tools import tool

from water_quality_agent.tools.data_tools import select_series


@tool
def plot_time_series(
    parameter: str,
    point: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    title: str | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> str:
    """
    Plota uma série temporal diretamente do dataset harmonizado.

    `parameter` deve ser o nome canônico previamente resolvido.

    Os dados são recuperados diretamente do backend e não devem ser
    enviados pelo LLM como lista de observações.

    Linhas regulatórias mínima e máxima são adicionadas quando
    fornecidas.
    """

    df = select_series(
        parameter=parameter,
        point=point,
        start_date=start_date,
        end_date=end_date,
    )

    if df.empty:
        raise ValueError(
            f"Sem observações para parameter={parameter!r}"
            + (
                f", point={point!r}."
                if point is not None
                else "."
            )
        )

    required = {
        "date",
        "result",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Colunas obrigatórias ausentes para série temporal: "
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
            color="orange",
            linestyle="--",
            linewidth=1.5,
            label=(
                f"Mínimo CONAMA: {minimum}"
                + (f" {unit}" if unit else "")
            ),
        )

    if maximum is not None:
        ax.axhline(
            maximum,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=(
                f"Máximo CONAMA: {maximum}"
                + (f" {unit}" if unit else "")
            ),
        )

    # ---------------------------------------------------------
    # Aparência
    # ---------------------------------------------------------

    if title is None:
        title = parameter

        if point is not None:
            title += f" - {point}"

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
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ResultadoCarga:
    dataframe: pd.DataFrame
    caminho: Path
    encoding: str
    separador: str


ENCODINGS = (
    "utf-8",
    "utf-8-sig",
    "cp1252",
    "latin1",
)


def _detectar_encoding(caminho: Path) -> str:
    """
    Tenta identificar um encoding capaz de decodificar o arquivo.

    Não pretende realizar detecção probabilística: apenas testa
    encodings comuns em bases CSV brasileiras.
    """
    amostra = caminho.read_bytes()[:100_000]

    for encoding in ENCODINGS:
        try:
            amostra.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue

    raise ValueError(
        f"Não foi possível decodificar '{caminho.name}' "
        f"com os encodings suportados: {ENCODINGS}."
    )


def _detectar_separador(caminho: Path, encoding: str) -> str:
    """
    Detecta delimitador utilizando csv.Sniffer.
    """
    with caminho.open(
        "r",
        encoding=encoding,
        errors="replace",
    ) as arquivo:
        amostra = arquivo.read(20_000)

    try:
        dialect = csv.Sniffer().sniff(
            amostra,
            delimiters=",;\t|",
        )
        return dialect.delimiter

    except csv.Error:
        # CSV brasileiro frequentemente usa ;
        # Mas não queremos assumir isso silenciosamente.
        contagens = {
            separador: amostra.count(separador)
            for separador in (",", ";", "\t", "|")
        }

        separador = max(contagens, key=contagens.get)

        if contagens[separador] == 0:
            raise ValueError(
                "Não foi possível identificar o separador do CSV."
            )

        return separador


def carregar_csv(
    caminho: str | Path,
    *,
    encoding: str | None = None,
    separador: str | None = None,
) -> ResultadoCarga:
    """
    Carrega um arquivo CSV.

    Parameters
    ----------
    caminho:
        Caminho para o arquivo.

    encoding:
        Encoding informado manualmente.
        Se None, será detectado.

    separador:
        Delimitador informado manualmente.
        Se None, será detectado.

    Returns
    -------
    ResultadoCarga
        DataFrame e metadados utilizados durante a leitura.
    """
    caminho = Path(caminho)

    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}"
        )

    if not caminho.is_file():
        raise ValueError(
            f"O caminho informado não corresponde a um arquivo: {caminho}"
        )

    if caminho.suffix.lower() != ".csv":
        raise ValueError(
            f"Formato não suportado: '{caminho.suffix}'. "
            "Nesta versão, apenas arquivos CSV são aceitos."
        )

    if caminho.stat().st_size == 0:
        raise ValueError("O arquivo CSV está vazio.")

    encoding_usado = encoding or _detectar_encoding(caminho)
    separador_usado = separador or _detectar_separador(
        caminho,
        encoding_usado,
    )

    try:
        df = pd.read_csv(
            caminho,
            sep=separador_usado,
            encoding=encoding_usado,
            dtype_backend="numpy_nullable",
        )

    except Exception as exc:
        raise ValueError(
            f"Erro ao carregar '{caminho.name}': {exc}"
        ) from exc

    if len(df.columns) == 0:
        raise ValueError(
            "O arquivo não contém colunas identificáveis."
        )

    # Remove apenas colunas completamente vazias.
    df = df.dropna(axis=1, how="all")


    return ResultadoCarga(
        dataframe=df,
        caminho=caminho,
        encoding=encoding_usado,
        separador=separador_usado,
    )
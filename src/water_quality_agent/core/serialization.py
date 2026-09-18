from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def to_json_safe(value: Any) -> Any:
    """
    Converte estruturas Python/pandas/numpy para valores seguros
    para serialização JSON.

    Exemplos:
    - NaN / NaT / pd.NA -> None
    - numpy scalar -> Python scalar
    - Timestamp -> ISO 8601
    - dict/list/tuple -> conversão recursiva
    """

    if value is None:
        return None

    if value is pd.NA or value is pd.NaT:
        return None

    if isinstance(value, (float, np.floating)):
        if pd.isna(value):
            return None
        return float(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, (pd.Timestamp,)):
        if pd.isna(value):
            return None
        return value.isoformat()

    if isinstance(value, dict):
        return {
            key: to_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            to_json_safe(item)
            for item in value
        ]

    return value
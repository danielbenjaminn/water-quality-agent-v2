from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import pandas as pd


@dataclass
class DatasetSession:
    """Estado do dataset fora do contexto do LLM.

    O modelo recebe IDs, perfis e resultados pequenos; o DataFrame permanece no backend.
    """
    source: Path | None = None
    raw: pd.DataFrame | None = None
    canonical: pd.DataFrame | None = None
    schema_map: dict[str, str] = field(default_factory=dict)
    capabilities: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def require_data(self) -> pd.DataFrame:
        if self.canonical is None:
            raise RuntimeError("Nenhum dataset preparado na sessão.")
        return self.canonical


SESSION = DatasetSession()

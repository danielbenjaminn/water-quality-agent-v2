"""Compatibilidade: o schema semântico agora pertence à camada `ingestion`."""
from water_quality_agent.ingestion.models import *  # noqa: F401,F403
from water_quality_agent.ingestion.profiler import profile_dataframe  # noqa: F401
from water_quality_agent.ingestion.semantic_mapper import (  # noqa: F401
    ALIASES, deterministic_mapping, semantic_mapping,
)
from water_quality_agent.ingestion.validator import canonicalize  # noqa: F401

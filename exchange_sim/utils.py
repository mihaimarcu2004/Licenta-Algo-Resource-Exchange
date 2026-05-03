from __future__ import annotations

from .types import AgentId, Edge


def normalize_edge(u: AgentId, v: AgentId) -> Edge:
    """Return a stable undirected edge key."""
    return tuple(sorted((u, v), key=lambda x: str(x)))  # type: ignore[return-value]


__all__ = ["normalize_edge"]

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .types import AgentId, Edge, Allocation
from .utils import normalize_edge


@dataclass
class LinkState:
    """State of one active undirected link."""

    u: AgentId
    v: AgentId
    remaining_life: Optional[int] = None  # None means permanent.
    quality: float = 1.0  # Utility/flow multiplier caused by degradation.
    degradation: float = 1.0  # Per-step degradation multiplier for this link.
    initial: bool = False  # Initial links may be treated differently.
    created_at: int = 0

    def other(self, agent: AgentId) -> AgentId:
        if agent == self.u:
            return self.v
        if agent == self.v:
            return self.u
        raise ValueError(f"Agent {agent!r} is not incident to link {(self.u, self.v)!r}.")


@dataclass
class SimulationState:
    """Snapshot-like state exposed to strategies and utility functions."""

    t: int
    agents: List[AgentId]
    production: Dict[AgentId, float]
    links: Dict[Edge, LinkState]
    last_allocation: Allocation = field(default_factory=dict)
    last_creation_costs: Dict[AgentId, float] = field(default_factory=dict)
    last_renewal_costs: Dict[AgentId, float] = field(default_factory=dict)
    last_renewals: Dict[tuple[AgentId, AgentId], int] = field(default_factory=dict)
    last_received: Dict[AgentId, Dict[AgentId, float]] = field(default_factory=dict)
    exchange_ratios: Dict[AgentId, float] = field(default_factory=dict)
    utilities: Dict[AgentId, float] = field(default_factory=dict)
    _neighbors_cache: Optional[Dict[AgentId, List[AgentId]]] = field(default=None, init=False, repr=False)

    def neighbors(self, agent: AgentId) -> List[AgentId]:
        if self._neighbors_cache is None:
            cache: Dict[AgentId, List[AgentId]] = {agent_id: [] for agent_id in self.agents}
            for edge, link in self.links.items():
                cache.setdefault(edge[0], []).append(link.other(edge[0]))
                cache.setdefault(edge[1], []).append(link.other(edge[1]))
            self._neighbors_cache = cache
        return self._neighbors_cache.get(agent, [])

    def degree(self, agent: AgentId) -> int:
        return len(self.neighbors(agent))

    def link_quality(self, u: AgentId, v: AgentId) -> float:
        edge = normalize_edge(u, v)
        return self.links[edge].quality if edge in self.links else 0.0


__all__ = ["LinkState", "SimulationState"]

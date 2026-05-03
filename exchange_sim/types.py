from __future__ import annotations

from typing import Callable, Dict, Hashable, Tuple

AgentId = Hashable
Edge = Tuple[AgentId, AgentId]
Allocation = Dict[Tuple[AgentId, AgentId], float]
UtilityFn = Callable[[AgentId, Dict[AgentId, float], "SimulationState"], float]

__all__ = [
    "AgentId",
    "Edge",
    "Allocation",
    "UtilityFn",
]

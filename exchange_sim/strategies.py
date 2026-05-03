from __future__ import annotations

from typing import Dict

from .state import SimulationState
from .types import AgentId


class Strategy:
    """Base interface for allocation strategies."""

    def allocate(self, state: SimulationState, agent: AgentId, budget: float) -> Dict[AgentId, float]:
        raise NotImplementedError


class ConstantStrategy(Strategy):
    """Split the available exchange budget equally among active neighbors."""

    def allocate(self, state: SimulationState, agent: AgentId, budget: float) -> Dict[AgentId, float]:
        neighbors = state.neighbors(agent)
        if not neighbors or budget <= 0:
            return {}
        share = budget / len(neighbors)
        return {j: share for j in neighbors}


class ProportionalStrategy(Strategy):
    """
    Give proportionally to what each neighbor gave the agent in the previous slot.

    Cold start rule:
    - If the agent has no positive previous received amounts, split equally.
    - This avoids the zero-flow lock-in of pure proportional response.
    """

    def allocate(self, state: SimulationState, agent: AgentId, budget: float) -> Dict[AgentId, float]:
        neighbors = state.neighbors(agent)
        if not neighbors or budget <= 0:
            return {}

        received_from = state.last_received.get(agent, {})
        weights = {j: max(received_from.get(j, 0.0), 0.0) for j in neighbors}
        total = sum(weights.values())

        if total <= 0:
            share = budget / len(neighbors)
            return {j: share for j in neighbors}

        return {j: budget * weights[j] / total for j in neighbors}


__all__ = [
    "Strategy",
    "ConstantStrategy",
    "ProportionalStrategy",
]

from __future__ import annotations

import math
from typing import Dict

from .state import SimulationState
from .types import AgentId
from .utils import normalize_edge


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


class ProportionalConstantBootstrapStrategy(ProportionalStrategy):
    """
    Proportional response with an equal-share bootstrap for new links.

    This addresses the lock-in case where a newly created link has previous flow
    x_ij(t) = 0, so pure proportional response would allocate 0 forever. A new
    link first receives budget / degree(agent), then the remaining budget is
    allocated proportionally over the other links.
    """

    def __init__(self, bootstrap_age: int = 0) -> None:
        if bootstrap_age < 0:
            raise ValueError("bootstrap_age must be non-negative.")
        self.bootstrap_age = bootstrap_age

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

        bootstrap_neighbors = [j for j in neighbors if self._link_age(state, agent, j) <= self.bootstrap_age]
        if not bootstrap_neighbors:
            return {j: budget * weights[j] / total for j in neighbors}

        exploration_share = budget / len(neighbors)
        exploration_budget = exploration_share * len(bootstrap_neighbors)
        remaining_budget = budget - exploration_budget
        remaining_neighbors = [j for j in neighbors if j not in bootstrap_neighbors]
        remaining_total = sum(weights[j] for j in remaining_neighbors)

        allocation = {j: 0.0 for j in neighbors}
        for j in bootstrap_neighbors:
            allocation[j] += exploration_share
        if remaining_neighbors:
            if remaining_total <= 0:
                remaining_share = remaining_budget / len(remaining_neighbors)
                for j in remaining_neighbors:
                    allocation[j] += remaining_share
            else:
                for j in remaining_neighbors:
                    allocation[j] += remaining_budget * weights[j] / remaining_total
        return allocation

    @staticmethod
    def _link_age(state: SimulationState, agent: AgentId, neighbor: AgentId) -> int:
        edge = normalize_edge(agent, neighbor)
        link = state.links[edge]
        return max(state.t - link.created_at, 0)


class ProportionalDecayingBoostStrategy(ProportionalStrategy):
    """
    Proportional response with exploration floor and temporary new-link boost.

    The simulation weight is:
    z_ij(t) = q_ij(t - 1) x_ji(t - 1) + epsilon
              + lambda * 1{age_ij(t) < L} * exp(-alpha * age_ij(t)).
    """

    def __init__(
        self,
        exploration_floor: float = 1e-9,
        boost_lambda: float = 1.0,
        boost_alpha: float = 0.5,
        boost_lifetime: int = 5,
    ) -> None:
        if exploration_floor < 0:
            raise ValueError("exploration_floor must be non-negative.")
        if boost_lambda < 0:
            raise ValueError("boost_lambda must be non-negative.")
        if boost_alpha < 0:
            raise ValueError("boost_alpha must be non-negative.")
        if boost_lifetime < 0:
            raise ValueError("boost_lifetime must be non-negative.")
        self.exploration_floor = exploration_floor
        self.boost_lambda = boost_lambda
        self.boost_alpha = boost_alpha
        self.boost_lifetime = boost_lifetime

    def allocate(self, state: SimulationState, agent: AgentId, budget: float) -> Dict[AgentId, float]:
        neighbors = state.neighbors(agent)
        if not neighbors or budget <= 0:
            return {}

        received_from = state.last_received.get(agent, {})
        weights = {
            j: max(received_from.get(j, 0.0), 0.0)
            + self.exploration_floor
            + self._decaying_boost(state, agent, j)
            for j in neighbors
        }
        total = sum(weights.values())

        if total <= 0:
            share = budget / len(neighbors)
            return {j: share for j in neighbors}

        return {j: budget * weights[j] / total for j in neighbors}

    def _decaying_boost(self, state: SimulationState, agent: AgentId, neighbor: AgentId) -> float:
        edge = normalize_edge(agent, neighbor)
        link = state.links[edge]
        age = max(state.t - link.created_at, 0)
        if age >= self.boost_lifetime:
            return 0.0
        return self.boost_lambda * math.exp(-self.boost_alpha * age)


__all__ = [
    "Strategy",
    "ConstantStrategy",
    "ProportionalStrategy",
    "ProportionalConstantBootstrapStrategy",
    "ProportionalDecayingBoostStrategy",
]

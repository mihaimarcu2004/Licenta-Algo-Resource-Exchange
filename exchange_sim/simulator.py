from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Union
import math
import random

from .state import LinkState, SimulationState
from .strategies import ConstantStrategy, ProportionalStrategy, Strategy
from .types import AgentId, Allocation, Edge, UtilityFn
from .utils import normalize_edge


class DecentralizedExchangeSimulator:
    """
    Simulator for resource exchange with expiring/degrading links.

    Main modeling choices:
    - Agents produce `production[i]` units each timeslot.
    - Active links allow exchange in both directions.
    - Initial links are permanent.
    - Created links have a cost, an expiry lifetime, and a degradation factor (per-edge or default).
    - Link creation is unilateral: the endpoint with positive best gain pays the cost.
    - Links can be refreshed when quality falls below a threshold (optional).
    - Link creation is evaluated from one snapshot per slot, then profitable actions are
      applied together. This removes order effects during candidate selection.
    """

    def __init__(
        self,
        agents: Iterable[AgentId],
        production: Dict[AgentId, float],
        initial_edges: Iterable[Edge],
        strategy: Union[str, Strategy] = "constant",
        utility_formula: Optional[UtilityFn] = None,
        edge_costs: Optional[Dict[Edge, float]] = None,
        expiry: int = 5,
        degradation: float = 1.0,
        edge_lifetimes: Optional[Dict[Edge, Optional[int]]] = None,
        edge_degradations: Optional[Dict[Edge, float]] = None,
        refresh_threshold: Optional[float] = 0.8,
        discount: float = 0.95,
        max_new_links_per_slot: Optional[int] = None,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.agents = list(agents)
        self.production = dict(production)
        self.edge_costs = {normalize_edge(*e): c for e, c in (edge_costs or {}).items()}
        self.expiry = expiry
        self.degradation = degradation
        self.edge_lifetimes = {normalize_edge(*e): v for e, v in (edge_lifetimes or {}).items()}
        self.edge_degradations = {normalize_edge(*e): v for e, v in (edge_degradations or {}).items()}
        self.refresh_threshold = refresh_threshold
        self.discount = discount
        self.max_new_links_per_slot = max_new_links_per_slot
        self.rng = random.Random(rng_seed)

        missing = set(self.agents) - set(self.production)
        if missing:
            raise ValueError(f"Missing production values for agents: {missing}")
        if expiry <= 0:
            raise ValueError("expiry must be positive.")
        if not 0 < degradation <= 1:
            raise ValueError("degradation must be in (0, 1].")
        if any(v is not None and v <= 0 for v in self.edge_lifetimes.values()):
            raise ValueError("All edge lifetimes must be positive or None.")
        if any(not 0 < v <= 1 for v in self.edge_degradations.values()):
            raise ValueError("All edge degradations must be in (0, 1].")
        if refresh_threshold is not None and not 0 < refresh_threshold <= 1:
            raise ValueError("refresh_threshold must be in (0, 1] or None.")
        if not 0 <= discount < 1:
            raise ValueError("discount must be in [0, 1).")
        if max_new_links_per_slot is not None and max_new_links_per_slot <= 0:
            raise ValueError("max_new_links_per_slot must be positive or None.")

        self.strategy = self._make_strategy(strategy)
        self.utility_formula = utility_formula or self.linear_received_utility

        links: Dict[Edge, LinkState] = {}
        for u, v in initial_edges:
            edge = normalize_edge(u, v)
            links[edge] = LinkState(
                u=edge[0],
                v=edge[1],
                remaining_life=None,
                quality=1.0,
                degradation=self.degradation,
                initial=True,
                created_at=0,
            )

        self.state = SimulationState(
            t=0,
            agents=self.agents,
            production=self.production,
            links=links,
            last_received={i: {} for i in self.agents},
            last_creation_costs={i: 0.0 for i in self.agents},
            exchange_ratios={i: 1.0 for i in self.agents},
            utilities={i: 0.0 for i in self.agents},
        )
        self.history: List[SimulationState] = []

    def _make_strategy(self, strategy: Union[str, Strategy]) -> Strategy:
        if isinstance(strategy, Strategy):
            return strategy
        if strategy == "constant":
            return ConstantStrategy()
        if strategy == "proportional":
            return ProportionalStrategy()
        raise ValueError(f"Unknown strategy: {strategy!r}")

    @staticmethod
    def linear_received_utility(agent: AgentId, received_from: Dict[AgentId, float], state: SimulationState) -> float:
        """Default utility: total quality-adjusted resource received."""
        return sum(received_from.values())

    def active_edges(self) -> List[Edge]:
        return list(self.state.links.keys())

    def neighbors(self, agent: AgentId) -> List[AgentId]:
        return self.state.neighbors(agent)

    def candidate_edges(self) -> List[Edge]:
        """Possible links are all edges listed in edge_costs/lifetimes/degradations that are not active."""
        active = set(self.state.links)
        candidates = set(self.edge_costs) | set(self.edge_lifetimes) | set(self.edge_degradations)
        return [edge for edge in candidates if edge not in active]

    def refresh_candidates(self) -> List[Edge]:
        if self.refresh_threshold is None:
            return []
        return [
            edge
            for edge, link in self.state.links.items()
            if link.quality <= self.refresh_threshold
        ]

    def run(self, steps: int) -> List[SimulationState]:
        for _ in range(steps):
            self.step()
        return self.history

    def is_full_allocation(self, state: Optional[SimulationState] = None, eps: float = 1e-9) -> bool:
        """
        Check whether each agent allocates its full exchange budget in a given state.

        Full budget is production minus link-creation costs for that step (clamped at zero).
        """
        st = state or self.state
        outgoing: Dict[AgentId, float] = {i: 0.0 for i in self.agents}
        for (sender, _), amount in st.last_allocation.items():
            outgoing[sender] = outgoing.get(sender, 0.0) + amount

        for i in self.agents:
            budget = max(self.production[i] - st.last_creation_costs.get(i, 0.0), 0.0)
            if abs(outgoing.get(i, 0.0) - budget) > eps:
                return False
        return True

    def find_equilibrium(
        self,
        max_steps: int = 100000,
        precision: int = 9,
        eps: float = 1e-9,
    ) -> Optional[Tuple[int, int]]:
        result = self.find_equilibrium_states(max_steps=max_steps, precision=precision, eps=eps)
        if result is None:
            return None
        start_time, period, _ = result
        return (start_time, period)

    def find_equilibrium_states(
        self,
        max_steps: int = 100000,
        precision: int = 9,
        eps: float = 1e-9,
    ) -> Optional[Tuple[int, int, List[SimulationState]]]:
        """
        Advance the simulation until a periodic state is detected.

        Returns (start_time, period, cycle_states) if the detected cycle satisfies full allocation
        for all states in the cycle. Returns None if no cycle is found within max_steps or if the
        detected cycle violates the full-allocation condition.
        """
        seen: Dict[Tuple[object, ...], int] = {}
        states: List[SimulationState] = []
        t0 = self.state.t

        for step_idx in range(max_steps + 1):
            sig = self._state_signature(self.state, precision)
            if sig in seen:
                start_idx = seen[sig]
                period = step_idx - start_idx
                cycle_states = states[start_idx:step_idx]
                if all(self.is_full_allocation(s, eps) for s in cycle_states):
                    return (t0 + start_idx, period, cycle_states)
                return None

            seen[sig] = step_idx
            states.append(self.state)
            if step_idx == max_steps:
                break
            self.step()

        return None

    def step(self) -> SimulationState:
        """
        One timeslot:
        1. Age/degrade links and remove expired links.
        2. Evaluate profitable link actions from the current snapshot and apply them together.
        3. Allocate resources using the selected strategy.
        4. Compute received resources, utilities, and exchange ratios.
        """
        self._age_and_remove_expired_links()
        _, creation_costs = self._create_profitable_links()
        allocation = self._compute_allocations(creation_costs)
        received = self._compute_received(allocation)
        utilities = self._compute_utilities(received)
        ratios = self._compute_exchange_ratios(received, creation_costs)

        self.state = SimulationState(
            t=self.state.t + 1,
            agents=self.agents,
            production=self.production,
            links={edge: LinkState(**vars(link)) for edge, link in self.state.links.items()},
            last_allocation=allocation,
            last_creation_costs=creation_costs,
            last_received=received,
            exchange_ratios=ratios,
            utilities=utilities,
        )
        self.history.append(self.state)
        return self.state

    def _age_and_remove_expired_links(self) -> None:
        to_delete: List[Edge] = []
        for edge, link in self.state.links.items():
            link.quality *= link.degradation
            if link.remaining_life is not None:
                link.remaining_life -= 1
                if link.remaining_life <= 0:
                    to_delete.append(edge)
        for edge in to_delete:
            del self.state.links[edge]

    def _compute_allocations(self, creation_costs: Dict[AgentId, float]) -> Allocation:
        allocation: Allocation = {}
        for i in self.agents:
            exchange_budget = max(self.production[i] - creation_costs.get(i, 0.0), 0.0)
            out = self.strategy.allocate(self.state, i, exchange_budget)
            for j, amount in out.items():
                if amount <= 0:
                    continue
                if normalize_edge(i, j) not in self.state.links:
                    continue
                allocation[(i, j)] = amount
        return allocation

    def _compute_received(self, allocation: Allocation) -> Dict[AgentId, Dict[AgentId, float]]:
        received: Dict[AgentId, Dict[AgentId, float]] = {i: {} for i in self.agents}
        for (sender, receiver), amount in allocation.items():
            quality = self.state.link_quality(sender, receiver)
            effective_amount = amount * quality
            received[receiver][sender] = received[receiver].get(sender, 0.0) + effective_amount
        return received

    def _compute_utilities(self, received: Dict[AgentId, Dict[AgentId, float]]) -> Dict[AgentId, float]:
        return {i: self.utility_formula(i, received.get(i, {}), self.state) for i in self.agents}

    def _compute_exchange_ratios(
        self,
        received: Dict[AgentId, Dict[AgentId, float]],
        creation_costs: Dict[AgentId, float],
    ) -> Dict[AgentId, float]:
        ratios: Dict[AgentId, float] = {}
        for i in self.agents:
            received_total = sum(received.get(i, {}).values())
            contributed = max(self.production[i], 1e-9)
            ratios[i] = received_total / contributed
        return ratios

    def _create_profitable_links(self) -> Tuple[List[Edge], Dict[AgentId, float]]:
        creation_costs: Dict[AgentId, float] = {i: 0.0 for i in self.agents}
        actions: List[Tuple[float, str, Edge, AgentId]] = []

        for edge in self.candidate_edges():
            u, v = edge
            cost = self.edge_costs.get(edge, 0.0)
            lifetime = self.edge_lifetimes.get(edge, self.expiry)
            degradation = self.edge_degradations.get(edge, self.degradation)

            gain_u = self.estimate_link_gain(u, v, cost, lifetime, degradation)
            gain_v = self.estimate_link_gain(v, u, cost, lifetime, degradation)

            if gain_u >= gain_v:
                proposer = u
                score = gain_u
            else:
                proposer = v
                score = gain_v

            if score > 0:
                actions.append((score, "new", edge, proposer))

        for edge in self.refresh_candidates():
            link = self.state.links[edge]
            u, v = edge
            cost = self.edge_costs.get(edge, 0.0)
            lifetime = self.edge_lifetimes.get(edge, self.expiry)
            degradation = self.edge_degradations.get(edge, link.degradation)

            gain_u = self.estimate_refresh_gain(u, v, cost, lifetime, degradation)
            gain_v = self.estimate_refresh_gain(v, u, cost, lifetime, degradation)

            if gain_u >= gain_v:
                proposer = u
                score = gain_u
            else:
                proposer = v
                score = gain_v

            if score > 0:
                actions.append((score, "refresh", edge, proposer))

        actions.sort(key=lambda item: (-item[0], str(item[2]), str(item[3]), item[1]))

        selected_actions: List[Tuple[float, str, Edge, AgentId]] = []
        for action in actions:
            _, _, edge, proposer = action
            cost = self.edge_costs.get(edge, 0.0)
            if creation_costs[proposer] + cost > self.production[proposer]:
                continue
            selected_actions.append(action)
            creation_costs[proposer] += cost
            if self.max_new_links_per_slot is not None and len(selected_actions) >= self.max_new_links_per_slot:
                break

        created: List[Edge] = []
        for _, action_kind, edge, _ in selected_actions:
            u, v = edge
            lifetime = self.edge_lifetimes.get(edge, self.expiry)

            if action_kind == "refresh":
                if edge not in self.state.links:
                    continue
                link = self.state.links[edge]
                degradation = self.edge_degradations.get(edge, link.degradation)
                link.remaining_life = lifetime
                link.quality = 1.0
                link.degradation = degradation
                link.created_at = self.state.t
            else:
                if edge in self.state.links:
                    continue
                degradation = self.edge_degradations.get(edge, self.degradation)
                self.state.links[edge] = LinkState(
                    u=u,
                    v=v,
                    remaining_life=lifetime,
                    quality=1.0,
                    degradation=degradation,
                    initial=False,
                    created_at=self.state.t,
                )
            created.append(edge)

        return created, creation_costs

    def estimate_link_gain(
        self,
        agent: AgentId,
        new_neighbor: AgentId,
        paid_cost: float,
        lifetime: Optional[int],
        degradation: float,
    ) -> float:
        """
        Estimate whether adding one link is useful for `agent` under the current strategy.

        This is a local, one-step approximation:
        - compare expected received amount before vs. after adding the link;
        - convert the one-step gain into a discounted lifetime value;
        - subtract the cost paid by this agent.

        Candidate links are scored against the same pre-creation snapshot for the slot,
        so simultaneous additions do not rescore each other during selection.
        """
        if paid_cost > self.production[agent]:
            return -math.inf

        before = self._expected_one_step_received(agent)

        edge = normalize_edge(agent, new_neighbor)
        if edge in self.state.links:
            return -math.inf

        # Temporarily add the candidate link.
        self.state.links[edge] = LinkState(
            u=edge[0],
            v=edge[1],
            remaining_life=lifetime,
            quality=1.0,
            degradation=degradation,
            initial=False,
            created_at=self.state.t,
        )
        after = self._expected_one_step_received(agent)
        del self.state.links[edge]

        one_step_gain = after - before
        factor = self.discount * degradation
        if lifetime is None:
            discounted_lifetime_multiplier = factor / (1 - factor) if factor > 0 else 0.0
        else:
            discounted_lifetime_multiplier = sum((factor) ** s for s in range(1, lifetime + 1))
        return -paid_cost + discounted_lifetime_multiplier * one_step_gain

    def estimate_refresh_gain(
        self,
        agent: AgentId,
        neighbor: AgentId,
        paid_cost: float,
        lifetime: Optional[int],
        degradation: float,
    ) -> float:
        if paid_cost > self.production[agent]:
            return -math.inf

        edge = normalize_edge(agent, neighbor)
        if edge not in self.state.links:
            return -math.inf

        before = self._expected_one_step_received(agent)

        # Temporarily refresh the link.
        current = self.state.links[edge]
        saved = LinkState(**vars(current))
        current.remaining_life = lifetime
        current.quality = 1.0
        current.degradation = degradation
        after = self._expected_one_step_received(agent)
        self.state.links[edge] = saved

        one_step_gain = after - before
        factor = self.discount * degradation
        if lifetime is None:
            discounted_lifetime_multiplier = factor / (1 - factor) if factor > 0 else 0.0
        else:
            discounted_lifetime_multiplier = sum((factor) ** s for s in range(1, lifetime + 1))
        return -paid_cost + discounted_lifetime_multiplier * one_step_gain

    def _expected_one_step_received(self, agent: AgentId) -> float:
        """
        Predict next-slot received resource if every agent uses the strategy once.

        This is used only for evaluating candidate links. It is intentionally myopic and local-ish.
        """
        predicted_allocation: Allocation = {}
        for i in self.agents:
            out = self.strategy.allocate(self.state, i, self.production[i])
            for j, amount in out.items():
                if amount > 0 and normalize_edge(i, j) in self.state.links:
                    predicted_allocation[(i, j)] = amount

        received = 0.0
        for (sender, receiver), amount in predicted_allocation.items():
            if receiver == agent:
                received += amount * self.state.link_quality(sender, receiver)
        return received

    def summary(self) -> Dict[str, object]:
        return {
            "t": self.state.t,
            "active_edges": self.active_edges(),
            "exchange_ratios": dict(self.state.exchange_ratios),
            "utilities": dict(self.state.utilities),
            "last_allocation": dict(self.state.last_allocation),
        }

    def _state_signature(self, state: SimulationState, precision: int) -> Tuple[object, ...]:
        def r(value: float) -> float:
            return round(value, precision)

        links_sig = [
            (
                edge,
                link.remaining_life,
                r(link.quality),
                r(link.degradation),
                link.initial,
            )
            for edge, link in state.links.items()
        ]
        links_sig = tuple(sorted(links_sig, key=lambda item: str(item[0])))

        allocations_sig = [
            (((sender, receiver)), r(amount)) for (sender, receiver), amount in state.last_allocation.items()
        ]
        allocations_sig = tuple(sorted(allocations_sig, key=lambda item: str(item[0])))

        creation_costs_sig = [(agent, r(cost)) for agent, cost in state.last_creation_costs.items()]
        creation_costs_sig = tuple(sorted(creation_costs_sig, key=lambda item: str(item[0])))

        last_received_sig = [
            (
                agent,
                tuple(
                    sorted(((src, r(val)) for src, val in received.items()), key=lambda item: str(item[0]))
                ),
            )
            for agent, received in state.last_received.items()
        ]
        last_received_sig = tuple(sorted(last_received_sig, key=lambda item: str(item[0])))

        ratios_sig = [(agent, r(rho)) for agent, rho in state.exchange_ratios.items()]
        ratios_sig = tuple(sorted(ratios_sig, key=lambda item: str(item[0])))

        return (links_sig, allocations_sig, creation_costs_sig, last_received_sig, ratios_sig)


__all__ = ["DecentralizedExchangeSimulator"]

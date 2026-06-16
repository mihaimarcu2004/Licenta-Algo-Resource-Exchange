from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Union
import math
import random

from .heuristics import (
    EV_HEURISTIC_ADAPTIVE_HORIZON,
    EV_HEURISTIC_CAP_HORIZON_20,
    EV_HEURISTIC_CAP_HORIZON_50,
    normalize_ev_heuristics,
)
from .state import LinkState, SimulationState
from .strategies import (
    ConstantStrategy,
    ProportionalConstantBootstrapStrategy,
    ProportionalDecayingBoostStrategy,
    ProportionalStrategy,
    Strategy,
)
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
        renewal_costs: Optional[Dict[Edge, float]] = None,
        expiry: int = 5,
        degradation: float = 1.0,
        edge_lifetimes: Optional[Dict[Edge, Optional[int]]] = None,
        edge_degradations: Optional[Dict[Edge, float]] = None,
        edge_discounts: Optional[Dict[Edge, float]] = None,
        refresh_threshold: Optional[float] = 0.8,
        renewal_life_threshold: Optional[int] = 1,
        min_link_flow: float = 1e-9,
        discount: float = 0.95,
        ev_heuristics: Optional[Iterable[str]] = None,
        ev_adaptive_epsilon: float = 0.01,
        link_estimation_mode: str = "formula",
        max_candidate_edges_per_agent: Optional[int] = None,
        max_new_links_per_slot: Optional[int] = None,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.agents = list(agents)
        self.production = dict(production)
        self.edge_costs = {normalize_edge(*e): c for e, c in (edge_costs or {}).items()}
        self.renewal_costs = {
            normalize_edge(*e): c for e, c in (renewal_costs if renewal_costs is not None else self.edge_costs).items()
        }
        self.expiry = expiry
        self.degradation = degradation
        self.edge_lifetimes = {normalize_edge(*e): v for e, v in (edge_lifetimes or {}).items()}
        self.edge_degradations = {normalize_edge(*e): v for e, v in (edge_degradations or {}).items()}
        self.edge_discounts = {normalize_edge(*e): v for e, v in (edge_discounts or {}).items()}
        self.candidate_edge_universe = (
            set(self.edge_costs)
            | set(self.edge_lifetimes)
            | set(self.edge_degradations)
            | set(self.edge_discounts)
        )
        self.refresh_threshold = refresh_threshold
        self.renewal_life_threshold = renewal_life_threshold
        self.min_link_flow = min_link_flow
        self.discount = discount
        self.ev_heuristics = normalize_ev_heuristics(ev_heuristics)
        self.ev_adaptive_epsilon = ev_adaptive_epsilon
        self.link_estimation_mode = link_estimation_mode
        self.max_candidate_edges_per_agent = max_candidate_edges_per_agent
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
        if any(not 0 <= v < 1 for v in self.edge_discounts.values()):
            raise ValueError("All edge discounts must be in [0, 1).")
        if refresh_threshold is not None and not 0 < refresh_threshold <= 1:
            raise ValueError("refresh_threshold must be in (0, 1] or None.")
        if renewal_life_threshold is not None and renewal_life_threshold < 0:
            raise ValueError("renewal_life_threshold must be non-negative or None.")
        if min_link_flow < 0:
            raise ValueError("min_link_flow must be non-negative.")
        if not 0 <= discount < 1:
            raise ValueError("discount must be in [0, 1).")
        if ev_adaptive_epsilon < 0:
            raise ValueError("ev_adaptive_epsilon must be non-negative.")
        if link_estimation_mode != "formula":
            raise ValueError("link_estimation_mode must be 'formula'.")
        if max_candidate_edges_per_agent is not None and max_candidate_edges_per_agent <= 0:
            raise ValueError("max_candidate_edges_per_agent must be positive or None.")
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
            last_renewal_costs={i: 0.0 for i in self.agents},
            last_renewals={},
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
        if strategy == "proportional_constant_bootstrap":
            return ProportionalConstantBootstrapStrategy()
        if strategy == "proportional_decaying_boost":
            return ProportionalDecayingBoostStrategy()
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
        """Possible links are configured non-active edges."""
        active = set(self.state.links)
        candidates = [edge for edge in self.candidate_edge_universe if edge not in active]
        if self.max_candidate_edges_per_agent is None:
            return candidates
        return self._top_sharing_ratio_candidate_edges_per_agent(candidates)

    def _top_sharing_ratio_candidate_edges_per_agent(self, candidates: List[Edge]) -> List[Edge]:
        candidate_set = set(candidates)
        selected: List[Edge] = []
        selected_set: set[Edge] = set()
        ranked_agents = sorted(
            self.agents,
            key=lambda agent: (-self.state.exchange_ratios.get(agent, 0.0), str(agent)),
        )

        for agent in ranked_agents:
            incident = [
                edge
                for edge in candidate_set
                if edge not in selected_set and (edge[0] == agent or edge[1] == agent)
            ]
            incident.sort(
                key=lambda edge: (
                    -max(
                        self.state.exchange_ratios.get(edge[0], 0.0),
                        self.state.exchange_ratios.get(edge[1], 0.0),
                    ),
                    str(edge),
                )
            )
            for edge in incident[: self.max_candidate_edges_per_agent]:
                if edge not in selected_set:
                    selected.append(edge)
                    selected_set.add(edge)

        return selected

    def refresh_candidates(self) -> List[Edge]:
        if self.refresh_threshold is None:
            quality_candidates = set()
        else:
            quality_candidates = {
                edge
                for edge, link in self.state.links.items()
                if link.quality <= self.refresh_threshold
            }
        expiry_candidates = {
            edge
            for edge, link in self.state.links.items()
            if (
                self.renewal_life_threshold is not None
                and link.remaining_life is not None
                and link.remaining_life <= self.renewal_life_threshold
            )
        }
        return [
            edge
            for edge in self.state.links
            if edge in quality_candidates or edge in expiry_candidates
        ]

    def run(self, steps: int) -> List[SimulationState]:
        for _ in range(steps):
            self.step()
        return self.history

    def is_full_allocation(self, state: Optional[SimulationState] = None, eps: float = 1e-9) -> bool:
        """
        Check whether each agent allocates its full exchange budget in a given state.

        Link decisions are applied after the current allocation, so the exchange
        budget for the allocation itself is the full production amount.
        """
        st = state or self.state
        outgoing: Dict[AgentId, float] = {i: 0.0 for i in self.agents}
        for (sender, _), amount in st.last_allocation.items():
            outgoing[sender] = outgoing.get(sender, 0.0) + amount

        for i in self.agents:
            budget = self.production[i]
            if abs(outgoing.get(i, 0.0) - budget) > eps:
                return False
        return True

    def find_equilibrium(
        self,
        max_steps: int = 1000000000,
        precision: int = 9,
        eps: float = 1e-9,
        min_steps: int = 0,
    ) -> Optional[Tuple[int, int]]:
        result = self.find_equilibrium_states(
            max_steps=max_steps,
            precision=precision,
            eps=eps,
            min_steps=min_steps,
        )
        if result is None:
            return None
        start_time, period, _ = result
        return (start_time, period)

    def find_equilibrium_states(
        self,
        max_steps: int = 1000000000,
        precision: int = 9,
        eps: float = 1e-9,
        min_steps: int = 0,
    ) -> Optional[Tuple[int, int, List[SimulationState]]]:
        """
        Advance the simulation until a periodic state is detected.

        Returns (start_time, period, cycle_states) when the hashed state repeats.
        The detected cycle must satisfy full allocation for all states in the cycle.
        """
        if min_steps < 0:
            raise ValueError("min_steps must be non-negative.")

        seen: Dict[Tuple[object, ...], int] = {}
        states: List[SimulationState] = []
        t0 = self.state.t

        for step_idx in range(max_steps + 1):
            sig = self._state_signature(self.state, precision)
            if sig in seen:
                if step_idx >= min_steps:
                    start_idx = seen[sig]
                    period = step_idx - start_idx
                    cycle_states = states[start_idx:step_idx]
                    if all(self.is_full_allocation(s, eps) for s in cycle_states):
                        return (t0 + start_idx, period, cycle_states)
                    return None
                states.append(self.state)
                seen[sig] = step_idx
                if step_idx == max_steps:
                    break
                self.step()
                continue

            seen[sig] = step_idx
            states.append(self.state)
            if step_idx == max_steps:
                break
            self.step()

        return None

    def step(self) -> SimulationState:
        """
        One timeslot:
        1. Age/degrade links.
        2. Allocate resources using the selected strategy.
        3. Compute received resources, utilities, and exchange ratios.
        4. Evaluate renewal and new-link decisions from that snapshot.
        5. Apply all accepted link decisions simultaneously for the next slot.
        """
        self._age_links()
        allocation = self._compute_allocations({})
        received = self._compute_received(allocation)
        self._remove_inactive_links(allocation)
        utilities = self._compute_utilities(received)
        ratios = self._compute_exchange_ratios(received, {})

        self.state.last_allocation = allocation
        self.state.last_received = received
        self.state.utilities = utilities
        self.state.exchange_ratios = ratios
        _, action_costs, renewal_costs, renewals = self._create_profitable_links()
        self._remove_expired_links()

        self.state = SimulationState(
            t=self.state.t + 1,
            agents=self.agents,
            production=self.production,
            links={edge: LinkState(**vars(link)) for edge, link in self.state.links.items()},
            last_allocation=allocation,
            last_creation_costs=action_costs,
            last_renewal_costs=renewal_costs,
            last_renewals=renewals,
            last_received=received,
            exchange_ratios=ratios,
            utilities=utilities,
        )
        self.history.append(self.state)
        return self.state

    def _age_links(self) -> None:
        for edge, link in self.state.links.items():
            if link.created_at >= self.state.t:
                continue
            link.quality *= link.degradation
            if link.remaining_life is not None:
                link.remaining_life -= 1

    def _remove_expired_links(self) -> None:
        to_delete: List[Edge] = []
        for edge, link in self.state.links.items():
            if link.remaining_life is not None and link.remaining_life <= 0:
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

    def _remove_inactive_links(self, allocation: Allocation) -> None:
        if self.min_link_flow <= 0:
            return

        effective_flow: Dict[Edge, float] = {edge: 0.0 for edge in self.state.links}
        for (sender, receiver), amount in allocation.items():
            edge = normalize_edge(sender, receiver)
            if edge not in self.state.links:
                continue
            effective_flow[edge] += amount * self.state.link_quality(sender, receiver)

        to_delete = [
            edge
            for edge, flow in effective_flow.items()
            if not self.state.links[edge].initial and flow < self.min_link_flow
        ]
        for edge in to_delete:
            del self.state.links[edge]

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

    def _create_profitable_links(
        self,
    ) -> Tuple[List[Edge], Dict[AgentId, float], Dict[AgentId, float], Dict[Tuple[AgentId, AgentId], int]]:
        action_costs: Dict[AgentId, float] = {i: 0.0 for i in self.agents}
        renewal_costs: Dict[AgentId, float] = {i: 0.0 for i in self.agents}
        renewals: Dict[Tuple[AgentId, AgentId], int] = {}
        actions: List[Tuple[float, str, Edge, AgentId]] = []

        actions.extend(self._new_link_actions())

        for edge in self.refresh_candidates():
            link = self.state.links[edge]
            u, v = edge
            cost = self.renewal_costs.get(edge, self.edge_costs.get(edge, 0.0))
            lifetime = self.edge_lifetimes.get(edge, self.expiry)
            degradation = self.edge_degradations.get(edge, link.degradation)
            discount = self.edge_discounts.get(edge, self.discount)

            gain_u = self.estimate_refresh_gain(u, v, cost, lifetime, degradation, discount)
            gain_v = self.estimate_refresh_gain(v, u, cost, lifetime, degradation, discount)

            if gain_u >= gain_v:
                proposer = u
                score = gain_u
            else:
                proposer = v
                score = gain_v

            if score >= 0:
                actions.append((score, "refresh", edge, proposer))

        actions.sort(key=lambda item: (-item[0], str(item[2]), str(item[3]), item[1]))

        selected_actions: List[Tuple[float, str, Edge, AgentId]] = []
        for action in actions:
            _, _, edge, proposer = action
            action_kind = action[1]
            if action_kind == "refresh":
                cost = self.renewal_costs.get(edge, self.edge_costs.get(edge, 0.0))
            else:
                cost = self.edge_costs.get(edge, 0.0)
            if action_costs[proposer] + cost > self.production[proposer]:
                continue
            selected_actions.append(action)
            action_costs[proposer] += cost
            if action_kind == "refresh":
                renewal_costs[proposer] += cost
            if self.max_new_links_per_slot is not None and len(selected_actions) >= self.max_new_links_per_slot:
                break

        created: List[Edge] = []
        for _, action_kind, edge, proposer in selected_actions:
            u, v = edge
            lifetime = self.edge_lifetimes.get(edge, self.expiry)

            if action_kind == "refresh":
                if edge not in self.state.links:
                    continue
                link = self.state.links[edge]
                renewals[(proposer, link.other(proposer))] = 1
                degradation = self.edge_degradations.get(edge, link.degradation)
                link.remaining_life = lifetime
                link.quality = 1.0
                link.degradation = degradation
                link.created_at = self.state.t + 1
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
                    created_at=self.state.t + 1,
                )
            created.append(edge)

        return created, action_costs, renewal_costs, renewals

    def _new_link_actions(self) -> List[Tuple[float, str, Edge, AgentId]]:
        actions: List[Tuple[float, str, Edge, AgentId]] = []
        for edge in self.candidate_edges():
            action = self._evaluate_candidate_edge_action(edge)
            if action is not None:
                actions.append(action)
        return actions

    def _evaluate_candidate_edge_action(self, edge: Edge) -> Optional[Tuple[float, str, Edge, AgentId]]:
        u, v = edge
        cost = self.edge_costs.get(edge, 0.0)
        lifetime = self.edge_lifetimes.get(edge, self.expiry)
        degradation = self.edge_degradations.get(edge, self.degradation)
        discount = self.edge_discounts.get(edge, self.discount)

        gain_u = self.estimate_link_gain(u, v, cost, lifetime, degradation, discount)
        gain_v = self.estimate_link_gain(v, u, cost, lifetime, degradation, discount)

        if gain_u >= gain_v:
            proposer = u
            score = gain_u
        else:
            proposer = v
            score = gain_v

        if score >= 0:
            return (score, "new", edge, proposer)
        return None

    def _adaptive_horizon(
        self,
        horizon: int,
        neighbor: AgentId,
        cost: float,
        degradation: float,
        discount: Optional[float],
    ) -> int:
        if EV_HEURISTIC_ADAPTIVE_HORIZON not in self.ev_heuristics:
            return horizon
        if cost <= 0:
            return horizon

        time_discount = self.discount if discount is None else discount
        factor = time_discount * degradation
        if factor <= 0 or factor >= 1:
            return horizon

        for periods in range(1, horizon + 1):
            remaining_upper_bound = (
                self.production[neighbor]
                * (time_discount ** (periods + 1))
                * (degradation ** periods)
                / (1 - factor)
            )
            if remaining_upper_bound < self.ev_adaptive_epsilon * cost:
                return periods
        return horizon

    def estimate_link_gain(
        self,
        agent: AgentId,
        new_neighbor: AgentId,
        paid_cost: float,
        lifetime: Optional[int],
        degradation: float,
        discount: Optional[float] = None,
    ) -> float:
        """
        Estimate whether adding one link is useful for `agent` under the current strategy.

        New-link estimation uses the closed-form sharing-ratio estimator over the
        candidate lifetime/horizon.
        """
        if paid_cost > self.production[agent]:
            return -math.inf

        edge = normalize_edge(agent, new_neighbor)
        if edge in self.state.links:
            return -math.inf

        expected_benefit = self._formula_link_benefit(
            agent,
            new_neighbor,
            paid_cost,
            lifetime,
            degradation,
            discount,
        )
        return expected_benefit - paid_cost

    def _formula_link_benefit(
        self,
        agent: AgentId,
        new_neighbor: AgentId,
        paid_cost: float,
        lifetime: Optional[int],
        degradation: float,
        discount: Optional[float],
    ) -> float:
        """
        Closed-form sharing-ratio estimate for a candidate link.

        E_ij = x_ij / (rho_j + eps) * sum_{s=0}^{H-1} (delta * beta)^s
        """
        horizon = self._expected_utility_horizon(lifetime)
        time_discount = self.discount if discount is None else discount
        amount_given = self._candidate_initial_allocation(agent, new_neighbor)
        if amount_given <= 0 or horizon <= 0:
            return 0.0

        rho = self.state.exchange_ratios.get(new_neighbor)
        if rho is None:
            received_total = sum(self.state.last_received.get(new_neighbor, {}).values())
            rho = received_total / max(self.production[new_neighbor], 1e-9)

        ratio_epsilon = 1e-9
        factor = time_discount * degradation
        if math.isclose(factor, 1.0):
            multiplier = float(horizon)
        else:
            multiplier = (1 - (factor ** horizon)) / (1 - factor)
        return (amount_given / (rho + ratio_epsilon)) * multiplier

    def _candidate_initial_allocation(self, agent: AgentId, new_neighbor: AgentId) -> float:
        neighbors = self.state.neighbors(agent)
        if new_neighbor in neighbors:
            return 0.0

        budget = self.production[agent]
        degree_after = len(neighbors) + 1
        if budget <= 0 or degree_after <= 0:
            return 0.0

        if isinstance(self.strategy, ConstantStrategy):
            return budget / degree_after

        received_from = self.state.last_received.get(agent, {})
        if isinstance(self.strategy, ProportionalConstantBootstrapStrategy):
            total_weight = sum(max(received_from.get(j, 0.0), 0.0) for j in neighbors)
            if total_weight <= 0:
                return budget / degree_after
            return budget / degree_after

        if isinstance(self.strategy, ProportionalDecayingBoostStrategy):
            candidate_boost = self.strategy.boost_lambda if self.strategy.boost_lifetime > 0 else 0.0
            candidate_weight = self.strategy.exploration_floor + candidate_boost
            total_weight = candidate_weight
            for neighbor in neighbors:
                if normalize_edge(agent, neighbor) not in self.state.links:
                    continue
                total_weight += (
                    max(received_from.get(neighbor, 0.0), 0.0)
                    + self.strategy.exploration_floor
                    + self.strategy._decaying_boost(self.state, agent, neighbor)
                )
            if total_weight <= 0:
                return budget / degree_after
            return budget * candidate_weight / total_weight

        if isinstance(self.strategy, ProportionalStrategy):
            total_weight = sum(max(received_from.get(j, 0.0), 0.0) for j in neighbors)
            if total_weight <= 0:
                return budget / degree_after
            return 0.0

        return self._candidate_initial_allocation_fallback(agent, new_neighbor)

    def _candidate_initial_allocation_fallback(self, agent: AgentId, new_neighbor: AgentId) -> float:
        edge = normalize_edge(agent, new_neighbor)
        candidate_links = self._copy_links(self.state.links)
        candidate_links[edge] = LinkState(
            u=edge[0],
            v=edge[1],
            remaining_life=self.edge_lifetimes.get(edge, self.expiry),
            quality=1.0,
            degradation=self.edge_degradations.get(edge, self.degradation),
            initial=False,
            created_at=self.state.t + 1,
        )
        candidate_state = SimulationState(
            t=self.state.t + 1,
            agents=self.agents,
            production=self.production,
            links=candidate_links,
            last_received={
                receiver: dict(received)
                for receiver, received in self.state.last_received.items()
            },
            last_creation_costs={i: 0.0 for i in self.agents},
            last_renewal_costs={i: 0.0 for i in self.agents},
            last_renewals={},
            exchange_ratios=dict(self.state.exchange_ratios),
            utilities=dict(self.state.utilities),
        )
        allocation = self.strategy.allocate(candidate_state, agent, self.production[agent])
        return max(allocation.get(new_neighbor, 0.0), 0.0)

    def estimate_refresh_gain(
        self,
        agent: AgentId,
        neighbor: AgentId,
        paid_cost: float,
        lifetime: Optional[int],
        degradation: float,
        discount: Optional[float] = None,
    ) -> float:
        if paid_cost > self.production[agent]:
            return -math.inf

        edge = normalize_edge(agent, neighbor)
        if edge not in self.state.links:
            return -math.inf

        expected_gain = self._formula_refresh_benefit(
            agent,
            neighbor,
            lifetime,
            degradation,
            discount,
        )
        return expected_gain - paid_cost

    def _formula_refresh_benefit(
        self,
        agent: AgentId,
        neighbor: AgentId,
        lifetime: Optional[int],
        degradation: float,
        discount: Optional[float],
    ) -> float:
        edge = normalize_edge(agent, neighbor)
        link = self.state.links[edge]
        horizon = self._expected_utility_horizon(lifetime)
        if horizon <= 0:
            return 0.0

        existing_amount = self.state.last_allocation.get((agent, neighbor))
        if existing_amount is None:
            existing_amount = max(
                self.strategy.allocate(self.state, agent, self.production[agent]).get(neighbor, 0.0),
                0.0,
            )
        renewed_amount = self._renewed_initial_allocation(agent, neighbor, lifetime, degradation)

        rho = self.state.exchange_ratios.get(neighbor)
        if rho is None:
            received_total = sum(self.state.last_received.get(neighbor, {}).values())
            rho = received_total / max(self.production[neighbor], 1e-9)

        factor = (self.discount if discount is None else discount) * degradation
        if math.isclose(factor, 1.0):
            multiplier = float(horizon)
        else:
            multiplier = (1 - (factor ** horizon)) / (1 - factor)
        ratio_epsilon = 1e-9
        return ((renewed_amount - link.quality * existing_amount) / (rho + ratio_epsilon)) * multiplier

    def _renewed_initial_allocation(
        self,
        agent: AgentId,
        neighbor: AgentId,
        lifetime: Optional[int],
        degradation: float,
    ) -> float:
        edge = normalize_edge(agent, neighbor)
        renewed_links = self._copy_links(self.state.links)
        renewed = renewed_links[edge]
        renewed.remaining_life = lifetime
        renewed.quality = 1.0
        renewed.degradation = degradation
        renewed.created_at = self.state.t + 1
        renewed_state = SimulationState(
            t=self.state.t + 1,
            agents=self.agents,
            production=self.production,
            links=renewed_links,
            last_received={
                receiver: dict(received)
                for receiver, received in self.state.last_received.items()
            },
            last_creation_costs={i: 0.0 for i in self.agents},
            last_renewal_costs={i: 0.0 for i in self.agents},
            last_renewals={},
            exchange_ratios=dict(self.state.exchange_ratios),
            utilities=dict(self.state.utilities),
        )
        allocation = self.strategy.allocate(renewed_state, agent, self.production[agent])
        return max(allocation.get(neighbor, 0.0), 0.0)

    def _copy_links(self, links: Dict[Edge, LinkState]) -> Dict[Edge, LinkState]:
        return {edge: LinkState(**vars(link)) for edge, link in links.items()}

    def _expected_utility_horizon(self, lifetime: Optional[int]) -> int:
        if lifetime is None:
            horizon = 100
        else:
            horizon = max(int(lifetime), 1)

        if EV_HEURISTIC_CAP_HORIZON_20 in self.ev_heuristics:
            horizon = min(horizon, 20)
        if EV_HEURISTIC_CAP_HORIZON_50 in self.ev_heuristics:
            horizon = min(horizon, 50)
        return horizon

    def summary(self) -> Dict[str, object]:
        return {
            "t": self.state.t,
            "active_edges": self.active_edges(),
            "exchange_ratios": dict(self.state.exchange_ratios),
            "utilities": dict(self.state.utilities),
            "last_allocation": dict(self.state.last_allocation),
            "last_renewals": dict(self.state.last_renewals),
        }

    def _state_signature(self, state: SimulationState, precision: int) -> Tuple[object, ...]:
        def r(value: float) -> float:
            return round(value, precision)

        allocations_sig = [
            (((sender, receiver)), r(amount)) for (sender, receiver), amount in state.last_allocation.items()
        ]
        allocations_sig = tuple(sorted(allocations_sig, key=lambda item: str(item[0])))

        if self._uses_permanent_cycle_signature():
            return ("allocations", allocations_sig)

        lifetimes_sig = [
            (edge, link.remaining_life)
            for edge, link in state.links.items()
        ]
        lifetimes_sig = tuple(sorted(lifetimes_sig, key=lambda item: str(item[0])))

        return ("allocations_lifetimes", allocations_sig, lifetimes_sig)

    def _uses_permanent_cycle_signature(self) -> bool:
        return bool(self.edge_lifetimes) and all(lifetime is None for lifetime in self.edge_lifetimes.values())


__all__ = ["DecentralizedExchangeSimulator"]

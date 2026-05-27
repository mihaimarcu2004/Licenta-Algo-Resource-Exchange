from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Tuple, Union
import math
import random

from .heuristics import (
    EV_HEURISTIC_ADAPTIVE_HORIZON,
    EV_HEURISTIC_CAP_HORIZON_20,
    EV_HEURISTIC_CAP_HORIZON_50,
    EV_HEURISTIC_COOLDOWN_REJECTIONS,
    EV_HEURISTIC_FREEZE_OTHER_ALLOCATIONS,
    EV_HEURISTIC_HYSTERESIS_THRESHOLD,
    EV_HEURISTIC_LOCAL_ROLLOUT,
    EV_HEURISTIC_NO_FORECAST_RENEWALS,
    EV_HEURISTIC_ONE_STEP_CONSTANT_UTILITY,
    EV_HEURISTIC_PROPORTIONAL_SCREENING,
    EV_HEURISTIC_SKIP_INACTIVE_PRUNING,
    EV_HEURISTIC_TOP_K_CANDIDATES,
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
        ev_screening_top_k: int = 3,
        ev_screening_margin: float = 0.1,
        ev_screening_weight: float = 0.7,
        ev_initial_signal_fraction: float = 0.05,
        ev_cooldown_period: int = 5,
        ev_cooldown_ratio_threshold: float = 0.1,
        ev_adaptive_epsilon: float = 0.01,
        link_evaluation_workers: int = 1,
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
        self.ev_screening_top_k = ev_screening_top_k
        self.ev_screening_margin = ev_screening_margin
        self.ev_screening_weight = ev_screening_weight
        self.ev_initial_signal_fraction = ev_initial_signal_fraction
        self.ev_cooldown_period = ev_cooldown_period
        self.ev_cooldown_ratio_threshold = ev_cooldown_ratio_threshold
        self.ev_adaptive_epsilon = ev_adaptive_epsilon
        self.link_evaluation_workers = link_evaluation_workers
        self.rejected_link_cooldowns: Dict[Tuple[AgentId, AgentId], Tuple[int, float, int]] = {}
        self._ev_baseline_cache: Dict[Tuple[AgentId, int, Optional[float], int, int, Tuple[AgentId, ...]], float] = {}
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
        if ev_screening_top_k <= 0:
            raise ValueError("ev_screening_top_k must be positive.")
        if ev_screening_margin < 0:
            raise ValueError("ev_screening_margin must be non-negative.")
        if not 0 <= ev_screening_weight <= 1:
            raise ValueError("ev_screening_weight must be in [0, 1].")
        if ev_initial_signal_fraction < 0:
            raise ValueError("ev_initial_signal_fraction must be non-negative.")
        if ev_cooldown_period < 0:
            raise ValueError("ev_cooldown_period must be non-negative.")
        if ev_cooldown_ratio_threshold < 0:
            raise ValueError("ev_cooldown_ratio_threshold must be non-negative.")
        if ev_adaptive_epsilon < 0:
            raise ValueError("ev_adaptive_epsilon must be non-negative.")
        if link_evaluation_workers <= 0:
            raise ValueError("link_evaluation_workers must be positive.")
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
        return [edge for edge in self.candidate_edge_universe if edge not in active]

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
    ) -> Optional[Tuple[int, int]]:
        result = self.find_equilibrium_states(max_steps=max_steps, precision=precision, eps=eps)
        if result is None:
            return None
        start_time, period, _ = result
        return (start_time, period)

    def find_equilibrium_states(
        self,
        max_steps: int = 1000000000,
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
        self._ev_baseline_cache = {}
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
        if self._uses_proportional_screening():
            return self._screened_new_link_actions()

        actions: List[Tuple[float, str, Edge, AgentId]] = []
        for action in self._parallel_map(self.candidate_edges(), self._evaluate_candidate_edge_action):
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

    def _parallel_map(self, items: Iterable[object], fn):
        item_list = list(items)
        if self.link_evaluation_workers <= 1 or len(item_list) <= 1:
            return [fn(item) for item in item_list]
        with ThreadPoolExecutor(max_workers=self.link_evaluation_workers) as executor:
            return list(executor.map(fn, item_list))

    def _evaluate_screened_rollout_candidate(
        self,
        candidate: Tuple[float, Edge, AgentId, AgentId, float, Optional[int], float, float],
    ) -> Tuple[Optional[Tuple[float, str, Edge, AgentId]], Optional[Tuple[AgentId, AgentId]]]:
        _fast_score, edge, proposer, neighbor, cost, lifetime, degradation, discount = candidate
        gain = self.estimate_link_gain(
            proposer,
            neighbor,
            cost,
            lifetime,
            degradation,
            discount,
            use_screening=False,
        )
        threshold = self._screening_margin_cost(cost)
        if gain >= threshold:
            return ((gain, "new", edge, proposer), None)
        return (None, (proposer, neighbor))

    def _evaluate_screening_candidate(
        self,
        candidate: Tuple[Edge, AgentId, AgentId, float, Optional[int], float, float],
    ) -> Tuple[Edge, AgentId, AgentId, str, float]:
        edge, proposer, neighbor, cost, lifetime, degradation, discount = candidate
        score_kind, score = self._screen_directed_new_link(
            proposer,
            neighbor,
            cost,
            lifetime,
            degradation,
            discount,
            record_rejection=False,
        )
        return (edge, proposer, neighbor, score_kind, score)

    def _uses_proportional_screening(self) -> bool:
        return (
            EV_HEURISTIC_PROPORTIONAL_SCREENING in self.ev_heuristics
            and not isinstance(self.strategy, ConstantStrategy)
        )

    def _screened_new_link_actions(self) -> List[Tuple[float, str, Edge, AgentId]]:
        accepted: Dict[Edge, Tuple[float, str, Edge, AgentId]] = {}
        borderline_by_agent: Dict[AgentId, List[Tuple[float, Edge, AgentId, AgentId, float, Optional[int], float, float]]] = {
            agent: [] for agent in self.agents
        }
        screening_candidates: List[Tuple[Edge, AgentId, AgentId, float, Optional[int], float, float]] = []
        candidate_params: Dict[Tuple[AgentId, AgentId], Tuple[float, Optional[int], float, float]] = {}

        for edge in self.candidate_edges():
            u, v = edge
            cost = self.edge_costs.get(edge, 0.0)
            lifetime = self.edge_lifetimes.get(edge, self.expiry)
            degradation = self.edge_degradations.get(edge, self.degradation)
            discount = self.edge_discounts.get(edge, self.discount)
            for proposer, neighbor in ((u, v), (v, u)):
                screening_candidates.append((edge, proposer, neighbor, cost, lifetime, degradation, discount))
                candidate_params[(proposer, neighbor)] = (cost, lifetime, degradation, discount)

        for edge, proposer, neighbor, score_kind, score in self._parallel_map(
            screening_candidates,
            self._evaluate_screening_candidate,
        ):
            cost, lifetime, degradation, discount = candidate_params[(proposer, neighbor)]
            if score_kind == "accept":
                self._keep_best_screened_action(accepted, score, edge, proposer)
            elif score_kind == "rollout":
                borderline_by_agent[proposer].append(
                    (score, edge, proposer, neighbor, cost, lifetime, degradation, discount)
                )
            elif score_kind == "reject":
                self._record_rejected_candidate(proposer, neighbor)

        rollout_candidates: List[Tuple[float, Edge, AgentId, AgentId, float, Optional[int], float, float]] = []
        for proposer, candidates in borderline_by_agent.items():
            candidates.sort(key=lambda item: (-item[0], str(item[1]), str(item[2])))
            if EV_HEURISTIC_TOP_K_CANDIDATES in self.ev_heuristics:
                candidates = candidates[: self.ev_screening_top_k]
            rollout_candidates.extend(candidates)

        for action, rejected in self._parallel_map(rollout_candidates, self._evaluate_screened_rollout_candidate):
            if action is not None:
                score, _kind, edge, proposer = action
                self._keep_best_screened_action(accepted, score, edge, proposer)
            if rejected is not None:
                proposer, neighbor = rejected
                self._record_rejected_candidate(proposer, neighbor)

        return list(accepted.values())

    def _keep_best_screened_action(
        self,
        actions: Dict[Edge, Tuple[float, str, Edge, AgentId]],
        score: float,
        edge: Edge,
        proposer: AgentId,
    ) -> None:
        current = actions.get(edge)
        candidate = (score, "new", edge, proposer)
        if current is None or score > current[0]:
            actions[edge] = candidate

    def _screen_directed_new_link(
        self,
        agent: AgentId,
        new_neighbor: AgentId,
        paid_cost: float,
        lifetime: Optional[int],
        degradation: float,
        discount: Optional[float],
        record_rejection: bool = True,
    ) -> Tuple[str, float]:
        if paid_cost > self.production[agent]:
            return ("reject", -math.inf)
        if self._candidate_on_cooldown(agent, new_neighbor):
            return ("reject", -math.inf)

        upper_bound = self._link_upper_bound(new_neighbor, paid_cost, lifetime, degradation, discount)
        if upper_bound < paid_cost:
            if record_rejection:
                self._record_rejected_candidate(agent, new_neighbor)
            return ("reject", upper_bound - paid_cost)

        fast_score = self._proportional_fast_score(agent, new_neighbor, paid_cost, lifetime, degradation, discount)
        threshold = self._screening_margin_cost(paid_cost)
        if fast_score > threshold:
            return ("accept", fast_score)
        if fast_score < -threshold:
            if record_rejection:
                self._record_rejected_candidate(agent, new_neighbor)
            return ("reject", fast_score)
        return ("rollout", fast_score)

    def _screening_margin_cost(self, cost: float) -> float:
        if EV_HEURISTIC_HYSTERESIS_THRESHOLD in self.ev_heuristics or self._uses_proportional_screening():
            return self.ev_screening_margin * cost
        return 0.0

    def _link_upper_bound(
        self,
        neighbor: AgentId,
        cost: float,
        lifetime: Optional[int],
        degradation: float,
        discount: Optional[float],
    ) -> float:
        horizon = self._expected_utility_horizon(lifetime)
        horizon = self._adaptive_horizon(horizon, neighbor, cost, degradation, discount)
        return self.production[neighbor] * self._discounted_response_multiplier(horizon, degradation, discount)

    def _proportional_fast_score(
        self,
        agent: AgentId,
        new_neighbor: AgentId,
        paid_cost: float,
        lifetime: Optional[int],
        degradation: float,
        discount: Optional[float],
    ) -> float:
        horizon = self._expected_utility_horizon(lifetime)
        horizon = self._adaptive_horizon(horizon, new_neighbor, paid_cost, degradation, discount)
        multiplier = self._discounted_response_multiplier(horizon, degradation, discount)

        neighbor_received = sum(self.state.last_received.get(new_neighbor, {}).values())
        initial_signal = self.ev_initial_signal_fraction * self.production[agent]
        initial_signal += getattr(self.strategy, "boost_lambda", 0.0)
        denominator = neighbor_received + initial_signal
        if denominator <= 0:
            response_value = 0.0
        else:
            first_response = self.production[new_neighbor] * initial_signal / denominator
            response_value = first_response * multiplier

        ratio = self.state.exchange_ratios.get(new_neighbor, 1.0)
        ratio_value = multiplier / (ratio + 1e-9)
        return (self.ev_screening_weight * response_value) + (
            (1 - self.ev_screening_weight) * ratio_value
        ) - paid_cost

    def _discounted_response_multiplier(
        self,
        horizon: int,
        degradation: float,
        discount: Optional[float],
    ) -> float:
        time_discount = self.discount if discount is None else discount
        if horizon <= 0:
            return 0.0
        factor = time_discount * degradation
        if math.isclose(factor, 1.0):
            return time_discount * float(horizon)
        return time_discount * (1 - (factor ** horizon)) / (1 - factor)

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

    def _candidate_on_cooldown(self, agent: AgentId, neighbor: AgentId) -> bool:
        if EV_HEURISTIC_COOLDOWN_REJECTIONS not in self.ev_heuristics:
            return False

        entry = self.rejected_link_cooldowns.get((agent, neighbor))
        if entry is None:
            return False

        until_time, rejected_ratio, rejected_degree = entry
        if self.state.t >= until_time:
            return False

        current_ratio = self.state.exchange_ratios.get(neighbor, 1.0)
        current_degree = self.state.degree(neighbor)
        if abs(current_ratio - rejected_ratio) > self.ev_cooldown_ratio_threshold:
            return False
        if current_degree != rejected_degree:
            return False
        return True

    def _record_rejected_candidate(self, agent: AgentId, neighbor: AgentId) -> None:
        if EV_HEURISTIC_COOLDOWN_REJECTIONS not in self.ev_heuristics or self.ev_cooldown_period <= 0:
            return
        self.rejected_link_cooldowns[(agent, neighbor)] = (
            self.state.t + self.ev_cooldown_period,
            self.state.exchange_ratios.get(neighbor, 1.0),
            self.state.degree(neighbor),
        )

    def _local_rollout_nodes(self, agent: AgentId, neighbor: AgentId) -> set[AgentId]:
        return {agent, neighbor, *self.state.neighbors(agent), *self.state.neighbors(neighbor)}

    @staticmethod
    def _filter_links_to_nodes(links: Dict[Edge, LinkState], nodes: set[AgentId]) -> Dict[Edge, LinkState]:
        return {
            edge: link
            for edge, link in links.items()
            if edge[0] in nodes and edge[1] in nodes
        }

    def estimate_link_gain(
        self,
        agent: AgentId,
        new_neighbor: AgentId,
        paid_cost: float,
        lifetime: Optional[int],
        degradation: float,
        discount: Optional[float] = None,
        use_screening: bool = True,
    ) -> float:
        """
        Estimate whether adding one link is useful for `agent` under the current strategy.

        Expected utility is the discounted utility the agent would receive over the
        candidate link lifetime, assuming no other links are created and all other
        existing links are renewed when their lifetime ends.
        """
        if paid_cost > self.production[agent]:
            return -math.inf

        edge = normalize_edge(agent, new_neighbor)
        if edge in self.state.links:
            return -math.inf

        if use_screening and self._uses_proportional_screening():
            score_kind, score = self._screen_directed_new_link(
                agent,
                new_neighbor,
                paid_cost,
                lifetime,
                degradation,
                discount,
            )
            if score_kind == "rollout":
                return self.estimate_link_gain(
                    agent,
                    new_neighbor,
                    paid_cost,
                    lifetime,
                    degradation,
                    discount,
                    use_screening=False,
                )
            return score

        if isinstance(self.strategy, ConstantStrategy):
            expected_benefit = self._egalitarian_creation_benefit(new_neighbor, lifetime, degradation, discount)
            return expected_benefit - paid_cost

        baseline_links = self._copy_links(self.state.links)
        created_links = self._copy_links(self.state.links)
        created_links[edge] = LinkState(
            u=edge[0],
            v=edge[1],
            remaining_life=lifetime,
            quality=1.0,
            degradation=degradation,
            initial=False,
            created_at=self.state.t + 1,
        )

        local_nodes_tuple: Tuple[AgentId, ...] = ()
        if EV_HEURISTIC_LOCAL_ROLLOUT in self.ev_heuristics:
            local_nodes = self._local_rollout_nodes(agent, new_neighbor)
            local_nodes_tuple = tuple(sorted(local_nodes, key=str))
            baseline_links = self._filter_links_to_nodes(baseline_links, local_nodes)
            created_links = self._filter_links_to_nodes(created_links, local_nodes)

        horizon = self._expected_utility_horizon(lifetime)
        horizon = self._adaptive_horizon(horizon, new_neighbor, paid_cost, degradation, discount)
        baseline_cache_key = (agent, horizon, discount, self.state.t, 1, local_nodes_tuple)
        if baseline_cache_key in self._ev_baseline_cache:
            baseline = self._ev_baseline_cache[baseline_cache_key]
        else:
            baseline = self._forecast_expected_utility(
                agent,
                baseline_links,
                horizon,
                discount,
                excluded_renewal_edge=edge,
                start_time=self.state.t + 1,
                discount_start_power=1,
            )
            self._ev_baseline_cache[baseline_cache_key] = baseline
        with_link = self._forecast_expected_utility(
            agent,
            created_links,
            horizon,
            discount,
            excluded_renewal_edge=edge,
            start_time=self.state.t + 1,
            discount_start_power=1,
        )
        return (with_link - baseline) - paid_cost

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

        if isinstance(self.strategy, ConstantStrategy):
            expected_gain = self._egalitarian_renewal_gain(agent, neighbor, lifetime, degradation, discount)
            return expected_gain - paid_cost

        baseline_links = self._copy_links(self.state.links)
        renewed_links = self._copy_links(self.state.links)
        renewed = renewed_links[edge]
        renewed.remaining_life = lifetime
        renewed.quality = 1.0
        renewed.degradation = degradation
        renewed.created_at = self.state.t + 1

        if EV_HEURISTIC_LOCAL_ROLLOUT in self.ev_heuristics:
            local_nodes = self._local_rollout_nodes(agent, neighbor)
            baseline_links = self._filter_links_to_nodes(baseline_links, local_nodes)
            renewed_links = self._filter_links_to_nodes(renewed_links, local_nodes)

        horizon = self._expected_utility_horizon(lifetime)
        horizon = self._adaptive_horizon(horizon, neighbor, paid_cost, degradation, discount)
        baseline = self._forecast_expected_utility(
            agent,
            baseline_links,
            horizon,
            discount,
            excluded_renewal_edge=edge,
            start_time=self.state.t + 1,
        )
        renewed_value = self._forecast_expected_utility(
            agent,
            renewed_links,
            horizon,
            discount,
            excluded_renewal_edge=edge,
            start_time=self.state.t + 1,
        )
        return (renewed_value - baseline) - paid_cost

    def _egalitarian_creation_benefit(
        self,
        new_neighbor: AgentId,
        lifetime: Optional[int],
        degradation: float,
        discount: Optional[float],
    ) -> float:
        neighbor_degree_after_creation = self.state.degree(new_neighbor) + 1
        if neighbor_degree_after_creation <= 0:
            return 0.0

        direct_flow = self.production[new_neighbor] / neighbor_degree_after_creation
        return direct_flow * self._discounted_degradation_multiplier(lifetime, degradation, discount)

    def _egalitarian_renewal_gain(
        self,
        agent: AgentId,
        neighbor: AgentId,
        lifetime: Optional[int],
        degradation: float,
        discount: Optional[float],
    ) -> float:
        edge = normalize_edge(agent, neighbor)
        link = self.state.links[edge]
        neighbor_degree = self.state.degree(neighbor)
        if neighbor_degree <= 0:
            return 0.0

        direct_flow = self.production[neighbor] / neighbor_degree
        renewed_benefit = direct_flow * self._discounted_degradation_multiplier(lifetime, degradation, discount)

        remaining_life = link.remaining_life
        age = max(self.state.t - link.created_at, 0)
        current_degradation = link.degradation
        not_renewed_benefit = (
            direct_flow
            * (current_degradation ** age)
            * self._discounted_degradation_multiplier(remaining_life, current_degradation, discount)
        )
        return renewed_benefit - not_renewed_benefit

    def _discounted_degradation_multiplier(
        self,
        lifetime: Optional[int],
        degradation: float,
        discount: Optional[float],
    ) -> float:
        time_discount = self.discount if discount is None else discount
        factor = time_discount * degradation
        if lifetime is None:
            if factor >= 1:
                return math.inf
            return 1 / (1 - factor)

        periods = max(int(lifetime), 0)
        if periods == 0:
            return 0.0
        if math.isclose(factor, 1.0):
            return float(periods)
        return (1 - (factor ** periods)) / (1 - factor)

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

    def _forecast_expected_utility(
        self,
        agent: AgentId,
        links: Dict[Edge, LinkState],
        horizon: int,
        discount: Optional[float],
        excluded_renewal_edge: Optional[Edge] = None,
        start_time: Optional[int] = None,
        discount_start_power: int = 0,
    ) -> float:
        time_discount = self.discount if discount is None else discount
        first_time = self.state.t if start_time is None else start_time
        forecast_links = self._copy_links(links)
        last_received = {
            receiver: dict(received)
            for receiver, received in self.state.last_received.items()
        }
        expected_utility = 0.0

        for step in range(horizon):
            if EV_HEURISTIC_NO_FORECAST_RENEWALS not in self.ev_heuristics:
                self._apply_forecast_renewals(forecast_links, excluded_renewal_edge)
            else:
                self._remove_expired_forecast_links(forecast_links)
            if not forecast_links:
                break

            forecast_state = SimulationState(
                t=first_time + step,
                agents=self.agents,
                production=self.production,
                links=forecast_links,
                last_received=last_received,
                last_creation_costs={i: 0.0 for i in self.agents},
                last_renewal_costs={i: 0.0 for i in self.agents},
                last_renewals={},
                exchange_ratios={i: 1.0 for i in self.agents},
                utilities={i: 0.0 for i in self.agents},
            )
            allocation = self._forecast_allocations(forecast_state, excluded_renewal_edge)
            received = self._forecast_received(forecast_state, allocation)
            utility = self.utility_formula(agent, received.get(agent, {}), forecast_state)
            if EV_HEURISTIC_ONE_STEP_CONSTANT_UTILITY in self.ev_heuristics:
                multiplier = self._discount_sum(horizon, time_discount, discount_start_power)
                return utility * multiplier
            expected_utility += (time_discount ** (step + discount_start_power)) * utility
            last_received = received
            if EV_HEURISTIC_SKIP_INACTIVE_PRUNING not in self.ev_heuristics:
                self._remove_inactive_forecast_links(forecast_state, allocation)
            self._age_forecast_links(forecast_links)

        return expected_utility

    @staticmethod
    def _discount_sum(horizon: int, discount: float, start_power: int = 0) -> float:
        if horizon <= 0:
            return 0.0
        if math.isclose(discount, 1.0):
            return float(horizon)
        return (discount ** start_power) * (1 - (discount ** horizon)) / (1 - discount)

    def _apply_forecast_renewals(
        self,
        links: Dict[Edge, LinkState],
        excluded_renewal_edge: Optional[Edge],
    ) -> None:
        to_delete: List[Edge] = []
        for edge, link in links.items():
            if link.remaining_life is None or link.remaining_life > 0:
                continue
            if excluded_renewal_edge is not None and edge == excluded_renewal_edge:
                to_delete.append(edge)
                continue
            lifetime = self.edge_lifetimes.get(edge, self.expiry)
            if lifetime is None:
                link.remaining_life = None
            else:
                link.remaining_life = lifetime
            link.quality = 1.0
            link.degradation = self.edge_degradations.get(edge, link.degradation)
        for edge in to_delete:
            del links[edge]

    def _remove_expired_forecast_links(self, links: Dict[Edge, LinkState]) -> None:
        to_delete = [
            edge
            for edge, link in links.items()
            if link.remaining_life is not None and link.remaining_life <= 0
        ]
        for edge in to_delete:
            del links[edge]

    def _forecast_allocations(
        self,
        state: SimulationState,
        target_edge: Optional[Edge] = None,
    ) -> Allocation:
        if (
            target_edge is not None
            and EV_HEURISTIC_FREEZE_OTHER_ALLOCATIONS in self.ev_heuristics
        ):
            return self._forecast_allocations_freeze_other(state, target_edge)

        allocation: Allocation = {}
        for i in self.agents:
            out = self.strategy.allocate(state, i, self.production[i])
            for j, amount in out.items():
                if amount > 0 and normalize_edge(i, j) in state.links:
                    allocation[(i, j)] = amount
        return allocation

    def _forecast_allocations_freeze_other(
        self,
        state: SimulationState,
        target_edge: Edge,
    ) -> Allocation:
        allocation: Allocation = {}
        remaining_budget = {agent: self.production[agent] for agent in self.agents}

        for (sender, receiver), current_amount in self.state.last_allocation.items():
            edge = normalize_edge(sender, receiver)
            if edge == target_edge or edge not in state.links or current_amount <= 0:
                continue
            amount = min(current_amount, remaining_budget.get(sender, 0.0))
            if amount <= 0:
                continue
            allocation[(sender, receiver)] = amount
            remaining_budget[sender] -= amount

        if target_edge in state.links:
            u, v = target_edge
            for sender, receiver in ((u, v), (v, u)):
                if sender not in self.production or receiver not in self.production:
                    continue
                remaining = remaining_budget.get(sender, 0.0)
                if remaining <= 0:
                    continue
                out = self.strategy.allocate(state, sender, self.production[sender])
                target_amount = min(out.get(receiver, 0.0), remaining)
                if target_amount > 0:
                    allocation[(sender, receiver)] = target_amount

        return allocation

    def _forecast_received(
        self,
        state: SimulationState,
        allocation: Allocation,
    ) -> Dict[AgentId, Dict[AgentId, float]]:
        received: Dict[AgentId, Dict[AgentId, float]] = {i: {} for i in self.agents}
        for (sender, receiver), amount in allocation.items():
            quality = state.link_quality(sender, receiver)
            received[receiver][sender] = received[receiver].get(sender, 0.0) + amount * quality
        return received

    def _age_forecast_links(self, links: Dict[Edge, LinkState]) -> None:
        for link in links.values():
            link.quality *= link.degradation
            if link.remaining_life is not None:
                link.remaining_life -= 1

    def _remove_inactive_forecast_links(
        self,
        state: SimulationState,
        allocation: Allocation,
    ) -> None:
        if self.min_link_flow <= 0:
            return

        effective_flow: Dict[Edge, float] = {edge: 0.0 for edge in state.links}
        for (sender, receiver), amount in allocation.items():
            edge = normalize_edge(sender, receiver)
            if edge not in state.links:
                continue
            effective_flow[edge] += amount * state.link_quality(sender, receiver)

        to_delete = [
            edge
            for edge, flow in effective_flow.items()
            if not state.links[edge].initial and flow < self.min_link_flow
        ]
        for edge in to_delete:
            del state.links[edge]

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

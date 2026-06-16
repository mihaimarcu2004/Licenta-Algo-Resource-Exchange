import itertools
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exchange_sim import DecentralizedExchangeSimulator, normalize_edge

COST_BANDS = {
    "small": (1.0, 3.0),
    "medium": (3.5, 7.0),
    "big": (8.0, 14.0),
}
LIFETIME_BANDS = {
    "small_lifetime": (1, 3),
    "medium_lifetime": (3, 7),
    "large_lifetime": (7, 12),
}
PRODUCTION_MEAN = 10.0
PRODUCTION_STD = 2.0
PRODUCTION_MIN = 1.0


def sample_production(rng: random.Random) -> float:
    return max(PRODUCTION_MIN, rng.gauss(PRODUCTION_MEAN, PRODUCTION_STD))


def jain_fairness(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 1.0
    s = sum(vals)
    if s == 0:
        return 0.0
    return (s * s) / (len(vals) * sum(v * v for v in vals))


def welfare_of_others(utilities: Dict[int, float]) -> Dict[int, float]:
    total = sum(utilities.values())
    return {agent: total - value for agent, value in utilities.items()}


def eisenberg_gale_welfare(utilities: Dict[int, float], eps: float = 1e-9) -> float:
    return sum(math.log(max(value, eps)) for value in utilities.values())


def ensure_connected(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G
    if nx.is_connected(G):
        return G
    connected = G.copy()
    components = [list(component) for component in nx.connected_components(connected)]
    for left, right in zip(components, components[1:]):
        connected.add_edge(left[0], right[0])
    return connected


def build_edge_costs(
    nodes: List[int],
    initial_edges: List[Tuple[int, int]],
    cost_fn,
) -> Dict[Tuple[int, int], float]:
    initial = {normalize_edge(u, v) for u, v in initial_edges}
    edge_costs: Dict[Tuple[int, int], float] = {}
    for u, v in itertools.combinations(nodes, 2):
        e = normalize_edge(u, v)
        if e in initial:
            continue
        cost = cost_fn(u, v)
        if cost is None:
            continue
        edge_costs[e] = float(cost)
    return edge_costs


def generate_graph(kind: str, n: int, seed: int) -> nx.Graph:
    if kind == "sparse":
        G = nx.erdos_renyi_graph(n, 0.15, seed=seed)
    elif kind == "dense":
        G = nx.erdos_renyi_graph(n, 0.6, seed=seed)
    elif kind == "small_world":
        G = nx.watts_strogatz_graph(n, 4, 0.2, seed=seed)
    elif kind == "scale_free":
        m = max(1, min(3, n - 1))
        G = nx.barabasi_albert_graph(n, m, seed=seed)
    elif kind == "two_communities":
        left = n // 2
        right = n - left
        G = nx.random_partition_graph([left, right], 0.45, 0.04, seed=seed)
    elif kind == "core_periphery":
        rng = random.Random(seed)
        core_size = max(3, n // 4)
        core = set(range(core_size))
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u, v in itertools.combinations(range(n), 2):
            if u in core and v in core:
                p = 0.75
            elif u in core or v in core:
                p = 0.28
            else:
                p = 0.04
            if rng.random() < p:
                G.add_edge(u, v)
    else:
        raise ValueError(f"Unknown graph kind: {kind}")
    return ensure_connected(G)


def make_distance_costs(G: nx.Graph, factor: float = 0.8) -> Dict[Tuple[int, int], float]:
    lengths = dict(nx.all_pairs_shortest_path_length(G))

    def distance_cost(u: int, v: int) -> float:
        return factor * lengths[u][v]

    nodes = list(G.nodes())
    initial_edges = list(G.edges())
    return build_edge_costs(nodes, initial_edges, distance_cost)


def make_constant_costs(G: nx.Graph, value: float = 1.0) -> Dict[Tuple[int, int], float]:
    nodes = list(G.nodes())
    initial_edges = list(G.edges())
    return build_edge_costs(nodes, initial_edges, lambda _u, _v: value)


def make_zero_costs(G: nx.Graph) -> Dict[Tuple[int, int], float]:
    nodes = list(G.nodes())
    initial_edges = list(G.edges())
    return build_edge_costs(nodes, initial_edges, lambda _u, _v: 0.0)


def make_random_costs(
    G: nx.Graph,
    rng: random.Random,
    low: float,
    high: float,
) -> Dict[Tuple[int, int], float]:
    nodes = list(G.nodes())
    initial_edges = list(G.edges())
    return build_edge_costs(nodes, initial_edges, lambda _u, _v: rng.uniform(low, high))


def make_degree_costs(G: nx.Graph, base: float = 0.5, scale: float = 4.0) -> Dict[Tuple[int, int], float]:
    nodes = list(G.nodes())
    initial_edges = list(G.edges())
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1

    def degree_cost(u: int, v: int) -> float:
        normalized_degree = (degrees[u] + degrees[v]) / max(2 * max_degree, 1)
        return base + scale * normalized_degree

    return build_edge_costs(nodes, initial_edges, degree_cost)


def make_mixed_costs(G: nx.Graph, rng: random.Random) -> Dict[Tuple[int, int], float]:
    distance_costs = make_distance_costs(G, factor=0.7)
    degree_costs = make_degree_costs(G, base=0.4, scale=2.5)
    edge_costs: Dict[Tuple[int, int], float] = {}
    for edge in set(distance_costs) | set(degree_costs):
        random_component = rng.uniform(0.25, 1.75)
        edge_costs[edge] = random_component + distance_costs.get(edge, 0.0) + degree_costs.get(edge, 0.0)
    return edge_costs


def build_edge_degradations(
    edge_costs: Dict[Tuple[int, int], float],
    rng: random.Random,
    low: float = 0.85,
    high: float = 1.0,
) -> Dict[Tuple[int, int], float]:
    return {edge: rng.uniform(low, high) for edge in edge_costs}


def build_edge_lifetimes(
    edge_costs: Dict[Tuple[int, int], float],
    rng: random.Random,
    expiry: Optional[int],
    lifetime_kind: str = "small_lifetime",
) -> Dict[Tuple[int, int], Optional[int]]:
    if expiry is None or lifetime_kind == "permanent":
        return {edge: None for edge in edge_costs}

    if lifetime_kind == "same_lifetime":
        return {edge: int(expiry) for edge in edge_costs}

    if lifetime_kind in LIFETIME_BANDS:
        low, high = LIFETIME_BANDS[lifetime_kind]
        return {edge: rng.randint(low, high) for edge in edge_costs}

    if lifetime_kind == "mixed_lifetime":
        band_names = list(LIFETIME_BANDS)
        return {
            edge: rng.randint(*LIFETIME_BANDS[rng.choice(band_names)])
            for edge in edge_costs
        }

    raise ValueError(f"Unknown lifetime kind: {lifetime_kind}")


def state_welfare_metrics(state, edge_costs: Dict[Tuple[int, int], float]) -> Dict[str, float]:
    utilities = state.utilities
    exchange_ratios = state.exchange_ratios
    moment_link_cost = sum(state.last_creation_costs.values()) + sum(state.last_renewal_costs.values())
    average_utility = sum(utilities.values()) / len(utilities) if utilities else 0.0
    welfare_eg = eisenberg_gale_welfare(utilities) if utilities else 0.0
    active_edges = float(len(state.links))
    average_degree = (2.0 * active_edges / len(state.agents)) if state.agents else 0.0
    return {
        "average_utility": average_utility,
        "welfare_eisenberg_gale": welfare_eg,
        "welfare_net": sum(utilities.values()) - moment_link_cost,
        "fairness_jain": jain_fairness(utilities.values()),
        "fairness_exchange_ratio_jain": jain_fairness(exchange_ratios.values()),
        "active_edges": active_edges,
        "average_degree": average_degree,
    }


def run_experiment(
    graph_kind: str,
    cost_kind: str,
    expiry: Optional[int],
    n: int = 20,
    seed: int = 1,
    strategy: str = "constant",
    constant_cost: float = 1.0,
    distance_factor: float = 0.8,
    cost_multiplier: float = 1.0,
    max_steps: int = 1000000000,
    min_equilibrium_steps: int = 0,
    equilibrium_precision: int = 9,
    equilibrium_eps: float = 1e-9,
    refresh_threshold: Optional[float] = 0.8,
    renewal_life_threshold: Optional[int] = 1,
    min_link_flow: float = 1e-4,
    lifetime_kind: str = "small_lifetime",
    ev_heuristics: Optional[Iterable[str]] = None,
    ev_adaptive_epsilon: float = 0.01,
    link_estimation_mode: str = "formula",
    max_candidate_edges_per_agent: Optional[int] = None,
    allow_link_formation: bool = True,
    include_details: bool = False,
    include_time_series: bool = False,
    welfare_time_interval: int = 100,
) -> Dict[str, object]:
    rng = random.Random(seed)
    G = generate_graph(graph_kind, n, seed)
    nodes = list(G.nodes())
    initial_edges = list(G.edges())

    production = {node: sample_production(rng) for node in nodes}
    cost_rng = random.Random(seed + 37)

    if cost_kind in COST_BANDS:
        low, high = COST_BANDS[cost_kind]
        edge_costs = make_random_costs(G, cost_rng, low, high)
    elif cost_kind == "zero":
        edge_costs = make_zero_costs(G)
    elif cost_kind == "constant":
        edge_costs = make_constant_costs(G, constant_cost)
    elif cost_kind == "distance":
        edge_costs = make_distance_costs(G, distance_factor)
    elif cost_kind == "degree":
        edge_costs = make_degree_costs(G)
    elif cost_kind == "mixed":
        edge_costs = make_mixed_costs(G, cost_rng)
    else:
        raise ValueError(f"Unknown cost kind: {cost_kind}")
    if cost_multiplier <= 0:
        raise ValueError("cost_multiplier must be positive.")
    if not math.isclose(cost_multiplier, 1.0):
        edge_costs = {edge: cost * cost_multiplier for edge, cost in edge_costs.items()}
    if not allow_link_formation:
        edge_costs = {}

    deg_rng = random.Random(seed + 91)
    edge_degradations = build_edge_degradations(edge_costs, deg_rng)
    lifetime_rng = random.Random(seed + 131)
    edge_lifetimes = build_edge_lifetimes(edge_costs, lifetime_rng, expiry, lifetime_kind)
    expiry_value = 5 if expiry is None else int(expiry)

    sim = DecentralizedExchangeSimulator(
        agents=nodes,
        production=production,
        initial_edges=initial_edges,
        strategy=strategy,
        edge_costs=edge_costs,
        renewal_costs=edge_costs,
        edge_lifetimes=edge_lifetimes,
        edge_degradations=edge_degradations,
        expiry=expiry_value,
        degradation=0.95,
        refresh_threshold=refresh_threshold,
        renewal_life_threshold=renewal_life_threshold,
        min_link_flow=min_link_flow,
        discount=0.9,
        ev_heuristics=ev_heuristics,
        ev_adaptive_epsilon=ev_adaptive_epsilon,
        link_estimation_mode=link_estimation_mode,
        max_candidate_edges_per_agent=max_candidate_edges_per_agent,
        max_new_links_per_slot=None,
        rng_seed=seed,
    )

    result = sim.find_equilibrium_states(
        max_steps=max_steps,
        precision=equilibrium_precision,
        eps=equilibrium_eps,
        min_steps=min_equilibrium_steps,
    )
    if result is None:
        states = sim.history[-1000:] or [sim.state]
        equilibrium_found = False
        equilibrium_start = None
        equilibrium_period = None
        measurement_window = "last_1000"
    else:
        equilibrium_start, equilibrium_period, states = result
        equilibrium_found = True
        measurement_window = "cycle"

    utility_acc: Dict[int, float] = {agent: 0.0 for agent in nodes}
    exchange_ratio_acc: Dict[int, float] = {agent: 0.0 for agent in nodes}
    allocation_acc: Dict[Tuple[int, int], float] = {}
    renewal_acc: Dict[Tuple[int, int], float] = {}
    net_welfare_acc = 0.0
    average_degree_acc = 0.0
    for state in states:
        for agent, value in state.utilities.items():
            utility_acc[agent] += value
        for agent, value in state.exchange_ratios.items():
            exchange_ratio_acc[agent] += value
        for pair, amount in state.last_allocation.items():
            allocation_acc[pair] = allocation_acc.get(pair, 0.0) + amount
        for pair, decision in state.last_renewals.items():
            renewal_acc[pair] = renewal_acc.get(pair, 0.0) + decision
        moment_link_cost = sum(state.last_creation_costs.values()) + sum(state.last_renewal_costs.values())
        net_welfare_acc += sum(state.utilities.values()) - moment_link_cost
        if nodes:
            average_degree_acc += 2.0 * len(state.links) / len(nodes)

    count = float(len(states))
    utilities_avg = {agent: value / count for agent, value in utility_acc.items()}
    exchange_ratios_avg = {agent: value / count for agent, value in exchange_ratio_acc.items()}
    allocations_avg = {pair: amount / count for pair, amount in allocation_acc.items()}
    renewals_avg = {pair: amount / count for pair, amount in renewal_acc.items()}

    welfare_others = welfare_of_others(utilities_avg)
    average_utility = sum(utilities_avg.values()) / len(utilities_avg) if utilities_avg else 0.0
    welfare_eg = eisenberg_gale_welfare(utilities_avg)
    welfare_net = net_welfare_acc / count
    average_degree = average_degree_acc / count
    fairness = jain_fairness(utilities_avg.values())
    fairness_exchange_ratio = jain_fairness(exchange_ratios_avg.values())

    payload: Dict[str, object] = {
        "equilibrium_found": equilibrium_found,
        "equilibrium_start": equilibrium_start,
        "equilibrium_period": equilibrium_period,
        "measurement_window": measurement_window,
        "measurement_state_count": len(states),
        "lifetime_kind": lifetime_kind,
        "average_utility": average_utility,
        "average_degree": average_degree,
        "welfare_eisenberg_gale": welfare_eg,
        "welfare_net": welfare_net,
        "fairness_jain": fairness,
        "fairness_exchange_ratio_jain": fairness_exchange_ratio,
    }

    if include_details:
        representative = states[-1] if states else sim.state
        edges = sorted(representative.links.keys(), key=lambda item: (str(item[0]), str(item[1])))
        payload["utilities"] = utilities_avg
        payload["exchange_ratios"] = exchange_ratios_avg
        payload["welfare_others"] = welfare_others
        payload["allocations"] = allocations_avg
        payload["renewals"] = renewals_avg
        payload["nodes"] = nodes
        payload["edges"] = edges

    if include_time_series:
        interval = max(1, int(welfare_time_interval))
        sampled_states = [state for state in sim.history if state.t % interval == 0]
        if sim.history and (not sampled_states or sampled_states[-1].t != sim.history[-1].t):
            sampled_states.append(sim.history[-1])
        payload["time_series"] = [
            {"t": state.t, **state_welfare_metrics(state, edge_costs)}
            for state in sampled_states
        ]

    return payload


def print_header():
    print(
        "expiry,equilibrium_found,equilibrium_start,equilibrium_period,"
        "measurement_window,measurement_state_count,average_utility,average_degree,"
        "welfare_eisenberg_gale,welfare_net,fairness_jain,fairness_exchange_ratio_jain"
    )


def print_result(expiry: Optional[int], result: Dict[str, object]):
    expiry_label = "permanent" if expiry is None else str(expiry)
    print(
        f"{expiry_label},"
        f"{result['equilibrium_found']},"
        f"{result['equilibrium_start']},"
        f"{result['equilibrium_period']},"
        f"{result['measurement_window']},"
        f"{result['measurement_state_count']},"
        f"{result['average_utility']:.6f},"
        f"{result['average_degree']:.6f},"
        f"{result['welfare_eisenberg_gale']:.6f},"
        f"{result['welfare_net']:.6f},"
        f"{result['fairness_jain']:.6f},"
        f"{result['fairness_exchange_ratio_jain']:.6f}"
    )

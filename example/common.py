import itertools
import random
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx

from exchange_sim import DecentralizedExchangeSimulator, normalize_edge


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


def ensure_connected(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G
    if nx.is_connected(G):
        return G
    largest = max(nx.connected_components(G), key=len)
    return G.subgraph(largest).copy()


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


def build_edge_degradations(
    edge_costs: Dict[Tuple[int, int], float],
    rng: random.Random,
    low: float = 0.85,
    high: float = 1.0,
) -> Dict[Tuple[int, int], float]:
    return {edge: rng.uniform(low, high) for edge in edge_costs}


def run_experiment(
    graph_kind: str,
    cost_kind: str,
    expiry: Optional[int],
    n: int = 20,
    seed: int = 1,
    strategy: str = "constant",
    constant_cost: float = 1.0,
    distance_factor: float = 0.8,
    max_steps: int = 2000,
    refresh_threshold: Optional[float] = 0.8,
    include_details: bool = False,
) -> Dict[str, object]:
    rng = random.Random(seed)
    G = generate_graph(graph_kind, n, seed)
    nodes = list(G.nodes())
    initial_edges = list(G.edges())

    production = {node: rng.randint(6, 14) for node in nodes}

    if cost_kind == "zero":
        edge_costs = make_zero_costs(G)
    elif cost_kind == "constant":
        edge_costs = make_constant_costs(G, constant_cost)
    elif cost_kind == "distance":
        edge_costs = make_distance_costs(G, distance_factor)
    else:
        raise ValueError(f"Unknown cost kind: {cost_kind}")

    deg_rng = random.Random(seed + 91)
    edge_degradations = build_edge_degradations(edge_costs, deg_rng)

    if expiry is None:
        edge_lifetimes = {edge: None for edge in edge_costs}
        expiry_value = 5
    else:
        edge_lifetimes = {edge: int(expiry) for edge in edge_costs}
        expiry_value = int(expiry)

    sim = DecentralizedExchangeSimulator(
        agents=nodes,
        production=production,
        initial_edges=initial_edges,
        strategy=strategy,
        edge_costs=edge_costs,
        edge_lifetimes=edge_lifetimes,
        edge_degradations=edge_degradations,
        expiry=expiry_value,
        degradation=0.95,
        refresh_threshold=refresh_threshold,
        discount=0.9,
        max_new_links_per_slot=1,
        rng_seed=seed,
    )

    result = sim.find_equilibrium_states(max_steps=max_steps)
    if result is None:
        states = [sim.state]
        equilibrium_found = False
        equilibrium_start = None
        equilibrium_period = None
    else:
        equilibrium_start, equilibrium_period, states = result
        equilibrium_found = True

    utility_acc: Dict[int, float] = {agent: 0.0 for agent in nodes}
    allocation_acc: Dict[Tuple[int, int], float] = {}
    for state in states:
        for agent, value in state.utilities.items():
            utility_acc[agent] += value
        for pair, amount in state.last_allocation.items():
            allocation_acc[pair] = allocation_acc.get(pair, 0.0) + amount

    count = float(len(states))
    utilities_avg = {agent: value / count for agent, value in utility_acc.items()}
    allocations_avg = {pair: amount / count for pair, amount in allocation_acc.items()}

    welfare_others = welfare_of_others(utilities_avg)
    welfare_others_avg = sum(welfare_others.values()) / len(welfare_others) if welfare_others else 0.0
    fairness = jain_fairness(utilities_avg.values())

    payload: Dict[str, object] = {
        "equilibrium_found": equilibrium_found,
        "equilibrium_start": equilibrium_start,
        "equilibrium_period": equilibrium_period,
        "welfare_others_avg": welfare_others_avg,
        "fairness_jain": fairness,
    }

    if include_details:
        representative = states[0] if states else sim.state
        edges = sorted(representative.links.keys(), key=lambda item: (str(item[0]), str(item[1])))
        payload["utilities"] = utilities_avg
        payload["welfare_others"] = welfare_others
        payload["allocations"] = allocations_avg
        payload["nodes"] = nodes
        payload["edges"] = edges

    return payload


def print_header():
    print(
        "expiry,equilibrium_found,equilibrium_start,equilibrium_period,welfare_others_avg,fairness_jain"
    )


def print_result(expiry: Optional[int], result: Dict[str, object]):
    expiry_label = "permanent" if expiry is None else str(expiry)
    print(
        f"{expiry_label},"
        f"{result['equilibrium_found']},"
        f"{result['equilibrium_start']},"
        f"{result['equilibrium_period']},"
        f"{result['welfare_others_avg']:.6f},"
        f"{result['fairness_jain']:.6f}"
    )

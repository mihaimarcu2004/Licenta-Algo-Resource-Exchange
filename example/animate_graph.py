from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import (
    COST_BANDS,
    build_edge_degradations,
    build_edge_lifetimes,
    generate_graph,
    make_constant_costs,
    make_degree_costs,
    make_distance_costs,
    make_mixed_costs,
    make_random_costs,
    make_zero_costs,
    sample_production,
)
from exchange_sim import DecentralizedExchangeSimulator

STRATEGIES = {
    "egalitarian": "constant",
    "proportional_1": "proportional_constant_bootstrap",
    "proportional_2": "proportional_decaying_boost",
}


def build_simulator(
    graph_kind: str,
    cost_kind: str,
    expiry: Optional[int],
    n: int,
    seed: int,
    strategy_label: str,
    constant_cost: float = 1.0,
    distance_factor: float = 0.8,
    refresh_threshold: Optional[float] = 0.8,
    lifetime_kind: str = "small_lifetime",
    min_link_flow: float = 1e-9,
) -> Tuple[DecentralizedExchangeSimulator, nx.Graph]:
    rng = random.Random(seed)
    graph = generate_graph(graph_kind, n, seed)
    nodes = list(graph.nodes())
    initial_edges = list(graph.edges())
    production = {node: sample_production(rng) for node in nodes}

    cost_rng = random.Random(seed + 37)
    if cost_kind in COST_BANDS:
        low, high = COST_BANDS[cost_kind]
        edge_costs = make_random_costs(graph, cost_rng, low, high)
    elif cost_kind == "zero":
        edge_costs = make_zero_costs(graph)
    elif cost_kind == "constant":
        edge_costs = make_constant_costs(graph, constant_cost)
    elif cost_kind == "distance":
        edge_costs = make_distance_costs(graph, distance_factor)
    elif cost_kind == "degree":
        edge_costs = make_degree_costs(graph)
    elif cost_kind == "mixed":
        edge_costs = make_mixed_costs(graph, cost_rng)
    else:
        raise ValueError(f"Unknown cost kind: {cost_kind}")

    deg_rng = random.Random(seed + 91)
    edge_degradations = build_edge_degradations(edge_costs, deg_rng)
    lifetime_rng = random.Random(seed + 131)
    edge_lifetimes = build_edge_lifetimes(edge_costs, lifetime_rng, expiry, lifetime_kind)
    expiry_value = 5 if expiry is None else int(expiry)

    strategy = STRATEGIES.get(strategy_label, strategy_label)
    simulator = DecentralizedExchangeSimulator(
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
        min_link_flow=min_link_flow,
        discount=0.9,
        max_new_links_per_slot=None,
        rng_seed=seed,
    )
    return simulator, graph


def snapshot(simulator: DecentralizedExchangeSimulator) -> Dict[str, object]:
    return {
        "t": simulator.state.t,
        "edges": {
            edge: {
                "quality": link.quality,
                "initial": link.initial,
                "remaining_life": link.remaining_life,
                "created_at": link.created_at,
            }
            for edge, link in simulator.state.links.items()
        },
        "utilities": dict(simulator.state.utilities),
    }


def collect_snapshots(simulator: DecentralizedExchangeSimulator, steps: int) -> list[Dict[str, object]]:
    frames = [snapshot(simulator)]
    for _ in range(steps):
        simulator.step()
        frames.append(snapshot(simulator))
    return frames


def visual_groups(graph_kind: str, graph: nx.Graph, agents: list[object]) -> Dict[object, int]:
    if graph_kind == "two_communities":
        split = len(agents) // 2
        return {agent: 0 if int(agent) < split else 1 for agent in agents}
    if graph_kind == "core_periphery":
        core_size = max(3, len(agents) // 4)
        return {agent: 0 if int(agent) < core_size else 1 for agent in agents}

    try:
        from networkx.algorithms.community import greedy_modularity_communities

        communities = greedy_modularity_communities(graph)
    except Exception:
        communities = [set(agents)]

    groups: Dict[object, int] = {}
    for group_idx, community in enumerate(communities):
        for agent in community:
            groups[agent] = group_idx
    for agent in agents:
        groups.setdefault(agent, 0)
    return groups


def save_animation(
    frames: list[Dict[str, object]],
    base_graph: nx.Graph,
    simulator: DecentralizedExchangeSimulator,
    output_path: Path,
    fps: int,
    seed: int,
    title: str,
    graph_kind: str,
) -> Path:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    all_edges = set(base_graph.edges()) | set(simulator.edge_costs)
    layout_graph = nx.Graph()
    layout_graph.add_nodes_from(simulator.agents)
    layout_graph.add_edges_from(all_edges)
    pos = nx.spring_layout(layout_graph, seed=seed)

    max_production = max(simulator.production.values()) if simulator.production else 1.0
    node_sizes = [220 + 380 * simulator.production[node] / max_production for node in simulator.agents]
    groups = visual_groups(graph_kind, base_graph, simulator.agents)
    palette = ["#e8f2ff", "#fff1d6", "#e8f7e8", "#f6e8ff", "#ffe8ee", "#e8fbff"]
    node_colors = [palette[groups[node] % len(palette)] for node in simulator.agents]

    fig, ax = plt.subplots(figsize=(8, 7))

    def draw(frame_idx: int) -> None:
        frame = frames[frame_idx]
        active_edges = frame["edges"]
        graph = nx.Graph()
        graph.add_nodes_from(simulator.agents)
        graph.add_edges_from(active_edges)

        ax.clear()
        ax.set_title(f"{title}\nt = {frame['t']} | active links = {len(active_edges)}")
        ax.axis("off")

        inactive_edges = [edge for edge in all_edges if edge not in active_edges]
        nx.draw_networkx_edges(
            layout_graph,
            pos,
            edgelist=inactive_edges,
            edge_color="#d0d0d0",
            width=0.8,
            alpha=0.25,
            ax=ax,
        )

        initial_edges = [edge for edge, data in active_edges.items() if data["initial"]]
        created_edges = [edge for edge, data in active_edges.items() if not data["initial"]]

        def edge_width(edge: Tuple[object, object]) -> float:
            quality = active_edges[edge]["quality"]
            return 0.8 + 3.2 * float(quality)

        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=initial_edges,
            edge_color="#555555",
            width=[edge_width(edge) for edge in initial_edges],
            alpha=0.75,
            ax=ax,
        )
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=created_edges,
            edge_color="#1976d2",
            width=[edge_width(edge) for edge in created_edges],
            alpha=0.85,
            ax=ax,
        )
        nx.draw_networkx_nodes(
            graph,
            pos,
            node_size=node_sizes,
            node_color=node_colors,
            edgecolors="#1f2937",
            linewidths=1.2,
            ax=ax,
        )
        nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)

        utilities = frame["utilities"]
        utility_text = ", ".join(
            f"{node}: {utilities.get(node, 0.0):.1f}" for node in sorted(simulator.agents, key=str)[:8]
        )
        if len(simulator.agents) > 8:
            utility_text += ", ..."
        ax.text(
            0.01,
            0.01,
            f"Utilities: {utility_text}\nBlue = created links, gray = initial links",
            transform=ax.transAxes,
            fontsize=8,
            ha="left",
            va="bottom",
        )

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps, repeat=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if output_path.suffix.lower() != ".mp4":
            output_path = output_path.with_suffix(".mp4")
        anim.save(output_path, writer="ffmpeg", fps=fps, dpi=150)
    except Exception:
        output_path = output_path.with_suffix(".gif")
        anim.save(output_path, writer="pillow", fps=fps, dpi=120)
    finally:
        plt.close(fig)

    return output_path


def parse_expiry(value: str) -> Optional[int]:
    if value == "permanent":
        return None
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate graph evolution for one exchange simulation.")
    parser.add_argument(
        "--graph-kind",
        choices=["sparse", "small_world", "scale_free", "two_communities", "core_periphery"],
        default="sparse",
    )
    parser.add_argument(
        "--cost-kind",
        choices=["small", "medium", "big", "zero", "constant", "distance", "degree", "mixed"],
        default="small",
    )
    parser.add_argument("--expiry", type=parse_expiry, default=5)
    parser.add_argument(
        "--lifetime-kind",
        choices=[
            "small_lifetime",
            "medium_lifetime",
            "large_lifetime",
            "mixed_lifetime",
            "same_lifetime",
            "permanent",
        ],
        default="small_lifetime",
    )
    parser.add_argument("--strategy", choices=list(STRATEGIES), default="proportional_1")
    parser.add_argument("--nodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--min-link-flow", type=float, default=1e-9)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    simulator, graph = build_simulator(
        graph_kind=args.graph_kind,
        cost_kind=args.cost_kind,
        expiry=args.expiry,
        n=args.nodes,
        seed=args.seed,
        strategy_label=args.strategy,
        lifetime_kind=args.lifetime_kind,
        min_link_flow=args.min_link_flow,
    )
    frames = collect_snapshots(simulator, args.steps)

    expiry_label = "0" if args.expiry is None else str(args.expiry)
    output_path = args.output
    if output_path is None:
        filename = (
            f"{args.graph_kind}_{args.cost_kind}_{expiry_label}_{args.lifetime_kind}_{args.strategy}_evolution.mp4"
        )
        output_path = Path("example/results") / filename

    title = f"{args.graph_kind}, {args.cost_kind}, expiry={expiry_label}, {args.lifetime_kind}, {args.strategy}"
    saved_path = save_animation(frames, graph, simulator, output_path, args.fps, args.seed, title, args.graph_kind)
    print(saved_path)


if __name__ == "__main__":
    main()

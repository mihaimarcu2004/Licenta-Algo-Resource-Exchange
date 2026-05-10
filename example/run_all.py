from pathlib import Path

import networkx as nx

from common import run_experiment

GRAPH_KINDS = ["sparse", "dense", "small_world"]
COST_KINDS = ["small", "medium", "big"]
EXPIRIES = [None, 1, 3, 5, 10, 100]


if __name__ == "__main__":
    results_dir = Path("example/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "summary.csv"
    summary_header = (
        "graph_kind,cost_kind,expiry,"
        "equilibrium_found,equilibrium_start,equilibrium_period,"
        "welfare_eisenberg_gale,fairness_jain"
    )
    print(summary_header)

    with summary_path.open("w", encoding="utf-8") as summary_file:
        summary_file.write(summary_header + "\n")

        combo_idx = 0
        for graph_kind in GRAPH_KINDS:
            for cost_kind in COST_KINDS:
                for expiry in EXPIRIES:
                    seed = 1000 + combo_idx
                    combo_idx += 1
                    result = run_experiment(
                        graph_kind=graph_kind,
                        cost_kind=cost_kind,
                        expiry=expiry,
                        n=20,
                        seed=seed,
                        include_details=True,
                    )
                    expiry_label = "permanent" if expiry is None else str(expiry)
                    line = (
                        f"{graph_kind},{cost_kind},{expiry_label},"
                        f"{result['equilibrium_found']},"
                        f"{result['equilibrium_start']},"
                        f"{result['equilibrium_period']},"
                        f"{result['welfare_eisenberg_gale']:.6f},"
                        f"{result['fairness_jain']:.6f}"
                    )
                    print(line)
                    summary_file.write(line + "\n")

                    utilities = result.get("utilities", {})
                    welfare_others = result.get("welfare_others", {})
                    allocations = result.get("allocations", {})
                    nodes = result.get("nodes", [])
                    edges = result.get("edges", [])

                    nodes_path = results_dir / f"{graph_kind}_{cost_kind}_{expiry_label}_nodes.csv"
                    with nodes_path.open("w", encoding="utf-8") as node_file:
                        node_file.write("node,utility_avg,welfare_others\n")
                        for node in sorted(utilities, key=str):
                            node_file.write(
                                f"{node},{utilities[node]:.6f},{welfare_others.get(node, 0.0):.6f}\n"
                            )

                    alloc_path = results_dir / f"{graph_kind}_{cost_kind}_{expiry_label}_allocations.csv"
                    with alloc_path.open("w", encoding="utf-8") as alloc_file:
                        alloc_file.write("sender,receiver,amount_avg\n")
                        for (sender, receiver), amount in sorted(allocations.items(), key=lambda item: str(item[0])):
                            alloc_file.write(f"{sender},{receiver},{amount:.6f}\n")

                    edges_path = results_dir / f"{graph_kind}_{cost_kind}_{expiry_label}_edges.csv"
                    with edges_path.open("w", encoding="utf-8") as edges_file:
                        edges_file.write("u,v\n")
                        for u, v in edges:
                            edges_file.write(f"{u},{v}\n")

                    try:
                        import matplotlib.pyplot as plt
                    except Exception:
                        plt = None

                    if plt is not None:
                        H = nx.Graph()
                        H.add_nodes_from(nodes)
                        H.add_edges_from(edges)
                        pos = nx.spring_layout(H, seed=seed)
                        plt.figure(figsize=(6, 6))
                        nx.draw_networkx(H, pos=pos, node_size=200, font_size=6, with_labels=True)
                        plt.axis("off")
                        plt.tight_layout()
                        fig_path = results_dir / f"{graph_kind}_{cost_kind}_{expiry_label}_graph.png"
                        plt.savefig(fig_path, dpi=150)
                        plt.close()

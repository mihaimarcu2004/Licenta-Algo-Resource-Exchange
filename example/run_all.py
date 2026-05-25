from pathlib import Path

import networkx as nx

from common import run_experiment

GRAPH_KINDS = ["sparse", "small_world", "scale_free", "two_communities", "core_periphery"]
COST_KINDS = ["small", "medium", "big", "distance", "degree", "mixed"]
LIFETIME_CONFIGS = [
    ("small_lifetime", 3, "small_lifetime"),
    ("medium_lifetime", 7, "medium_lifetime"),
    ("large_lifetime", 12, "large_lifetime"),
    ("xl_lifetime", 20, "xl_lifetime"),
    ("xxl_lifetime", 100, "xxl_lifetime"),
    ("mixed_lifetime", 100, "mixed_lifetime"),
    ("permanent", None, "permanent"),
]
STRATEGIES = [
    ("egalitarian", "constant"),
    ("proportional_1", "proportional_constant_bootstrap"),
    ("proportional_2", "proportional_decaying_boost"),
]


if __name__ == "__main__":
    results_dir = Path("example/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "summary.csv"
    comparison_path = results_dir / "strategy_comparison.csv"
    summary_header = (
        "graph_kind,cost_kind,lifetime_config,expiry,lifetime_kind,strategy,"
        "equilibrium_found,equilibrium_start,equilibrium_period,"
        "measurement_window,measurement_state_count,average_utility,"
        "welfare_eisenberg_gale,welfare_net,fairness_jain,fairness_exchange_ratio_jain"
    )
    comparison_header = (
        "graph_kind,cost_kind,lifetime_config,expiry,lifetime_kind,"
        "egalitarian_average_utility,egalitarian_welfare_eisenberg_gale,egalitarian_welfare_net,egalitarian_fairness_exchange_ratio_jain,"
        "proportional_1_average_utility,proportional_1_welfare_eisenberg_gale,proportional_1_welfare_net,proportional_1_fairness_exchange_ratio_jain,"
        "proportional_2_average_utility,proportional_2_welfare_eisenberg_gale,proportional_2_welfare_net,proportional_2_fairness_exchange_ratio_jain,"
        "best_welfare_strategy,best_net_welfare_strategy,best_fairness_strategy"
    )
    print(summary_header)

    with summary_path.open("w", encoding="utf-8") as summary_file, comparison_path.open(
        "w",
        encoding="utf-8",
    ) as comparison_file:
        summary_file.write(summary_header + "\n")
        comparison_file.write(comparison_header + "\n")

        combo_idx = 0
        for graph_kind in GRAPH_KINDS:
            for cost_kind in COST_KINDS:
                for lifetime_config, expiry, lifetime_kind in LIFETIME_CONFIGS:
                    seed = 1000 + combo_idx
                    combo_idx += 1
                    expiry_label = "0" if expiry is None else str(expiry)
                    strategy_results = {}

                    for strategy_label, strategy_name in STRATEGIES:
                        result = run_experiment(
                            graph_kind=graph_kind,
                            cost_kind=cost_kind,
                            expiry=expiry,
                            n=20,
                            seed=seed,
                            strategy=strategy_name,
                            max_steps=100000,
                            lifetime_kind=lifetime_kind,
                            include_details=True,
                        )
                        strategy_results[strategy_label] = result
                        line = (
                            f"{graph_kind},{cost_kind},{lifetime_config},{expiry_label},{lifetime_kind},{strategy_label},"
                            f"{result['equilibrium_found']},"
                            f"{result['equilibrium_start']},"
                            f"{result['equilibrium_period']},"
                            f"{result['measurement_window']},"
                            f"{result['measurement_state_count']},"
                            f"{result['average_utility']:.6f},"
                            f"{result['welfare_eisenberg_gale']:.6f},"
                            f"{result['welfare_net']:.6f},"
                            f"{result['fairness_jain']:.6f},"
                            f"{result['fairness_exchange_ratio_jain']:.6f}"
                        )
                        print(line)
                        summary_file.write(line + "\n")

                        utilities = result.get("utilities", {})
                        exchange_ratios = result.get("exchange_ratios", {})
                        welfare_others = result.get("welfare_others", {})
                        allocations = result.get("allocations", {})
                        renewals = result.get("renewals", {})
                        nodes = result.get("nodes", [])
                        edges = result.get("edges", [])
                        file_prefix = f"{graph_kind}_{cost_kind}_{lifetime_config}_{strategy_label}"

                        nodes_path = results_dir / f"{file_prefix}_nodes.csv"
                        with nodes_path.open("w", encoding="utf-8") as node_file:
                            node_file.write("node,utility_avg,exchange_ratio_avg,welfare_others\n")
                            for node in sorted(utilities, key=str):
                                node_file.write(
                                    f"{node},{utilities[node]:.6f},{exchange_ratios.get(node, 0.0):.6f},"
                                    f"{welfare_others.get(node, 0.0):.6f}\n"
                                )

                        alloc_path = results_dir / f"{file_prefix}_allocations.csv"
                        with alloc_path.open("w", encoding="utf-8") as alloc_file:
                            alloc_file.write("sender,receiver,amount_avg\n")
                            for (sender, receiver), amount in sorted(allocations.items(), key=lambda item: str(item[0])):
                                alloc_file.write(f"{sender},{receiver},{amount:.6f}\n")

                        edges_path = results_dir / f"{file_prefix}_edges.csv"
                        with edges_path.open("w", encoding="utf-8") as edges_file:
                            edges_file.write("u,v\n")
                            for u, v in edges:
                                edges_file.write(f"{u},{v}\n")

                        renewals_path = results_dir / f"{file_prefix}_renewals.csv"
                        with renewals_path.open("w", encoding="utf-8") as renewals_file:
                            renewals_file.write("agent,neighbor,renewal_decision_avg\n")
                            for (agent, neighbor), decision in sorted(renewals.items(), key=lambda item: str(item[0])):
                                renewals_file.write(f"{agent},{neighbor},{decision:.6f}\n")

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
                            fig_path = results_dir / f"{file_prefix}_graph.png"
                            plt.savefig(fig_path, dpi=150)
                            plt.close()

                    best_welfare_strategy = max(
                        strategy_results,
                        key=lambda label: strategy_results[label]["welfare_eisenberg_gale"],
                    )
                    best_fairness_strategy = max(
                        strategy_results,
                        key=lambda label: strategy_results[label]["fairness_exchange_ratio_jain"],
                    )
                    best_net_welfare_strategy = max(
                        strategy_results,
                        key=lambda label: strategy_results[label]["welfare_net"],
                    )
                    comparison_line = (
                        f"{graph_kind},{cost_kind},{lifetime_config},{expiry_label},{lifetime_kind},"
                        f"{strategy_results['egalitarian']['average_utility']:.6f},"
                        f"{strategy_results['egalitarian']['welfare_eisenberg_gale']:.6f},"
                        f"{strategy_results['egalitarian']['welfare_net']:.6f},"
                        f"{strategy_results['egalitarian']['fairness_exchange_ratio_jain']:.6f},"
                        f"{strategy_results['proportional_1']['average_utility']:.6f},"
                        f"{strategy_results['proportional_1']['welfare_eisenberg_gale']:.6f},"
                        f"{strategy_results['proportional_1']['welfare_net']:.6f},"
                        f"{strategy_results['proportional_1']['fairness_exchange_ratio_jain']:.6f},"
                        f"{strategy_results['proportional_2']['average_utility']:.6f},"
                        f"{strategy_results['proportional_2']['welfare_eisenberg_gale']:.6f},"
                        f"{strategy_results['proportional_2']['welfare_net']:.6f},"
                        f"{strategy_results['proportional_2']['fairness_exchange_ratio_jain']:.6f},"
                        f"{best_welfare_strategy},{best_net_welfare_strategy},{best_fairness_strategy}"
                    )
                    comparison_file.write(comparison_line + "\n")

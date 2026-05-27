import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple

from common import run_experiment

GRAPH_KINDS = ["sparse", "dense", "small_world"]
COST_KINDS = ["small", "medium", "big"]
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

EV_HEURISTICS = [
    "proportional_screening",
    "top_k_candidates",
    "adaptive_horizon",
    "cooldown_rejections",
    "hysteresis_threshold",
    "skip_inactive_pruning",
]

N_AGENTS = int(os.environ.get("RUN_ALL_N_AGENTS", "16"))
MAX_STEPS = int(os.environ.get("RUN_ALL_MAX_STEPS", "2000"))
RUN_WORKERS = int(os.environ.get("RUN_ALL_WORKERS", str(max(min((os.cpu_count() or 2) - 1, 4), 1))))
LINK_EVALUATION_WORKERS = int(os.environ.get("RUN_ALL_LINK_WORKERS", "1"))
TASK_LIMIT = int(os.environ.get("RUN_ALL_LIMIT", "0"))
RUN_MODE = os.environ.get("RUN_ALL_MODE", "process").lower()
GENERATE_GRAPH_PNGS = os.environ.get("RUN_ALL_GRAPHS", "0") == "1"

SUMMARY_HEADER = (
    "graph_kind,cost_kind,lifetime_config,expiry,lifetime_kind,strategy,"
    "equilibrium_found,equilibrium_start,equilibrium_period,"
    "measurement_window,measurement_state_count,average_utility,"
    "welfare_eisenberg_gale,welfare_net,fairness_jain,fairness_exchange_ratio_jain"
)
COMPARISON_HEADER = (
    "graph_kind,cost_kind,lifetime_config,expiry,lifetime_kind,"
    "egalitarian_average_utility,egalitarian_welfare_eisenberg_gale,egalitarian_welfare_net,egalitarian_fairness_exchange_ratio_jain,"
    "proportional_1_average_utility,proportional_1_welfare_eisenberg_gale,proportional_1_welfare_net,proportional_1_fairness_exchange_ratio_jain,"
    "proportional_2_average_utility,proportional_2_welfare_eisenberg_gale,proportional_2_welfare_net,proportional_2_fairness_exchange_ratio_jain,"
    "best_welfare_strategy,best_net_welfare_strategy,best_fairness_strategy"
)

ExperimentKey = Tuple[str, str, str, Optional[int], str, int]


def _make_tasks():
    tasks = []
    combo_idx = 0
    for graph_kind in GRAPH_KINDS:
        for cost_kind in COST_KINDS:
            for lifetime_config, expiry, lifetime_kind in LIFETIME_CONFIGS:
                seed = 1000 + combo_idx
                combo_idx += 1
                key = (graph_kind, cost_kind, lifetime_config, expiry, lifetime_kind, seed)
                for strategy_label, strategy_name in STRATEGIES:
                    tasks.append((key, strategy_label, strategy_name))
    return tasks


def _run_task(task):
    key, strategy_label, strategy_name = task
    graph_kind, cost_kind, _lifetime_config, expiry, lifetime_kind, seed = key
    result = run_experiment(
        graph_kind=graph_kind,
        cost_kind=cost_kind,
        expiry=expiry,
        n=N_AGENTS,
        seed=seed,
        strategy=strategy_name,
        max_steps=MAX_STEPS,
        lifetime_kind=lifetime_kind,
        ev_heuristics=EV_HEURISTICS,
        link_evaluation_workers=LINK_EVALUATION_WORKERS,
        include_details=True,
    )
    return key, strategy_label, result


def _run_tasks_serial(tasks):
    for task in tasks:
        yield _run_task(task)


def _run_tasks_process(tasks):
    with ProcessPoolExecutor(max_workers=RUN_WORKERS) as executor:
        future_to_task = {executor.submit(_run_task, task): task for task in tasks}
        for future in as_completed(future_to_task):
            yield future.result()


def _expiry_label(expiry: Optional[int]) -> str:
    return "0" if expiry is None else str(expiry)


def _summary_line(key: ExperimentKey, strategy_label: str, result: Dict[str, object]) -> str:
    graph_kind, cost_kind, lifetime_config, expiry, lifetime_kind, _seed = key
    return (
        f"{graph_kind},{cost_kind},{lifetime_config},{_expiry_label(expiry)},{lifetime_kind},{strategy_label},"
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


def _write_detail_files(results_dir: Path, key: ExperimentKey, strategy_label: str, result: Dict[str, object]) -> None:
    graph_kind, cost_kind, lifetime_config, _expiry, _lifetime_kind, _seed = key
    file_prefix = f"{graph_kind}_{cost_kind}_{lifetime_config}_{strategy_label}"

    utilities = result.get("utilities", {})
    exchange_ratios = result.get("exchange_ratios", {})
    welfare_others = result.get("welfare_others", {})
    allocations = result.get("allocations", {})
    renewals = result.get("renewals", {})
    nodes = result.get("nodes", [])
    edges = result.get("edges", [])

    with (results_dir / f"{file_prefix}_nodes.csv").open("w", encoding="utf-8") as node_file:
        node_file.write("node,utility_avg,exchange_ratio_avg,welfare_others\n")
        for node in sorted(utilities, key=str):
            node_file.write(
                f"{node},{utilities[node]:.6f},{exchange_ratios.get(node, 0.0):.6f},"
                f"{welfare_others.get(node, 0.0):.6f}\n"
            )

    with (results_dir / f"{file_prefix}_allocations.csv").open("w", encoding="utf-8") as alloc_file:
        alloc_file.write("sender,receiver,amount_avg\n")
        for (sender, receiver), amount in sorted(allocations.items(), key=lambda item: str(item[0])):
            alloc_file.write(f"{sender},{receiver},{amount:.6f}\n")

    with (results_dir / f"{file_prefix}_edges.csv").open("w", encoding="utf-8") as edges_file:
        edges_file.write("u,v\n")
        for u, v in edges:
            edges_file.write(f"{u},{v}\n")

    with (results_dir / f"{file_prefix}_renewals.csv").open("w", encoding="utf-8") as renewals_file:
        renewals_file.write("agent,neighbor,renewal_decision_avg\n")
        for (agent, neighbor), decision in sorted(renewals.items(), key=lambda item: str(item[0])):
            renewals_file.write(f"{agent},{neighbor},{decision:.6f}\n")

    if GENERATE_GRAPH_PNGS:
        _write_graph_png(results_dir, file_prefix, nodes, edges, key[-1])


def _write_graph_png(results_dir: Path, file_prefix: str, nodes, edges, seed: int) -> None:
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except Exception:
        return

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    pos = nx.spring_layout(graph, seed=seed)
    plt.figure(figsize=(6, 6))
    nx.draw_networkx(graph, pos=pos, node_size=200, font_size=6, with_labels=True)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(results_dir / f"{file_prefix}_graph.png", dpi=150)
    plt.close()


def _comparison_line(key: ExperimentKey, strategy_results: Dict[str, Dict[str, object]]) -> str:
    graph_kind, cost_kind, lifetime_config, expiry, lifetime_kind, _seed = key
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
    return (
        f"{graph_kind},{cost_kind},{lifetime_config},{_expiry_label(expiry)},{lifetime_kind},"
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


if __name__ == "__main__":
    results_dir = Path("example/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    tasks = _make_tasks()
    if TASK_LIMIT > 0:
        tasks = tasks[:TASK_LIMIT]
    grouped_results: Dict[ExperimentKey, Dict[str, Dict[str, object]]] = {}

    print(SUMMARY_HEADER)
    print(
        f"running {len(tasks)} experiments; mode={RUN_MODE}; workers={RUN_WORKERS}; "
        f"n={N_AGENTS}; max_steps={MAX_STEPS}"
    )

    completed = 0
    try:
        iterator = _run_tasks_serial(tasks) if RUN_MODE == "serial" or RUN_WORKERS <= 1 else _run_tasks_process(tasks)
        for key, strategy_label, result in iterator:
            grouped_results.setdefault(key, {})[strategy_label] = result
            completed += 1
            print(f"{completed}/{len(tasks)} " + _summary_line(key, strategy_label, result), flush=True)
    except (OSError, PermissionError) as exc:
        if completed:
            raise
        print(f"process mode failed ({exc}); falling back to serial mode", flush=True)
        for key, strategy_label, result in _run_tasks_serial(tasks):
            grouped_results.setdefault(key, {})[strategy_label] = result
            completed += 1
            print(f"{completed}/{len(tasks)} " + _summary_line(key, strategy_label, result), flush=True)

    summary_path = results_dir / "summary.csv"
    comparison_path = results_dir / "strategy_comparison.csv"
    with summary_path.open("w", encoding="utf-8") as summary_file, comparison_path.open(
        "w",
        encoding="utf-8",
    ) as comparison_file:
        summary_file.write(SUMMARY_HEADER + "\n")
        comparison_file.write(COMPARISON_HEADER + "\n")

        for key in sorted(grouped_results, key=str):
            strategy_results = grouped_results[key]
            for strategy_label, _strategy_name in STRATEGIES:
                if strategy_label not in strategy_results:
                    continue
                result = strategy_results[strategy_label]
                summary_file.write(_summary_line(key, strategy_label, result) + "\n")
                _write_detail_files(results_dir, key, strategy_label, result)
            if all(strategy_label in strategy_results for strategy_label, _ in STRATEGIES):
                comparison_file.write(_comparison_line(key, strategy_results) + "\n")

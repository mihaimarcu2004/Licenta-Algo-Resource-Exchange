import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from common import run_experiment

GRAPH_KINDS = ["sparse", "dense", "small_world"]
COST_KINDS = ["small", "medium", "big"]
LIFETIME_CONFIGS = [
    ("small_lifetime", 3, "small_lifetime"),
    ("medium_lifetime", 7, "medium_lifetime"),
    ("large_lifetime", 12, "large_lifetime"),
    ("mixed_lifetime", 7, "mixed_lifetime"),
    ("same_lifetime_3", 3, "same_lifetime"),
    ("same_lifetime_7", 7, "same_lifetime"),
]
STRATEGIES = [
    ("egalitarian", "constant"),
    ("proportional_1", "proportional_constant_bootstrap"),
    ("proportional_2", "proportional_decaying_boost"),
]
LINK_FORMATION_MODES = [
    ("without_link_formation", False),
    ("with_link_formation", True),
]

EV_HEURISTICS = [
    "adaptive_horizon",
    "skip_inactive_pruning",
]

N_AGENTS = int(os.environ.get("RUN_ALL_N_AGENTS", "16"))
MAX_STEPS = int(os.environ.get("RUN_ALL_MAX_STEPS", "10000"))
EQUILIBRIUM_PRECISION = int(os.environ.get("RUN_ALL_EQUILIBRIUM_PRECISION", "3"))
EQUILIBRIUM_EPS = float(os.environ.get("RUN_ALL_EQUILIBRIUM_EPS", "0.001"))
MIN_LINK_FLOW = float(os.environ.get("RUN_ALL_MIN_LINK_FLOW", "1e-4"))
MAX_CANDIDATE_EDGES_PER_AGENT = int(os.environ.get("RUN_ALL_MAX_CANDIDATE_EDGES_PER_AGENT", "5"))
COST_MULTIPLIER = float(os.environ.get("RUN_ALL_COST_MULTIPLIER", "1.0"))
WELFARE_TIME_INTERVAL = int(os.environ.get("RUN_ALL_WELFARE_INTERVAL", "100"))
WELFARE_PLOT_MAX_STEP = int(os.environ.get("RUN_ALL_WELFARE_PLOT_MAX_STEP", "2500"))
WITHOUT_LINK_FORMATION_MIN_STEPS = int(os.environ.get("RUN_ALL_WITHOUT_LINK_FORMATION_MIN_STEPS", "2500"))
BASE_SEED = int(os.environ.get("RUN_ALL_BASE_SEED", "1000"))
SEED_COUNT = int(os.environ.get("RUN_ALL_SEED_COUNT", "1"))
TASK_LIMIT = int(os.environ.get("RUN_ALL_LIMIT", "0"))
GENERATE_GRAPH_PNGS = os.environ.get("RUN_ALL_GRAPHS", "0") == "1"
GENERATE_WELFARE_PLOTS = os.environ.get("RUN_ALL_WELFARE_PLOTS", "1") == "1"

SUMMARY_HEADER = (
    "graph_kind,cost_kind,lifetime_config,expiry,lifetime_kind,seed,link_formation,strategy,"
    "equilibrium_found,equilibrium_start,equilibrium_period,"
    "measurement_window,measurement_state_count,average_utility,average_degree,"
    "welfare_eisenberg_gale,welfare_net,fairness_jain,fairness_exchange_ratio_jain"
)
COMPARISON_HEADER = (
    "graph_kind,cost_kind,lifetime_config,expiry,lifetime_kind,seed,link_formation,"
    "egalitarian_average_utility,egalitarian_welfare_eisenberg_gale,egalitarian_welfare_net,egalitarian_fairness_exchange_ratio_jain,"
    "proportional_1_average_utility,proportional_1_welfare_eisenberg_gale,proportional_1_welfare_net,proportional_1_fairness_exchange_ratio_jain,"
    "proportional_2_average_utility,proportional_2_welfare_eisenberg_gale,proportional_2_welfare_net,proportional_2_fairness_exchange_ratio_jain,"
    "best_welfare_strategy,best_net_welfare_strategy,best_fairness_strategy"
)
TIME_SERIES_HEADER = (
    "graph_kind,cost_kind,lifetime_config,expiry,lifetime_kind,seed,link_formation,strategy,t,"
    "average_utility,welfare_eisenberg_gale,welfare_net,"
    "fairness_jain,fairness_exchange_ratio_jain,active_edges"
)
AGGREGATE_METRICS = [
    "average_utility",
    "welfare_eisenberg_gale",
    "welfare_net",
    "fairness_jain",
    "fairness_exchange_ratio_jain",
    "active_edges",
]
LINK_FORMATION_COMPARISON_HEADER = (
    "graph_kind,cost_kind,lifetime_config,expiry,lifetime_kind,seed,strategy,"
    "without_average_utility,with_average_utility,delta_average_utility,pct_delta_average_utility,"
    "without_welfare_eisenberg_gale,with_welfare_eisenberg_gale,delta_welfare_eisenberg_gale,"
    "without_welfare_net,with_welfare_net,delta_welfare_net,"
    "without_fairness_jain,with_fairness_jain,delta_fairness_jain,"
    "without_fairness_exchange_ratio_jain,with_fairness_exchange_ratio_jain,delta_fairness_exchange_ratio_jain"
)

ExperimentKey = Tuple[str, str, str, Optional[int], str, int, str]
BaseExperimentKey = Tuple[str, str, str, Optional[int], str, int]


def _parse_seed_list() -> List[int]:
    raw = os.environ.get("RUN_ALL_SEEDS")
    if raw:
        seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
    else:
        if SEED_COUNT <= 0:
            raise ValueError("RUN_ALL_SEED_COUNT must be positive.")
        seeds = [BASE_SEED + offset for offset in range(SEED_COUNT)]
    if not seeds:
        raise ValueError("At least one seed must be configured.")
    return seeds


RANDOM_SEEDS = _parse_seed_list()


def _make_tasks():
    tasks = []
    for graph_kind in GRAPH_KINDS:
        for cost_kind in COST_KINDS:
            for lifetime_config, expiry, lifetime_kind in LIFETIME_CONFIGS:
                for seed in RANDOM_SEEDS:
                    for link_formation_label, _allow_link_formation in LINK_FORMATION_MODES:
                        key = (
                            graph_kind,
                            cost_kind,
                            lifetime_config,
                            expiry,
                            lifetime_kind,
                            seed,
                            link_formation_label,
                        )
                        for strategy_label, strategy_name in STRATEGIES:
                            tasks.append((key, strategy_label, strategy_name))
    return tasks


def _run_task(task):
    key, strategy_label, strategy_name = task
    graph_kind, cost_kind, _lifetime_config, expiry, lifetime_kind, seed, link_formation_label = key
    allow_link_formation = link_formation_label == "with_link_formation"
    result = run_experiment(
        graph_kind=graph_kind,
        cost_kind=cost_kind,
        expiry=expiry,
        n=N_AGENTS,
        seed=seed,
        strategy=strategy_name,
        cost_multiplier=COST_MULTIPLIER,
        max_steps=MAX_STEPS,
        min_equilibrium_steps=0 if allow_link_formation else WITHOUT_LINK_FORMATION_MIN_STEPS,
        equilibrium_precision=EQUILIBRIUM_PRECISION,
        equilibrium_eps=EQUILIBRIUM_EPS,
        min_link_flow=MIN_LINK_FLOW,
        lifetime_kind=lifetime_kind,
        ev_heuristics=EV_HEURISTICS,
        link_estimation_mode="formula",
        max_candidate_edges_per_agent=MAX_CANDIDATE_EDGES_PER_AGENT,
        allow_link_formation=allow_link_formation,
        include_details=True,
        include_time_series=True,
        welfare_time_interval=WELFARE_TIME_INTERVAL,
    )
    return key, strategy_label, result


def _run_tasks_serial(tasks):
    for task in tasks:
        yield _run_task(task)


def _expiry_label(expiry: Optional[int]) -> str:
    return "0" if expiry is None else str(expiry)


def _summary_line(key: ExperimentKey, strategy_label: str, result: Dict[str, object]) -> str:
    graph_kind, cost_kind, lifetime_config, expiry, lifetime_kind, seed, link_formation_label = key
    return (
        f"{graph_kind},{cost_kind},{lifetime_config},{_expiry_label(expiry)},{lifetime_kind},"
        f"{seed},{link_formation_label},{strategy_label},"
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


def _write_detail_files(results_dir: Path, key: ExperimentKey, strategy_label: str, result: Dict[str, object]) -> None:
    graph_kind, cost_kind, lifetime_config, _expiry, _lifetime_kind, seed, link_formation_label = key
    file_prefix = f"{graph_kind}_{cost_kind}_{lifetime_config}_seed_{seed}_{link_formation_label}_{strategy_label}"

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
        _write_graph_png(results_dir, file_prefix, nodes, edges, seed)


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
    graph_kind, cost_kind, lifetime_config, expiry, lifetime_kind, seed, link_formation_label = key
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
        f"{seed},{link_formation_label},"
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


def _time_series_rows(key: ExperimentKey, strategy_label: str, result: Dict[str, object]) -> List[Dict[str, object]]:
    graph_kind, cost_kind, lifetime_config, expiry, lifetime_kind, seed, link_formation_label = key
    rows = []
    for item in result.get("time_series", []):
        row = {
            "graph_kind": graph_kind,
            "cost_kind": cost_kind,
            "lifetime_config": lifetime_config,
            "expiry": _expiry_label(expiry),
            "lifetime_kind": lifetime_kind,
            "seed": seed,
            "link_formation": link_formation_label,
            "strategy": strategy_label,
            "t": item["t"],
        }
        for metric in AGGREGATE_METRICS:
            row[metric] = item[metric]
        rows.append(row)
    return rows


def _time_series_line(row: Dict[str, object]) -> str:
    return (
        f"{row['graph_kind']},{row['cost_kind']},{row['lifetime_config']},"
        f"{row['expiry']},{row['lifetime_kind']},{row['seed']},{row['link_formation']},"
        f"{row['strategy']},{row['t']},"
        f"{row['average_utility']:.6f},"
        f"{row['welfare_eisenberg_gale']:.6f},"
        f"{row['welfare_net']:.6f},"
        f"{row['fairness_jain']:.6f},"
        f"{row['fairness_exchange_ratio_jain']:.6f},"
        f"{row['active_edges']:.6f}"
    )


def _write_aggregate_table(
    results_dir: Path,
    filename: str,
    rows: List[Dict[str, object]],
    group_columns: Tuple[str, ...],
) -> List[Dict[str, object]]:
    aggregate: Dict[Tuple[object, ...], Dict[str, float]] = {}
    for row in rows:
        key = tuple(row[column] for column in group_columns) + (row["t"],)
        bucket = aggregate.setdefault(key, {"count": 0.0, **{metric: 0.0 for metric in AGGREGATE_METRICS}})
        bucket["count"] += 1.0
        for metric in AGGREGATE_METRICS:
            bucket[metric] += float(row[metric])

    header_columns = list(group_columns) + ["t", "count"] + AGGREGATE_METRICS
    aggregate_rows: List[Dict[str, object]] = []
    with (results_dir / filename).open("w", encoding="utf-8") as output:
        output.write(",".join(header_columns) + "\n")
        for key in sorted(aggregate, key=str):
            bucket = aggregate[key]
            count = bucket["count"]
            row = {column: value for column, value in zip(list(group_columns) + ["t"], key)}
            row["count"] = int(count)
            for metric in AGGREGATE_METRICS:
                row[metric] = bucket[metric] / count
            aggregate_rows.append(row)

            values = [str(value) for value in key]
            values.append(str(int(count)))
            values.extend(f"{bucket[metric] / count:.6f}" for metric in AGGREGATE_METRICS)
            output.write(",".join(values) + "\n")
    return aggregate_rows


def _group_label(row: Dict[str, object], group_columns: Tuple[str, ...]) -> str:
    return " / ".join(str(row[column]) for column in group_columns)


def _write_welfare_plot(
    results_dir: Path,
    filename: str,
    rows: List[Dict[str, object]],
    group_columns: Tuple[str, ...],
    metric: str,
    title: str,
) -> None:
    if not rows or not GENERATE_WELFARE_PLOTS:
        return
    plot_rows = [row for row in rows if int(row["t"]) <= WELFARE_PLOT_MAX_STEP]
    if not plot_rows:
        return
    try:
        cache_dir = results_dir / "matplotlib_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skipping welfare plot {filename}: {exc}", flush=True)
        return

    series: Dict[str, List[Tuple[int, float]]] = {}
    for row in plot_rows:
        label = _group_label(row, group_columns)
        series.setdefault(label, []).append((int(row["t"]), float(row[metric])))

    plt.figure(figsize=(10, 6))
    for label in sorted(series):
        points = sorted(series[label])
        plt.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            linewidth=1.8,
            markersize=3,
            label=label,
        )
    plt.title(title)
    plt.xlabel("time step")
    plt.ylabel(metric)
    plt.xlim(left=0, right=WELFARE_PLOT_MAX_STEP)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(results_dir / filename, dpi=160)
    plt.close()


def _write_welfare_plots(
    results_dir: Path,
    aggregate_tables: Dict[str, Tuple[List[Dict[str, object]], Tuple[str, ...]]],
) -> None:
    plot_specs = [
        ("graph", "Welfare by graph structure"),
        ("cost", "Welfare by link cost"),
        ("lifetime", "Welfare by link lifetime"),
        ("strategy", "Welfare by strategy"),
    ]
    for table_key, title_prefix in plot_specs:
        rows, group_columns = aggregate_tables[table_key]
        _write_welfare_plot(
            results_dir,
            f"welfare_eisenberg_gale_by_{table_key}.png",
            rows,
            group_columns,
            "welfare_eisenberg_gale",
            f"{title_prefix}: Eisenberg-Gale",
        )
        _write_welfare_plot(
            results_dir,
            f"welfare_net_by_{table_key}.png",
            rows,
            group_columns,
            "welfare_net",
            f"{title_prefix}: net welfare",
        )
        _write_welfare_plot(
            results_dir,
            f"fairness_jain_by_{table_key}.png",
            rows,
            group_columns,
            "fairness_jain",
            f"{title_prefix}: Jain fairness",
        )
        _write_welfare_plot(
            results_dir,
            f"fairness_exchange_ratio_jain_by_{table_key}.png",
            rows,
            group_columns,
            "fairness_exchange_ratio_jain",
            f"{title_prefix}: exchange-ratio Jain fairness",
        )


def _write_welfare_tables(results_dir: Path, rows: List[Dict[str, object]]) -> None:
    with (results_dir / "welfare_over_time.csv").open("w", encoding="utf-8") as output:
        output.write(TIME_SERIES_HEADER + "\n")
        for row in rows:
            output.write(_time_series_line(row) + "\n")

    aggregate_tables = {
        "graph": (
            _write_aggregate_table(
                results_dir,
                "welfare_by_graph_over_time.csv",
                rows,
                ("graph_kind", "link_formation"),
            ),
            ("graph_kind", "link_formation"),
        ),
        "cost": (
            _write_aggregate_table(
                results_dir,
                "welfare_by_cost_over_time.csv",
                rows,
                ("cost_kind", "link_formation"),
            ),
            ("cost_kind", "link_formation"),
        ),
        "lifetime": (
            _write_aggregate_table(
                results_dir,
                "welfare_by_lifetime_over_time.csv",
                rows,
                ("lifetime_config", "lifetime_kind", "link_formation"),
            ),
            ("lifetime_config", "lifetime_kind", "link_formation"),
        ),
        "strategy": (
            _write_aggregate_table(
                results_dir,
                "welfare_by_strategy_over_time.csv",
                rows,
                ("strategy", "link_formation"),
            ),
            ("strategy", "link_formation"),
        ),
    }
    _write_aggregate_table(
        results_dir,
        "welfare_by_graph_cost_lifetime_strategy_over_time.csv",
        rows,
        ("graph_kind", "cost_kind", "lifetime_config", "lifetime_kind", "strategy", "link_formation"),
    )
    _write_welfare_plots(results_dir, aggregate_tables)


def _base_key(key: ExperimentKey) -> BaseExperimentKey:
    graph_kind, cost_kind, lifetime_config, expiry, lifetime_kind, seed, _link_formation_label = key
    return graph_kind, cost_kind, lifetime_config, expiry, lifetime_kind, seed


def _pct_delta(with_value: float, without_value: float) -> Optional[float]:
    if without_value == 0:
        return None
    return 100.0 * (with_value - without_value) / abs(without_value)


def _format_optional(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.6f}"


def _link_formation_comparison_line(
    base_key: BaseExperimentKey,
    strategy_label: str,
    without_result: Dict[str, object],
    with_result: Dict[str, object],
) -> str:
    graph_kind, cost_kind, lifetime_config, expiry, lifetime_kind, seed = base_key
    without_average_utility = float(without_result["average_utility"])
    with_average_utility = float(with_result["average_utility"])
    delta_average_utility = with_average_utility - without_average_utility
    pct_delta_average_utility = _pct_delta(with_average_utility, without_average_utility)

    without_welfare_eg = float(without_result["welfare_eisenberg_gale"])
    with_welfare_eg = float(with_result["welfare_eisenberg_gale"])
    without_welfare_net = float(without_result["welfare_net"])
    with_welfare_net = float(with_result["welfare_net"])
    without_fairness = float(without_result["fairness_jain"])
    with_fairness = float(with_result["fairness_jain"])
    without_exchange_fairness = float(without_result["fairness_exchange_ratio_jain"])
    with_exchange_fairness = float(with_result["fairness_exchange_ratio_jain"])

    return (
        f"{graph_kind},{cost_kind},{lifetime_config},{_expiry_label(expiry)},{lifetime_kind},"
        f"{seed},{strategy_label},"
        f"{without_average_utility:.6f},{with_average_utility:.6f},{delta_average_utility:.6f},"
        f"{_format_optional(pct_delta_average_utility)},"
        f"{without_welfare_eg:.6f},{with_welfare_eg:.6f},{with_welfare_eg - without_welfare_eg:.6f},"
        f"{without_welfare_net:.6f},{with_welfare_net:.6f},{with_welfare_net - without_welfare_net:.6f},"
        f"{without_fairness:.6f},{with_fairness:.6f},{with_fairness - without_fairness:.6f},"
        f"{without_exchange_fairness:.6f},{with_exchange_fairness:.6f},"
        f"{with_exchange_fairness - without_exchange_fairness:.6f}"
    )


def _write_link_formation_comparison(
    results_dir: Path,
    grouped_results: Dict[ExperimentKey, Dict[str, Dict[str, object]]],
) -> None:
    by_base: Dict[BaseExperimentKey, Dict[str, Dict[str, Dict[str, object]]]] = {}
    for key, strategy_results in grouped_results.items():
        link_formation_label = key[-1]
        by_base.setdefault(_base_key(key), {})[link_formation_label] = strategy_results

    with (results_dir / "link_formation_comparison.csv").open("w", encoding="utf-8") as output:
        output.write(LINK_FORMATION_COMPARISON_HEADER + "\n")
        for base_key in sorted(by_base, key=str):
            mode_results = by_base[base_key]
            without_results = mode_results.get("without_link_formation", {})
            with_results = mode_results.get("with_link_formation", {})
            for strategy_label, _strategy_name in STRATEGIES:
                if strategy_label not in without_results or strategy_label not in with_results:
                    continue
                output.write(
                    _link_formation_comparison_line(
                        base_key,
                        strategy_label,
                        without_results[strategy_label],
                        with_results[strategy_label],
                    )
                    + "\n"
                )


if __name__ == "__main__":
    results_dir = Path("example/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    tasks = _make_tasks()
    if TASK_LIMIT > 0:
        tasks = tasks[:TASK_LIMIT]
    grouped_results: Dict[ExperimentKey, Dict[str, Dict[str, object]]] = {}
    welfare_rows: List[Dict[str, object]] = []

    print(SUMMARY_HEADER)
    print(
        f"running {len(tasks)} experiments; mode=serial; "
        f"n={N_AGENTS}; max_steps={MAX_STEPS}; precision={EQUILIBRIUM_PRECISION}; "
        f"eps={EQUILIBRIUM_EPS}; "
        f"min_link_flow={MIN_LINK_FLOW}; "
        f"max_candidate_edges_per_agent={MAX_CANDIDATE_EDGES_PER_AGENT}; "
        f"cost_multiplier={COST_MULTIPLIER}; "
        f"seeds={','.join(str(seed) for seed in RANDOM_SEEDS)}; "
        f"link_formation_modes={','.join(label for label, _enabled in LINK_FORMATION_MODES)}; "
        f"estimator=formula; "
        f"welfare_interval={WELFARE_TIME_INTERVAL}; "
        f"welfare_plot_max_step={WELFARE_PLOT_MAX_STEP}; "
        f"without_link_formation_min_steps={WITHOUT_LINK_FORMATION_MIN_STEPS}"
    )

    completed = 0
    for key, strategy_label, result in _run_tasks_serial(tasks):
        grouped_results.setdefault(key, {})[strategy_label] = result
        welfare_rows.extend(_time_series_rows(key, strategy_label, result))
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

    _write_link_formation_comparison(results_dir, grouped_results)
    _write_welfare_tables(results_dir, welfare_rows)

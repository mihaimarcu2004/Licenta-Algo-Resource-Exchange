from __future__ import annotations

import argparse
import os
from pathlib import Path

from animate_graph import build_simulator, collect_snapshots, save_animation

GRAPH_KINDS = ["sparse", "small_world", "scale_free", "two_communities", "core_periphery"]
COST_KINDS = ["small", "medium", "big", "distance", "degree", "mixed"]
LIFETIME_CONFIGS = {
    "small_lifetime": (3, "small_lifetime"),
    "medium_lifetime": (7, "medium_lifetime"),
    "large_lifetime": (12, "large_lifetime"),
    "mixed_lifetime": (7, "mixed_lifetime"),
    "same_lifetime_3": (3, "same_lifetime"),
    "same_lifetime_7": (7, "same_lifetime"),
}
STRATEGIES = ["egalitarian", "proportional_1", "proportional_2"]


def parse_csv_choices(value: str, allowed: list[str]) -> list[str]:
    if value == "all":
        return allowed
    choices = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in choices if item not in allowed]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown values: {', '.join(unknown)}")
    return choices


def main() -> None:
    parser = argparse.ArgumentParser(description="Render graph-evolution videos for multiple scenarios.")
    parser.add_argument("--graph-kinds", default="all")
    parser.add_argument("--cost-kinds", default="all")
    parser.add_argument(
        "--lifetime-configs",
        default="small_lifetime,medium_lifetime,large_lifetime,mixed_lifetime,same_lifetime_3,same_lifetime_7",
    )
    parser.add_argument("--strategies", default="all")
    parser.add_argument("--nodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--min-link-flow", type=float, default=1e-9)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("example/results/videos"))
    args = parser.parse_args()

    graph_kinds = parse_csv_choices(args.graph_kinds, GRAPH_KINDS)
    cost_kinds = parse_csv_choices(args.cost_kinds, COST_KINDS)
    strategies = parse_csv_choices(args.strategies, STRATEGIES)
    lifetime_configs = parse_csv_choices(args.lifetime_configs, list(LIFETIME_CONFIGS))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    combo_idx = 0
    for graph_kind in graph_kinds:
        for cost_kind in cost_kinds:
            for lifetime_config in lifetime_configs:
                expiry, lifetime_kind = LIFETIME_CONFIGS[lifetime_config]
                seed = args.seed_base + combo_idx
                combo_idx += 1
                expiry_label = "0" if expiry is None else str(expiry)

                for strategy in strategies:
                    simulator, graph = build_simulator(
                        graph_kind=graph_kind,
                        cost_kind=cost_kind,
                        expiry=expiry,
                        n=args.nodes,
                        seed=seed,
                        strategy_label=strategy,
                        lifetime_kind=lifetime_kind,
                        min_link_flow=args.min_link_flow,
                    )
                    frames = collect_snapshots(simulator, args.steps)
                    output_path = args.output_dir / (
                        f"{graph_kind}_{cost_kind}_{lifetime_config}_{strategy}_evolution.mp4"
                    )
                    title = f"{graph_kind}, {cost_kind}, expiry={expiry_label}, {lifetime_kind}, {strategy}"
                    saved_path = save_animation(
                        frames=frames,
                        base_graph=graph,
                        simulator=simulator,
                        output_path=output_path,
                        fps=args.fps,
                        seed=seed,
                        title=title,
                        graph_kind=graph_kind,
                    )
                    print(saved_path)


if __name__ == "__main__":
    main()

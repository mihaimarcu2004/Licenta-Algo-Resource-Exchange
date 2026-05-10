from __future__ import annotations

from exchange_sim import DecentralizedExchangeSimulator, normalize_edge


if __name__ == "__main__":
    # Small example.
    agents = ["A", "B", "C", "D"]
    production = {"A": 10, "B": 12, "C": 8, "D": 9}

    # Initial graph: A-B and B-C are already active.
    initial_edges = [("A", "B"), ("B", "C")]

    # Candidate edges and their creation costs.
    edge_costs = {
        normalize_edge("A", "C"): 2.0,
        normalize_edge("A", "D"): 3.0,
        normalize_edge("B", "D"): 4.0,
        normalize_edge("C", "D"): 2.0,
    }

    sim = DecentralizedExchangeSimulator(
        agents=agents,
        production=production,
        initial_edges=initial_edges,
        strategy="constant",
        edge_costs=edge_costs,
        expiry=3,
        degradation=0.95,
        discount=0.9,
        max_new_links_per_slot=None,
        rng_seed=1,
    )

    for _ in range(8):
        sim.step()
        print(sim.summary())

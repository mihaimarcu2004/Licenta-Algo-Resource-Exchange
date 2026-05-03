from common import print_header, print_result, run_experiment

EXPIRIES = [None, 1, 3, 5, 10, 100]


if __name__ == "__main__":
    print("graph_kind: sparse")
    print("cost_kind: distance")
    print_header()
    for expiry in EXPIRIES:
        result = run_experiment(
            graph_kind="sparse",
            cost_kind="distance",
            expiry=expiry,
            n=20,
            seed=103,
            distance_factor=0.8,
        )
        print_result(expiry, result)

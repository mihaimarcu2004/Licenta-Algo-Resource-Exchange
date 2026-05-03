from common import print_header, print_result, run_experiment

EXPIRIES = [None, 1, 3, 5, 10, 100]


if __name__ == "__main__":
    print("graph_kind: small_world")
    print("cost_kind: constant")
    print_header()
    for expiry in EXPIRIES:
        result = run_experiment(
            graph_kind="small_world",
            cost_kind="constant",
            expiry=expiry,
            n=20,
            seed=302,
            constant_cost=1.0,
        )
        print_result(expiry, result)

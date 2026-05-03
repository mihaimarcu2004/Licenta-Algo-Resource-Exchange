from common import print_header, print_result, run_experiment

EXPIRIES = [None, 1, 3, 5, 10, 100]


if __name__ == "__main__":
    print("graph_kind: dense")
    print("cost_kind: zero")
    print_header()
    for expiry in EXPIRIES:
        result = run_experiment(
            graph_kind="dense",
            cost_kind="zero",
            expiry=expiry,
            n=20,
            seed=201,
        )
        print_result(expiry, result)

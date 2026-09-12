"""CLI: python -m generator seed [--n-customers N] [--seed N] [--output-dir DIR]"""

import argparse

from fraudguard_core.config import GeneratorSettings

from generator.generate import generate, write_output


def main() -> None:
    parser = argparse.ArgumentParser(prog="generator")
    sub = parser.add_subparsers(dest="command", required=True)

    seed_cmd = sub.add_parser("seed", help="Generate the synthetic FraudGuard dataset")
    seed_cmd.add_argument("--seed", type=int, default=None)
    seed_cmd.add_argument("--n-customers", type=int, default=None)
    seed_cmd.add_argument("--n-merchants", type=int, default=None)
    seed_cmd.add_argument("--target-fraud-rate", type=float, default=None)
    seed_cmd.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    if args.command == "seed":
        overrides = {}
        if args.seed is not None:
            overrides["seed"] = args.seed
        if args.n_customers is not None:
            overrides["n_customers"] = args.n_customers
        if args.n_merchants is not None:
            overrides["n_merchants"] = args.n_merchants
        if args.target_fraud_rate is not None:
            overrides["target_fraud_rate"] = args.target_fraud_rate
        if args.output_dir is not None:
            overrides["output_dir"] = args.output_dir

        config = GeneratorSettings(**overrides)
        print(f"Generating with config: {config.model_dump()}")
        result = generate(config)
        write_output(result, config.output_dir)
        print(f"Wrote output to {config.output_dir}")
        print(f"Row counts: {result.meta['row_counts']}")
        print(
            f"Realised fraud rate: {result.meta['realised_fraud_rate']:.4%} "
            f"(target {config.target_fraud_rate:.4%})"
        )


if __name__ == "__main__":
    main()

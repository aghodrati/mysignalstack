"""Entry point for running finance hypotheses.

Usage:
    uv run main.py momentum_top10
"""

import argparse
import importlib

HYPOTHESES = {
    "momentum_top10": "finance.hypotheses.momentum_top10",
}


def main():
    parser = argparse.ArgumentParser(description="Run a finance hypothesis backtest.")
    parser.add_argument("hypothesis", choices=sorted(HYPOTHESES), help="Which hypothesis to run")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    module = importlib.import_module(HYPOTHESES[args.hypothesis])
    module.main(start=args.start, end=args.end)


if __name__ == "__main__":
    main()

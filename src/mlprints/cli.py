"""Command-line entry point for mlprints."""

import argparse
from importlib import import_module
import sys


COMMANDS = {
    "generate": "mlprints.scripts.generate_fingerprints",
    "train": "mlprints.scripts.train_fingerprints",
    "verify": "mlprints.scripts.verify_fingerprints",
}

MEASUREMENTS = {
    "utility": "mlprints.scripts.measure_utility",
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="mlprints")
    parser.add_argument("command", choices=[*COMMANDS, "measure"])
    args = parser.parse_args(argv[:1])

    if args.command == "measure":
        parser = argparse.ArgumentParser(prog="mlprints measure")
        parser.add_argument("measurement", choices=MEASUREMENTS)
        args = parser.parse_args(argv[1:2])
        return import_module(MEASUREMENTS[args.measurement]).main(argv[2:])

    return import_module(COMMANDS[args.command]).main(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())

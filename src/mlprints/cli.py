"""Command-line entry point for mlprints."""

import argparse
from importlib import import_module


COMMANDS = {
    "generate": "mlprints.scripts.generate_fingerprints",
    "train": "mlprints.scripts.train_fingerprints",
    "verify": "mlprints.scripts.verify_fingerprints",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlprints")
    parser.add_argument("command", choices=COMMANDS)
    args, remainder = parser.parse_known_args(argv)
    return import_module(COMMANDS[args.command]).main(remainder)


if __name__ == "__main__":
    raise SystemExit(main())

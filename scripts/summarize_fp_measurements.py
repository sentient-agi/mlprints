"""
   scripts/summarize_fp_measurements.py

   Summarize the measurements.
"""
import os
import argparse
from oml.measure.strength import summarize_strength_measurements


def add_args(parser: argparse.ArgumentParser) -> None:
    """
        Add the command-line arguments to the parser.
    """
    parser.add_argument(
        "fp_dir",
        type=os.path.abspath,
        help="Path to the fingerprint directory",
    )
    parser.add_argument(
        "--strength",
        action="store_true",
        help="Summarize strength subdirectory",
    )
    return


if __name__ == "__main__":

    # get args
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    args = parser.parse_args()

    msmt_path = os.path.join(args.fp_dir, "measurements")

    # strength measurements
    if args.strength:
        print("Summarizing strength measurements...")

        df = summarize_strength_measurements(msmt_path, save_csv=True)
        summary_path = os.path.join(msmt_path, "strength", "summary.csv")

        print(f"Strength summary saved at {summary_path}!")
        print(df)

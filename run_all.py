"""Single-command entry point for the current Part 1 project."""
import argparse
from part1 import main


def run():
    parser = argparse.ArgumentParser(description="Run Recommender Systems Part 1")
    parser.add_argument("--data-dir", required=True, help="Folder containing MovieLens .dat files")
    parser.add_argument("--output-dir", default="part1_outputs", help="Generated output folder")
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)


if __name__ == "__main__":
    run()

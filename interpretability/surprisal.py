"""Score 30-token trunk windows with the fixed A1 surprisal detector.

In a trunk cache, pi holds the probability of each observed token after
temperature 0.6, top-k 20 and top-p 0.95 sampling filters.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def score_windows(path):
    if path.suffix == ".parquet":
        columns = ["task_id", "trunk_seed", "position", "zero_pi", "surprise_sum"]
        for row in pq.read_table(path, columns=columns).to_pylist():
            score = 1000.0 if row["zero_pi"] else row["surprise_sum"]
            yield row["task_id"], row["trunk_seed"], row["position"], score
        return

    task, seed = map(int, path.parent.name.split("_"))
    with np.load(path) as cache:
        pi = cache["pi"].astype(np.float64)
    # Both ends of a window must be sampled prefix positions on this trunk.
    for start in range(0, len(pi) - 30, 30):
        window = pi[start : start + 30]
        score = 1000.0 if np.any(window == 0) else -np.log(window).sum()
        yield task, seed, start, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", type=Path, nargs="+",
        help="window features in Parquet, or <task>_<seed>/trunk.npz with sampled token probabilities (pi)",
    )
    parser.add_argument("--output", required=True, type=Path, help="output CSV")
    args = parser.parse_args()
    rows = [row for path in args.inputs for row in score_windows(path)]
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task_id", "trunk_seed", "position", "score"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()

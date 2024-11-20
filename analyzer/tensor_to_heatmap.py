import argparse
from pathlib import Path
import csv

import numpy as np
from matplotlib import pyplot as plt


def main(file, output):
    with file.open("r", newline="") as f:
        reader = csv.reader(f)
        contents = [[float(c) for c in row] for row in reader]
    contents = np.array(contents)

    vmin = contents.min()
    vmax = contents.max()
    fig, ax = plt.subplots(figsize=(12, 10))
    pc = ax.pcolormesh(contents, vmin=0, vmax=100)
    # pc = ax.pcolormesh(contents, vmin=vmin, vmax=vmax)
    fig.colorbar(pc, ax=ax)
    plt.savefig(output, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", type=Path, default="analyzer/raw/emb_4_bias_binary_kl.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="analyzer/result/emb_4_bias_binary_kl.png",
    )
    args = parser.parse_args()
    main(args.file, args.output)

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def main(dists, output_path):
    n_dists = len(dists)
    labels = [str(d.parent).split("/")[-1] for d in dists]
    dists = [torch.load(d, map_location="cpu") for d in dists]
    n_p = dists[0].shape[0]
    dists = [d.reshape(n_p, -1) for d in dists]
    n_c = dists[0].shape[-1]

    assert all([d.shape[-1] == n_c for d in dists])

    dists = torch.cat(dists).detach()
    dists = dists.exp()
    dists = dists.numpy()

    tsne = TSNE(
        n_components=2, init="pca", learning_rate="auto", max_iter=2000
    )
    reduced_data = tsne.fit_transform(dists)
    reduced_data = np.split(reduced_data, n_dists)

    fig, ax = plt.subplots(figsize=(10, 10))
    for l, d in zip(labels, reduced_data):
        ax.scatter(d[:, 0], d[:, 1], label=l)
    fig.legend()

    # if label:
    #     for i, txt in enumerate(label):
    #         ax.annotate(txt, (reduced_data[i, 0], reduced_data[i, 1]))

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dists", nargs="+", type=Path)
    parser.add_argument("--output_path", type=Path)
    args = parser.parse_args()

    main(args.dists, args.output_path)

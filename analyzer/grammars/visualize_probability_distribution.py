import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from matplotlib import pyplot as plt

from utils import read_tsv_tensor


def load_data(path, type="torch"):
    if type == "torch":
        return [torch.load(d, map_location="cpu") for d in path]
    elif type == "tsv":
        return [read_tsv_tensor(d) for d in path]


def main(
    dists,
    output_path,
    logit=False,
    data_labels=None,
    data_type="torch",
    method="pca",
    n_components=2,
    pivot=True,
):
    n_dists = len(dists)
    if data_labels is None:
        labels = [str(d.parent).split("/")[-1] for d in dists]
    else:
        labels = data_labels
    dists = load_data(dists, type=data_type)
    n_p = dists[0].shape[0]
    dists = [d.reshape(n_p, -1) for d in dists]
    n_c = dists[0].shape[-1]

    assert all([d.shape[-1] == n_c for d in dists])

    dists = torch.cat(dists).detach()
    if logit:
        dists = dists.exp()
    if pivot:
        pv = torch.eye(n_c)
        dists = torch.cat([pv, dists])
    dists = dists.numpy()

    if method == "pca":
        reducer = PCA(n_components=n_components)
    elif method == "t-sne":
        reducer = TSNE(
            n_components=n_components,
            init="pca",
            learning_rate="auto",
            max_iter=2000,
        )
    elif method == "umap":
        reducer = umap.UMAP(n_components=n_components, min_dist=0.1)

    reduced_data = reducer.fit_transform(dists)
    if pivot:
        pivot_data = reduced_data[:n_c]
        reduced_data = reduced_data[n_c:]
    reduced_data = np.split(reduced_data, n_dists)

    fig, ax = plt.subplots(figsize=(10, 10))
    if pivot:
        ax.scatter(pivot_data[:, 0], pivot_data[:, 1], label="pivot")
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
    parser.add_argument("--logit", action="store_true")
    parser.add_argument("--data_type", type=str, default="tsv")
    parser.add_argument("--data_labels", nargs="+", default=None)
    parser.add_argument("--method", type=str, default="pca")
    parser.add_argument("--pivot", action="store_true")
    parser.add_argument("--output_path", type=Path)
    args = parser.parse_args()

    main(
        dists=args.dists,
        output_path=args.output_path,
        logit=args.logit,
        data_labels=args.data_labels,
        method=args.method,
        pivot=args.pivot,
        data_type=args.data_type,
    )

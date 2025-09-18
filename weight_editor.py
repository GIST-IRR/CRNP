import argparse
from pathlib import Path
from collections import OrderedDict

import torch

keypairs = {
    "term_emb": "terms.term_emb",
    "root_emb": "root.root_emb",
    "nonterm_emb": "nonterms.nonterm_emb",
    "terms.term_mlp.0.weight": "terms.rule_mlp.0.weight",
    "terms.term_mlp.3.weight": "terms.child_emb",
    "root.root_mlp.0.weight": "root.rule_mlp.0.weight",
    "root.root_mlp.3.weight": "root.child_emb",
    "rank_proj": "nonterms.rank_proj.weight",
}


def change_keys(t: dict, keypairs: dict = keypairs):
    n_t = OrderedDict()

    for k, v in t.items():
        if k in ["nonterm_emb", "term_emb"]:
            continue
        elif k == "terms.term_emb":
            n_t["term_emb"] = v
        elif k == "root.root_emb":
            n_t["root_emb"] = v
        elif "term_mlp" in k:
            n_k = k.replace("term_mlp", "rule_mlp")
            n_t[n_k] = v
            if n_k == "terms.rule_mlp.3.weight":
                n_t["terms.child_emb"] = v
        elif "root_mlp" in k:
            n_k = k.replace("root_mlp", "rule_mlp")
            n_t[n_k] = v
            if n_k == "root.rule_mlp.3.weight":
                n_t["root.child_emb"] = v
        elif "rank_proj" in k:
            n_t["nonterms.rank_proj.weight"] = v.T
        else:
            n_t[k] = v
    # for k, v in t.items():
    #     if k in keypairs:
    #         n_t[keypairs[k]] = v
    #     else:
    #         n_t[k] = v

    return n_t


def main(old: Path, new: Path):
    # Load old weights
    with old.open("rb") as f:
        old_w = torch.load(f)

    # Change keys of the weights
    new_w = change_keys(old_w["model"])
    old_w["model"] = new_w

    # Save new weights
    with new.open("wb") as f:
        torch.save(old_w, new)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old",
        "-o",
        type=Path,
        default="log/n_english_std_base/NeuralPCFG2023-12-27-12_53_02/best.pt",
    )
    parser.add_argument(
        "--new",
        "-n",
        type=Path,
        default="log/n_english_std_base/NeuralPCFG2023-12-27-12_53_02/best_new.pt",
    )
    # parser.add_argument("--keypairs", "-k", type=Path, default="keypairs.yaml")
    args = parser.parse_args()
    main(args.old, args.new)

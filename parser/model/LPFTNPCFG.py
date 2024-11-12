from collections import defaultdict, Counter

import torch

from parser.model.TNPCFG import TNPCFG
from parser.model.PFTNPCFG import Parse_Focusing


class Labeled_Parse_Focusing(Parse_Focusing):
    def prepare_trees(self, model_paths):
        self.parse_trees = []
        # self.parse_labels = []
        nts = Counter()
        ts = Counter()
        for model in model_paths:
            parses = defaultdict(list)
            with open(model, "rb") as f:
                pts = torch.load(f)
            for t in pts:
                n_word = t["word"]
                label = t["label"]
                # Get trees
                if "pred_tree" in t.keys():
                    n_tree = [s for s in t["pred_tree"] if s[1] - s[0] > 1]
                elif "tree" in t.keys():
                    n_tree = [s[:2] for s in t["tree"] if s[1] - s[0] > 1]
                n_tree = [(t[0], t[1], l) for t, l in zip(n_tree, label)]
                parses[str(n_word)] = n_tree
                # self.label_set.update(label)
                for l in label:
                    nts[l] += 1
                # Add pos spans
                if "pos" in t.keys():
                    pos = t["pos"]
                    n_tree += [(i, i + 1, p) for i, p in enumerate(pos)]
                    for p in pos:
                        ts[p] += 1

            self.parse_trees.append(parses)

        # Indexing labels
        self.idx2nt = [l for l, c in nts.most_common()]
        self.nt2idx = {l[0]: i for i, l in enumerate(nts.most_common())}
        self.idx2t = [p for p, c in ts.most_common()]
        self.t2idx = {p[0]: i for i, p in enumerate(ts.most_common())}

        for pts in self.parse_trees:
            for k, t in pts.items():
                n_tree = []
                for s in t:
                    if s[1] - s[0] > 1:
                        n_tree += [(s[0], s[1], self.nt2idx[s[2]])]
                    else:
                        n_tree += [(s[0], s[1], len(nts) + self.t2idx[s[2]])]
                pts[k] = n_tree

    def get_pretrained_tree_mask(self, words):
        b, seq_len = words.shape[:2]

        trees = None
        tree_mask = None

        if self.pretrained_models:
            trees = []
            # Tree selected from pre-trained parser
            for parse in self.parse_trees:
                tree = words.new_tensor(
                    [parse.get(str(words[i].tolist())) for i in range(b)]
                )
                trees.append(tree)

            # Calculate span frequency in predicted trees
            trees = torch.stack(trees, dim=1)
            # # Concatenated mask
            trees = trees.reshape(b, trees.shape[1] * trees.shape[2], -1)

        if trees is not None:
            tree_mask = trees.new_zeros(
                b, seq_len + 1, seq_len + 1, self.NT
            ).float()
            for i, t in enumerate(trees):
                for s in t:
                    tree_mask[i, *s] += 1

            # Softmax mask
            if self.mask_mode == "soft":
                idx0, idx1 = torch.triu_indices(
                    seq_len + 1, seq_len + 1, offset=2
                ).unbind()
                masked_data = tree_mask[:, idx0, idx1]
                masked_data = masked_data.reshape(b, -1)
                # Normalization
                masked_data = masked_data.softmax(-1)
                # masked_data = masked_data / masked_data.sum(-1, keepdim=True)
                # End
                masked_data = masked_data.reshape(b, -1, self.NT)
                tree_mask[:, idx0, idx1] = masked_data
            elif self.mask_mode == "hard":
                tree_mask = tree_mask > 0
            elif self.mask_mode == "hybrid":
                tree_mask = torch.where(tree_mask > 0, tree_mask, 0)
            elif self.mask_mode == "no_process":
                pass

        return tree_mask


class LPFTNPCFG(Labeled_Parse_Focusing, TNPCFG):
    """Labeled Parse-Focused TN-PCFG"""

    def __init__(self, args):
        super(LPFTNPCFG, self).__init__(args)
        self._setup_parse_focusing(args)

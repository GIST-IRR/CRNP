from collections import defaultdict, Counter

import torch

from ..parse_foucsing.ParseFocusing import ParseFocusing


class LabeledParseFocusing(ParseFocusing):
    def _setup_parse_focusing(self, args):
        super()._setup_parse_focusing(args)
        args.NT = len(self.idx2nt)
        args.T = len(self.idx2t)

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
        return super().get_pretrained_tree_mask(words, label=True)

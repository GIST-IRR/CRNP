import torch

from ..model.FTNPCFG import FTNPCFG

from collections import defaultdict


class PFTNPCFG(FTNPCFG):
    """Parse focused FTN-PCFG"""

    def __init__(self, args):
        super(PFTNPCFG, self).__init__(args)

        if not getattr(args, "eval_mode", False):
            self.pretrained_models = getattr(args, "pretrained_models", None)
            if self.pretrained_models is not None:
                self.parse_trees = self.prepare_trees(self.pretrained_models)
            else:
                self.parse_trees = []

    def _set_arguments(self, args):
        super()._set_arguments(args)
        self.mask_mode = getattr(args, "mask_mode", "soft")

    def prepare_trees(self, model_paths):
        parse_trees = []
        for model in model_paths:
            parses = defaultdict(list)
            with open(model, "rb") as f:
                pts = torch.load(f)
            if isinstance(pts, dict):
                pts = pts["trees"]
            for t in pts:
                n_word = t["word"]
                if "pred_tree" in t.keys():
                    n_tree = [s[:2] for s in t["pred_tree"] if s[1] - s[0] > 1]
                elif "tree" in t.keys():
                    n_tree = [s[:2] for s in t["tree"] if s[1] - s[0] > 1]
                parses[str(n_word)] = n_tree
            parse_trees.append(parses)
        return parse_trees

    def loss(self, input, partition=False, soft=False, label=False, **kwargs):
        words = input["word"]
        b, seq_len = words.shape[:2]

        trees = None

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
            trees = trees.reshape(b, -1, 2)

        if trees is not None:
            tree_mask = trees.new_zeros(b, seq_len + 1, seq_len + 1).float()
            for i, t in enumerate(trees):
                for s in t:
                    tree_mask[i, s[0], s[1]] += 1

            # Softmax mask
            if self.mask_mode == "soft":
                idx0, idx1 = torch.triu_indices(
                    seq_len + 1, seq_len + 1, offset=2
                ).unbind()
                masked_data = tree_mask[:, idx0, idx1]
                masked_data = masked_data.softmax(-1)
                tree_mask[:, idx0, idx1] = masked_data
            elif self.mask_mode == "hard":
                tree_mask = tree_mask > 0

        self.rules = self.forward()
        self.rules = self.batchify(self.rules, words)

        result = self.pcfg(
            self.rules,
            self.rules["unary"],
            lens=input["seq_len"],
            tree=tree_mask,
        )
        return -result["partition"].mean()

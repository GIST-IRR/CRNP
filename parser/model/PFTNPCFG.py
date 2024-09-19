import torch

from ..model.NeuralPCFG import NeuralPCFG
from ..model.FTNPCFG import FTNPCFG

from collections import defaultdict


class Parse_Focusing(NeuralPCFG):
    def _setup_parse_focusing(self, args):
        self.mask_mode = getattr(args, "mask_mode", "soft")
        if not getattr(args, "eval_mode", False):
            self.pretrained_models = getattr(args, "pretrained_models", None)
            if self.pretrained_models is not None:
                self.parse_trees = self.prepare_trees(self.pretrained_models)
            else:
                self.parse_trees = []

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

        return tree_mask

    def loss(self, input, partition=False, soft=False, label=False, **kwargs):
        words = input["word"]

        tree_mask = self.get_pretrained_tree_mask(words)

        self.rules = self.forward()
        rules = self.batchify(self.rules, words)

        result = self.pcfg(
            rules,
            rules["unary"],
            lens=input["seq_len"],
            tree=tree_mask,
        )
        return -result["partition"].mean()


class PFTNPCFG(Parse_Focusing, FTNPCFG):
    """Parse focused FTN-PCFG"""

    def __init__(self, args):
        super(PFTNPCFG, self).__init__(args)
        self._setup_parse_focusing(args)

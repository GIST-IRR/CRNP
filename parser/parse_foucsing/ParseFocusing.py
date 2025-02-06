from collections import defaultdict

import torch


class ParseFocusing:
    def _setup_parse_focusing(self, args):
        self.args = args
        self.mask_mode = getattr(args, "mask_mode", "soft")
        if not getattr(args, "eval_mode", False):
            self.pretrained_models = getattr(args, "pretrained_models", None)
            if self.pretrained_models is not None:
                self.prepare_trees(self.pretrained_models)
            else:
                self.parse_trees = None

    def prepare_trees(self, model_paths):
        self.parse_trees = []
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
            self.parse_trees.append(parses)

    def get_pretrained_tree_mask(self, words, label=False):
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
            if label:
                tree_mask = trees.new_zeros(
                    b, seq_len + 1, seq_len + 1, self.NT_T
                ).float()
            else:
                tree_mask = trees.new_zeros(
                    b, seq_len + 1, seq_len + 1
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
                if label:
                    masked_data = masked_data.reshape(b, -1)
                masked_data = masked_data.softmax(-1)
                if label:
                    masked_data = masked_data.reshape(b, -1, self.NT_T)
                tree_mask[:, idx0, idx1] = masked_data
            elif self.mask_mode == "hard":
                # tree_mask = tree_mask > 0
                tree_mask = torch.where(tree_mask > 0, 1.0, 0.0001)
            elif self.mask_mode == "hybrid":
                tree_mask = torch.where(tree_mask > 0, tree_mask, 0.0001)

        return tree_mask

    def forward(
        self,
        input,
        partition=False,
        soft=False,
        label=False,
        reduction=None,
        **kwargs,
    ):
        words = input["word"]

        if self.parse_trees is not None:
            tree_mask = self.get_pretrained_tree_mask(words)
        else:
            tree_mask = None

        self.rules = self.get_grammar(input)
        rules = self.batchify(self.rules, words)
        self.rules["word"] = input["word"]

        result = self.pcfg(
            rules,
            lens=input["seq_len"],
            tree=tree_mask,
        )

        if reduction == "mean":
            res = -result["partition"].mean()
        elif reduction == "sum":
            res = -result["partition"].sum()
        elif reduction == None:
            res = -result["partition"]
        return res

import torch

from parser.model.TNPCFG import TNPCFG
from parser.model.LPFTNPCFG import Labeled_Parse_Focusing


class Splitted_Labeled_Parse_Focusing(Labeled_Parse_Focusing):
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
                b, seq_len + 1, seq_len + 1, self.NT_T
            ).float()
            for i, t in enumerate(trees):
                for s in t:
                    split_range = list(
                        range(
                            s[-1] * self.symbol_split,
                            (s[-1] + 1) * self.symbol_split,
                        )
                    )
                    tree_mask[i, *s[:2], split_range] += 1

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


class SLPFTNPCFG(Splitted_Labeled_Parse_Focusing, TNPCFG):
    """Splitted Labeled Parse-Focused TN-PCFG"""

    def _set_arguments(self, args):
        self.symbol_split = getattr(args, "symbol_split", 2)
        args.NT = self.symbol_split * args.NT
        args.T = self.symbol_split * args.T
        super(SLPFTNPCFG, self)._set_arguments(args)

    def __init__(self, args):
        self._setup_parse_focusing(args)
        args.NT = len(self.idx2nt)
        args.T = len(self.idx2t)
        super(SLPFTNPCFG, self).__init__(args)

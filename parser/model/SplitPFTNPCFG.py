from itertools import chain

import torch

from .PFTNPCFG import PFTNPCFG
from ..pcfgs.pcfg import PCFG


class SplitPFTNPCFG(PFTNPCFG):
    def __init__(self, args):
        args.V += 3
        super().__init__(args)

    def split_unknown(self, words, pos):
        n_words = words.clone()
        unk_idx = (words.flatten() == 1).nonzero().flatten().tolist()
        if len(unk_idx) != 0:
            pos_tags = list(chain.from_iterable(pos))
            new_tok = []
            for i in unk_idx:
                if "NN" in pos_tags[i]:
                    new_tok.append(10003)
                elif "JJ" in pos_tags[i]:
                    new_tok.append(10004)
                elif "VB" in pos_tags[i]:
                    new_tok.append(10005)
                else:
                    new_tok.append(1)
            unk_uidx = torch.unravel_index(
                torch.tensor(unk_idx).int(), words.shape
            )
            n_words[unk_uidx] = torch.tensor(new_tok, device=words.device)
        return n_words

    def loss(self, input, pos, **kwargs):
        words = input["word"]
        n_words = self.split_unknown(words, pos)

        self.rules = self.forward()
        self.rules = self.batchify(self.rules, n_words)

        tree_mask = self.get_pretrained_tree_mask(words)

        result = self.pcfg(
            self.rules,
            self.rules["unary"],
            lens=input["seq_len"],
            tree=tree_mask,
        )
        return -result["partition"]

    def evaluate(
        self,
        input,
        pos,
        decode_type="mbr",
        depth=0,
        label=True,
        rule_update=False,
        **kwargs,
    ):
        self.check_rule_update(rule_update)

        if decode_type == "viterbi":
            if not hasattr(self, "viterbi_pcfg"):
                self.viterbi_pcfg = PCFG()
                self.rules = self.compose(self.rules)

        n_words = self.split_unknown(input["word"], pos)
        rules = self.batchify(self.rules, n_words)

        result = self.decode(rules, input["seq_len"], decode_type, label)
        result.update({"word": n_words})
        return result

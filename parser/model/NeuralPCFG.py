import torch
import torch.nn as nn
from nltk import Tree
from nltk.grammar import Nonterminal

from parser.pfs.partition_function import PartitionFunction
from ..pcfgs.pcfg import PCFG
from .PCFG_module import (
    PCFG_module,
    Nonterm_parameterizer,
    UnaryRule_parameterizer,
)


class NeuralPCFG(PCFG_module):
    def __init__(self, args):
        super(NeuralPCFG, self).__init__()
        self.pcfg = PCFG()
        self.part = PartitionFunction()

        self._set_configs(args)
        self._init_grammar()
        self._initialize()

    def _set_configs(self, args):
        super()._set_configs(args)

        self.s_dim = getattr(args, "s_dim", 256)
        self.dropout = getattr(args, "dropout", 0.0)

        self.temperature = getattr(args, "temperature", 1.0)
        self.smooth = getattr(args, "smooth", 0.0)

        self.embedding_sharing = getattr(args, "embedding_sharing", False)

        # Partition function
        self.mode = getattr(args, "mode", "length_unary")

    def _embedding_sharing(self):
        if self.embedding_sharing:
            print("embedding sharing")
            self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.s_dim))
            self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        else:
            self.nonterm_emb = None
            self.term_emb = None

    def _init_grammar(self):
        self._embedding_sharing()
        # terms
        self.terms = UnaryRule_parameterizer(
            self.s_dim,
            self.T,
            self.V,
            parent_emb=self.term_emb,
            **self.cfgs.unary
        )
        self.nonterms = Nonterm_parameterizer(
            self.s_dim,
            self.NT,
            self.T,
            nonterm_emb=self.nonterm_emb,
            term_emb=self.term_emb,
            **self.cfgs.binary
        )
        # root
        self.root = UnaryRule_parameterizer(
            self.s_dim,
            1,
            self.NT,
            child_emb=self.nonterm_emb,
            **self.cfgs.root
        )

    def entropy(self, key, batch=False, probs=False, reduce="none"):
        assert key == "root" or key == "rule" or key == "unary"
        return self._entropy(
            self.rules[key], batch=batch, probs=probs, reduce=reduce
        )

    def get_entropy(self, batch=False, probs=False, reduce="mean"):
        r_ent, n_ent, t_ent = (
            self.entropy(k, batch=batch, probs=probs, reduce=reduce)
            for k in ["root", "rule", "unary"]
        )

        if reduce == "none":
            ent_prob = {"root": r_ent, "rule": n_ent, "unary": t_ent}
        elif reduce == "mean":
            ent_prob = torch.cat([r_ent, n_ent, t_ent]).mean()
        return ent_prob

    def forward(self, input=None):
        # Root
        root = self.root()
        # Rule
        rule = self.nonterms(reshape=True)
        # Unary
        unary = self.terms()

        # for gradient conflict by using gradients of rules
        if self.training:
            root.retain_grad()
            rule.retain_grad()
            unary.retain_grad()

        self.clear_metrics()  # clear metrics becuase we have new rules

        return {
            "unary": unary,
            "root": root,
            "rule": rule,
            # 'kl': torch.tensor(0, device=self.device)
        }

    def partition_function(self, max_length=200):
        return self.part(
            self.rules, lens=max_length, mode="depth", until_converge=True
        )

    def batchify(self, rules, words):
        b = words.shape[0]

        return {
            "root": rules["root"].expand(b, rules["root"].shape[-1]),
            "rule": rules["rule"].expand(b, *rules["rule"].shape),
            "unary": rules["unary"][
                torch.arange(self.T)[None, None], words[:, :, None]
            ],
        }

    def loss(self, input, partition=False, soft=False, **kwargs):
        words = input["word"]

        # Calculate rule distributions
        self.rules = self.forward(input)
        self.rules = self.batchify(self.rules, words)
        self.rules["word"] = input["word"]

        if partition:
            result = self.pcfg(
                self.rules, self.rules["unary"], lens=input["seq_len"], topk=1
            )
            self.pf = self.part(
                self.rules, lens=input["seq_len"], mode=self.mode
            )
            if soft:
                return -result["partition"].mean(), self.pf.mean()
            result["partition"] = result["partition"] - self.pf
        else:
            result = self.pcfg(
                self.rules,
                self.rules["unary"],
                lens=input["seq_len"],
                dropout=self.dropout,
            )

        return -result["partition"]

    def check_rule_update(self, rule_update):
        if rule_update:
            need_update = True
        else:
            if hasattr(self, "rules"):
                need_update = False
            else:
                need_update = True

        if need_update:
            self.rules = self.forward()

    def decode(self, rules, lens, decode_type, label=False):
        if decode_type == "viterbi":
            result = self.pcfg(
                rules,
                rules["unary"],
                lens=lens,
                viterbi=True,
                mbr=False,
                label=label,
            )
        elif decode_type == "mbr":
            result = self.pcfg(
                rules,
                rules["unary"],
                lens=lens,
                viterbi=False,
                mbr=True,
                label=label,
            )
        else:
            raise NotImplementedError
        return result

    def evaluate(
        self,
        input,
        decode_type="mbr",
        label=False,
        rule_update=False,
        **kwargs
    ):
        self.check_rule_update(rule_update)
        rules = self.batchify(self.rules, input["word"])
        result = self.decode(rules, input["seq_len"], decode_type, label)

        return result

    def calculate_tree_probability(self, tree: Tree, rule_update=False):
        self.check_rule_update(rule_update)
        rules = self.batchify(self.rules, input)

        def label_to_idx(label: Nonterminal):
            if isinstance(label, Nonterminal):
                return label.symbol().replace("'", "").split("-")[1]
            else:
                return label

        prod = tree.productions()
        prob = 1
        for p in prod:
            parent = label_to_idx(p.lhs())

            if len(p.rhs()) == 1:
                left_child = label_to_idx(p.rhs()[0])
            elif len(p.rhs()) == 2:
                left_child, right_child = list(map(label_to_idx, p.rhs()))
            else:
                raise "Only binary or unary rules can be calculated."

import numpy as np
import torch
import torch.nn as nn
from nltk import Tree
from nltk.grammar import Nonterminal

from parser.pfs.partition_function import PartitionFunction
from ..pcfgs.pcfg import PCFG
from .PCFG_module import (
    PCFG_module,
    Term_parameterizer,
    Nonterm_parameterizer,
    Root_parameterizer,
    UnaryRule_parameterizer,
)


class NeuralPCFG(PCFG_module):
    def __init__(self, args):
        super(NeuralPCFG, self).__init__()
        self.pcfg = PCFG()
        self.part = PartitionFunction()

        self._set_arguments(args)
        self._init_grammar()
        self._initialize()

    def _set_arguments(self, args):
        self.args = args

        # number of symbols
        self.NT = getattr(args, "NT", 30)
        self.T = getattr(args, "T", 60)
        self.NT_T = self.NT + self.T
        self.V = getattr(args, "V", 10003)

        self.s_dim = getattr(args, "s_dim", 256)
        self.dropout = getattr(args, "dropout", 0.0)

        self.temperature = getattr(args, "temperature", 1.0)
        self.smooth = getattr(args, "smooth", 0.0)
        self.activation = getattr(args, "activation", "relu")
        self.norm = getattr(args, "norm", None)

        self.embedding_sharing = getattr(args, "embedding_sharing", False)
        self.mlp_mode = getattr(args, "mlp_mode", "standard")
        self.cos_temp = getattr(args, "cos_temp", 1)

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
            activation=self.activation,
            norm=self.norm,
            parent_emb=self.term_emb,
            mlp_mode=self.mlp_mode,
            temp=self.cos_temp,
        )
        self.nonterms = Nonterm_parameterizer(
            self.s_dim,
            self.NT,
            self.T,
            nonterm_emb=self.nonterm_emb,
            term_emb=self.term_emb,
            mlp_mode="standard",
            temp=self.cos_temp,
        )
        # root
        self.root = UnaryRule_parameterizer(
            self.s_dim,
            1,
            self.NT,
            activation=self.activation,
            norm=self.norm,
            child_emb=self.nonterm_emb,
            mlp_mode=self.mlp_mode,
            temp=self.cos_temp,
        )

    def update_dropout(self, rate):
        self.apply_dropout = self.init_dropout * rate

    def entropy(self, key, batch=False, probs=False, reduce="none"):
        assert key == "root" or key == "rule" or key == "unary"
        return self._entropy(
            self.rules[key], batch=batch, probs=probs, reduce=reduce
        )

    def get_entropy(self, batch=False, probs=False, reduce="mean"):
        r_ent = self.entropy("root", batch=batch, probs=probs, reduce=reduce)
        n_ent = self.entropy("rule", batch=batch, probs=probs, reduce=reduce)
        t_ent = self.entropy("unary", batch=batch, probs=probs, reduce=reduce)

        # ent_prob = torch.cat([r_ent, n_ent, t_ent])
        # ent_prob = ent_prob.mean()
        if reduce == "none":
            ent_prob = {"root": r_ent, "rule": n_ent, "unary": t_ent}
        elif reduce == "mean":
            ent_prob = torch.cat([r_ent, n_ent, t_ent]).mean()
        return ent_prob

    def sentence_vectorizer(sent, model):
        sent_vec = []
        numw = 0
        for w in sent:
            try:
                if numw == 0:
                    sent_vec = model.wv[w]
                else:
                    sent_vec = np.add(sent_vec, model.wv[w])
                numw += 1
            except:
                pass
        return np.asarray(sent_vec) / numw

    def rules_similarity(self, rule=None, unary=None):
        if rule is None:
            rule = self.rules["rule"]
        if unary is None:
            unary = self.rules["unary"]

        b = rule.shape[0]

        tkl = self.kl_div(unary)  # KLD for terminal
        nkl = self.kl_div(rule)  # KLD for nonterminal
        tcs = self.cos_sim(unary)  # cos sim for terminal
        ncs = self.cos_sim(
            rule.reshape(b, self.NT, -1)
        )  # cos sim for nonterminal
        log_tcs = self.cos_sim(unary, log=True)  # log cos sim for terminal
        log_ncs = self.cos_sim(
            rule.reshape(b, self.NT, -1), log=True
        )  # log cos sim for nonterminal

        return {
            "kl_term": tkl,
            "kl_nonterm": nkl,
            "cos_term": tcs,
            "cos_nonterm": ncs,
            "log_cos_term": log_tcs,
            "log_cos_nonterm": log_ncs,
        }

    @property
    def metrics(self):
        if getattr(self, "_metrics", None) is None:
            self._metrics = self.rules_similarity()
        return self._metrics

    def clear_metrics(self):
        self._metrics = None

    @property
    def rules(self):
        if getattr(self, "_rules", None) is None:
            self._rules = self.forward()
        return self._rules

    @rules.setter
    def rules(self, rule):
        self._rules = rule

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

    def unique_terms(self, terms):
        b, n = terms.shape
        for t in terms:
            output, inverse, counts = torch.unique(
                t, return_inverse=True, return_counts=True
            )
            duplicated_index = counts.where(counts > 1)

    def batchify(self, rules, words):
        b = words.shape[0]

        root = rules["root"]
        root = root.expand(b, root.shape[-1])

        rule = rules["rule"]
        rule = rule.expand(b, *rule.shape)

        unary = rules["unary"]
        unary = unary[torch.arange(self.T)[None, None], words[:, :, None]]

        return {
            "unary": unary,
            "root": root,
            "rule": rule,
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
        depth=0,
        label=False,
        rule_update=False,
        **kwargs
    ):
        self.check_rule_update(rule_update)
        rules = self.batchify(self.rules, input["word"])
        result = self.decode(rules, input["seq_len"], decode_type, label)

        # if depth > 0:
        #     result["depth"] = self.part(
        #         self.rules, depth, mode="length", depth_output="full"
        #     )
        #     result["depth"] = result["depth"].exp()

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

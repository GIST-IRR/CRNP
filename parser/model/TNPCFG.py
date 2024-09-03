import torch
import torch.nn as nn
from ..model.PCFG_module import (
    PCFG_module,
    Term_parameterizer,
    Root_parameterizer,
)
from ..model.NeuralPCFG import NeuralPCFG

from ..pfs.td_partition_function import TDPartitionFunction
from ..pcfgs.pcfg import PCFG
from ..pcfgs.tdpcfg import TDPCFG
from torch.distributions.utils import logits_to_probs


class Nonterm_parameterizer(PCFG_module):
    def __init__(
        self, dim, NT, T, r, nonterm_emb=None, term_emb=None, rank_proj=False
    ):
        super(Nonterm_parameterizer, self).__init__()
        self.dim = dim
        self.NT = NT
        self.T = T
        self.r = r
        self.rank_proj = rank_proj

        if nonterm_emb is not None:
            self.nonterm_emb = nonterm_emb
        else:
            self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.dim))

        if term_emb is not None:
            self.term_emb = term_emb
        else:
            self.term_emb = nn.Parameter(torch.randn(self.T, self.dim))

        self.parent_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
        )
        self.left_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
        )
        self.right_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
        )

        if self.rank_proj:
            self.rank_proj_mlp = nn.Linear(self.dim, self.r, bias=False)
        else:
            for l in [self.parent_mlp, self.left_mlp, self.right_mlp]:
                l.append(nn.Linear(self.dim, self.r))

    def forward(self, softmax="log"):
        rule_state_emb = torch.cat([self.nonterm_emb, self.term_emb], dim=0)
        head = self.parent_mlp(self.nonterm_emb)
        left = self.left_mlp(rule_state_emb)
        right = self.right_mlp(rule_state_emb)

        if self.rank_proj:
            head = self.rank_proj_mlp(head)
            left = self.rank_proj_mlp(left)
            right = self.rank_proj_mlp(right)

        if softmax == "log":
            head = head.log_softmax(-1)
            left = left.log_softmax(-2)
            right = right.log_softmax(-2)
        elif softmax == "softmax":
            head = head.softmax(-1)
            left = left.softmax(-2)
            right = right.softmax(-2)
        return head, left, right


class TNPCFG(NeuralPCFG):
    def __init__(self, args):
        super(TNPCFG, self).__init__(args)
        self.pcfg = TDPCFG()
        self.part = TDPartitionFunction()

    def _set_arguments(self, args):
        super()._set_arguments(args)
        self.r = getattr(args, "r_dim", 1000)
        self.word_emb_size = getattr(args, "word_emb_size", 200)
        self.embedding_sharing = getattr(args, "embedding_sharing", False)
        self.rank_proj = getattr(args, "rank_proj", False)

    def _init_grammar(self):
        if self.embedding_sharing:
            print("embedding sharing")
            self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.s_dim))
            self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        else:
            self.nonterm_emb = None
            self.term_emb = None
        # terms
        self.terms = Term_parameterizer(
            self.s_dim,
            self.T,
            self.V,
            activation=self.activation,
            norm=self.norm,
            term_emb=self.term_emb,
        )
        self.nonterms = Nonterm_parameterizer(
            self.s_dim,
            self.NT,
            self.T,
            self.r,
            nonterm_emb=self.nonterm_emb,
            term_emb=self.term_emb,
            rank_proj=self.rank_proj,
        )
        # root
        self.root = Root_parameterizer(
            self.s_dim,
            self.NT,
            activation=self.activation,
            norm=self.norm,
            nonterm_emb=self.nonterm_emb,
        )

    def rules_similarity(self, unary=None):
        if unary is None:
            unary = self.rules["unary"]

        b = unary.shape[0]

        tkl = self.kl_div(unary)  # KLD for terminal
        tcs = self.cos_sim(unary)  # cos sim for terminal
        log_tcs = self.cos_sim(unary, log=True)  # log cos sim for terminal

        return {
            "kl_term": tkl,
            "cos_term": tcs,
            "log_cos_term": log_tcs,
        }

    @torch.no_grad()
    def entropy_rules(self, batch=False, probs=False, reduce="none"):
        head = self.rules["head"][0]
        left = self.rules["left"][0]
        right = self.rules["right"][0]

        head = head[:, None, None, :]
        left = left.unsqueeze(1)
        right = right.unsqueeze(0)
        ents = head.new_zeros(self.NT)
        for i, h in enumerate(head):
            t = (left + right + h).logsumexp(-1).reshape(-1)
            ent = logits_to_probs(t) * t
            ent = -ent.sum()
            ents[i] = ent

        if reduce == "mean":
            ents = ents.mean(-1)
        elif reduce == "sum":
            ents = ents.sum(-1)

        if probs:
            emax = 2 * self.max_entropy(self.NT + self.T)
            ents = (emax - ents) / emax

        return ents

    def compose(self, rules):
        head = rules["head"]
        left = rules["left"]
        right = rules["right"]
        unary = rules["unary"]
        root = rules["root"]

        h = head.shape[0]
        l = left.shape[0]
        r = right.shape[0]

        rule = torch.einsum("ir, jr, kr -> ijk", head, left, right)
        rule = rule.log().reshape(h, l, r)

        return {"unary": unary, "root": root, "rule": rule}

    def forward(self, **kwargs):
        root = self.root()
        unary = self.terms()
        head, left, right = self.nonterms()

        # for gradient conflict by using gradients of rules
        if self.training:
            root.retain_grad()
            # unary.retain_grad()
            head.retain_grad()
            left.retain_grad()
            right.retain_grad()

        self.clear_metrics()

        return {
            "unary": unary,
            "root": root,
            "head": head,
            "left": left,
            "right": right,
        }

    def batchify(self, rules, words):
        b = words.shape[0]

        root = rules["root"]
        unary = rules["unary"]

        root = root.expand(b, root.shape[-1])
        unary = unary[torch.arange(self.T)[None, None], words[:, :, None]]

        if len(rules.keys()) == 5:
            head = rules["head"]
            left = rules["left"]
            right = rules["right"]
            head = head.unsqueeze(0).expand(b, *head.shape)
            left = left.unsqueeze(0).expand(b, *left.shape)
            right = right.unsqueeze(0).expand(b, *right.shape)
            return {
                "unary": unary,
                "root": root,
                "head": head,
                "left": left,
                "right": right,
            }

        elif len(rules.keys()) == 3:
            rule = rules["rule"]
            rule = rule.unsqueeze(0).expand(b, *rule.shape)
            return {
                "unary": unary,
                "root": root,
                "rule": rule,
            }

    def loss(self, input, partition=False, soft=False, label=False, **kwargs):
        words = input["word"]
        self.rules = self.forward()
        self.rules = self.batchify(self.rules, words)

        result = self.pcfg(
            self.rules, self.rules["unary"], lens=input["seq_len"], label=label
        )
        # Partition function
        if partition:
            self.pf = self.part(self.rules, input["seq_len"], mode=self.mode)
            if soft:
                return -result["partition"].mean(), self.pf.mean()
            result["partition"] = result["partition"] - self.pf
        return -result["partition"].mean()

    def evaluate(
        self,
        input,
        decode_type="mbr",
        depth=0,
        label=False,
        rule_update=False,
        **kwargs
    ):
        if rule_update:
            need_update = True
        else:
            if hasattr(self, "rules"):
                need_update = False
            else:
                need_update = True

        if need_update:
            self.rules = self.forward()

        if decode_type == "viterbi":
            if not hasattr(self, "viterbi_pcfg"):
                self.viterbi_pcfg = PCFG()
                self.rules = self.compose(self.rules)

        rules = self.batchify(self.rules, input["word"])

        if decode_type == "viterbi":
            result = self.viterbi_pcfg(
                rules,
                rules["unary"],
                lens=input["seq_len"],
                viterbi=True,
                mbr=False,
                label=label,
            )
        elif decode_type == "mbr":
            result = self.pcfg(
                rules,
                rules["unary"],
                lens=input["seq_len"],
                viterbi=False,
                mbr=True,
                label=label,
            )
        else:
            raise NotImplementedError

        # if depth > 0:
        #     result["depth"] = self.part(
        #         rules, depth, mode="length", depth_output="full"
        #     )
        #     result["depth"] = result["depth"].exp()

        return result

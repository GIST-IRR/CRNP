import torch
import torch.nn as nn
from ..model.PCFG_module import (
    PCFG_module,
    UnaryRule_parameterizer,
)
from ..model.NeuralPCFG import NeuralPCFG

from ..pfs.td_partition_function import TDPartitionFunction
from ..pcfgs.pcfg import PCFG
from ..pcfgs.tdpcfg import TDPCFG
from ..modules.res import ResLayer, Sine


class Nonterm_parameterizer(PCFG_module):
    def __init__(
        self,
        dim,
        NT,
        T,
        r,
        nonterm_emb=None,
        term_emb=None,
        rank_proj=False,
        norm=None,
        activation="relu",
        elementwise_affine=True,
        mlp_mode="standard",
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

        if mlp_mode == "single":
            if norm == "layer":
                norm = nn.LayerNorm
            elif norm == "batch":
                norm = nn.BatchNorm1d

            if activation == "relu":
                activation = nn.ReLU
            elif activation == "sine":
                activation = Sine

            self.parent_mlp = nn.Sequential(
                nn.Linear(self.dim, self.dim),
            )
            self.left_mlp = nn.Sequential(
                nn.Linear(self.dim, self.dim),
            )
            self.right_mlp = nn.Sequential(
                nn.Linear(self.dim, self.dim),
            )
            if not isinstance(norm, str):
                self.parent_mlp.append(
                    norm(self.dim, elementwise_affine=elementwise_affine)
                )
                self.left_mlp.append(
                    norm(self.dim, elementwise_affine=elementwise_affine)
                )
                self.right_mlp.append(
                    norm(self.dim, elementwise_affine=elementwise_affine)
                )
            self.parent_mlp.append(activation())
            self.left_mlp.append(activation())
            self.right_mlp.append(activation())
        elif mlp_mode == "standard":
            self.parent_mlp = nn.Sequential(
                ResLayer(
                    self.dim,
                    self.dim,
                    activation=activation,
                    norm=norm,
                    elementwise_affine=elementwise_affine,
                ),
                ResLayer(
                    self.dim,
                    self.dim,
                    activation=activation,
                    norm=norm,
                    elementwise_affine=elementwise_affine,
                ),
            )
            self.left_mlp = nn.Sequential(
                ResLayer(
                    self.dim,
                    self.dim,
                    activation=activation,
                    norm=norm,
                    elementwise_affine=elementwise_affine,
                ),
                ResLayer(
                    self.dim,
                    self.dim,
                    activation=activation,
                    norm=norm,
                    elementwise_affine=elementwise_affine,
                ),
            )
            self.right_mlp = nn.Sequential(
                ResLayer(
                    self.dim,
                    self.dim,
                    activation=activation,
                    norm=norm,
                    elementwise_affine=elementwise_affine,
                ),
                ResLayer(
                    self.dim,
                    self.dim,
                    activation=activation,
                    norm=norm,
                    elementwise_affine=elementwise_affine,
                ),
            )

        if self.rank_proj:
            self.rank_proj = nn.Linear(self.dim, self.r, bias=False)
        else:
            for l in [self.parent_mlp, self.left_mlp, self.right_mlp]:
                l.append(nn.Linear(self.dim, self.r))

        if norm == "layer":
            # self.head_norm = nn.LayerNorm(self.r)
            # self.left_norm = nn.LayerNorm(self.r)
            # self.right_norm = nn.LayerNorm(self.r)
            self.norm = nn.LayerNorm(self.r, elementwise_affine=False)
            self.head_norm_std = nn.Parameter(torch.ones(NT, 1))
            self.left_norm_std = nn.Parameter(torch.ones(NT + T, 1))
            self.right_norm_std = nn.Parameter(torch.ones(NT + T, 1))
        else:
            self.norm = None

    def forward(self, softmax="log"):
        rule_state_emb = torch.cat([self.nonterm_emb, self.term_emb], dim=0)
        head = self.parent_mlp(self.nonterm_emb)
        left = self.left_mlp(rule_state_emb)
        right = self.right_mlp(rule_state_emb)

        if self.rank_proj:
            head = self.rank_proj(head)
            left = self.rank_proj(left)
            right = self.rank_proj(right)

        if self.norm:
            # head = self.head_norm(head)
            # left = self.left_norm(left)
            # right = self.right_norm(right)
            head = self.norm(head) * self.head_norm_std
            left = self.norm(left) * self.left_norm_std
            right = self.norm(right) * self.right_norm_std

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

    def _set_configs(self, args):
        super()._set_configs(args)
        self.r = getattr(args, "r_dim", 1000)
        self.word_emb_size = getattr(args, "word_emb_size", 200)
        self.rank_proj = getattr(args, "rank_proj", False)

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
            self.r,
            nonterm_emb=self.nonterm_emb,
            term_emb=self.term_emb,
            rank_proj=self.rank_proj,
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

    def forward(self, input=None, **kwargs):
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
        rules = self.batchify(self.rules, words)

        result = self.pcfg(
            rules, rules["unary"], lens=input["seq_len"], label=label
        )
        # Partition function
        if partition:
            self.pf = self.part(rules, input["seq_len"], mode=self.mode)
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
        self.check_rule_update(rule_update)

        if decode_type == "viterbi":
            if not isinstance(self.pcfg, PCFG):
                self.pcfg = PCFG()
                self.rules = self.compose(self.rules)

        rules = self.batchify(self.rules, input["word"])
        result = self.decode(rules, input["seq_len"], decode_type, label)

        return result

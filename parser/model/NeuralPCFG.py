import math
import re

import torch
import torch.nn as nn
from torch.distributions import Categorical
from nltk import Tree
from nltk.grammar import Nonterminal
import wandb

from parser.pfs.partition_function import PartitionFunction
from ..pcfgs.pcfg import PCFG
from .PCFG_module import (
    PCFG_module,
    Rule_Parameterizer,
    Nonterm_parameterizer,
    UnaryRule_parameterizer,
)

from torch_support import metric


class NeuralPCFG(PCFG_module):
    def __init__(self, args):
        super(NeuralPCFG, self).__init__()
        self.pcfg = PCFG()
        self.part = PartitionFunction()

        self._set_configs(args)
        self._init_grammar()
        self._initialize(self.init)

    def _set_configs(self, args):
        super()._set_configs(args)

        self.s_dim = getattr(args, "s_dim", 256)
        self.init = getattr(args, "init", "xavier_uniform")
        self.activation = getattr(args, "activation", "relu")
        self.dropout = getattr(args, "dropout", 0.0)

        self.temperature = getattr(args, "temperature", 1.0)
        self.smooth = getattr(args, "smooth", 0.0)

        self.embedding_sharing = getattr(args, "embedding_sharing", False)

        # Partition function
        self.mode = getattr(args, "mode", "length_unary")

    def set_forward_hooks(self, logger, current_step, once=True):
        if not hasattr(self, "forward_hooks_handles"):
            self.forward_hooks_handles = []
        for name, module in self.named_modules():
            # ignore modules
            if name in ["", "pcfg", "part"]:
                continue

            # Add name attribute to module
            module.name = name

            def activation_hook(module, input, output):
                k = f"activations/{module.name}"
                logger.log(
                    {
                        k: wandb.Histogram(output.detach().clone().cpu()),
                        "train/step": current_step,
                    },
                )
                if once:
                    module.forward_handle.remove()

            handle = module.register_forward_hook(activation_hook)
            module.forward_handle = handle
            self.forward_hooks_handles.append(handle)

    def drop_forward_hooks(self):
        if hasattr(self, "forward_hooks_handles"):
            return
        for handle in self.forward_hooks_handles:
            handle.remove()

    def load_state_dict(
        self, state_dict, module=None, frozen=False, strict=True, assign=False
    ):
        if module is None:
            return super().load_state_dict(state_dict, strict, assign)
        elif module == "root":
            self.root_emb = nn.Parameter(state_dict["root_emb"])
            std = {
                k.replace("root."): v
                for k, v in state_dict.items()
                if re.match("^root.", k)
            }
            self.root.load_state_dict(std)
            if frozen:
                self.root_emb.requires_grad_(False)
                self.root.requires_grad_(False)
        elif module == "nonterms":
            self.nonterm_emb = nn.Parameter(state_dict["nonterm_emb"])
            std = {
                k.replace("nonterms.", ""): v
                for k, v in state_dict.items()
                if re.match("^nonterms.", k)
            }
            self.nonterms.load_state_dict(std)
            if frozen:
                self.nonterm_emb.requires_grad_(False)
                self.nonterms.requires_grad_(False)
        elif module == "terms":
            self.term_emb = nn.Parameter(state_dict["term_emb"])
            # self.term_emb = nn.Parameter(state_dict["terms.parent_emb"])
            std = {
                k.replace("terms.", ""): v
                for k, v in state_dict.items()
                if re.match("^terms.", k)
            }
            self.terms.load_state_dict(std)
            if frozen:
                self.term_emb.requires_grad_(False)
                self.terms.requires_grad_(False)

    def _embedding_sharing(self):
        if self.embedding_sharing:
            print("embedding sharing")
        #     self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))
        #     self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.s_dim))
        #     self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        # else:
        #     self.root_emb = None
        #     self.nonterm_emb = None
        #     self.term_emb = None

    def _init_grammar(self):
        self._embedding_sharing()
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.s_dim))
        self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        # unary
        self.terms = Rule_Parameterizer(
            self.s_dim,
            self.T,
            self.V,
            **self.cfgs.unary,
        )
        # binary
        self.nonterms = Rule_Parameterizer(
            self.s_dim,
            self.NT,
            self.NT_T**2,
            **self.cfgs.binary,
        )
        # root
        self.root = Rule_Parameterizer(
            self.s_dim,
            1,
            self.NT,
            shared_child=self.nonterm_emb if self.embedding_sharing else None,
            **self.cfgs.root,
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

    def get_grammar(self, input=None, softmax="log_softmax"):
        # Root
        # self.root_tmp = self.root(self.root_emb, softmax=None)
        root = self.root(self.root_emb, softmax=softmax)
        # Rule
        # self.rule_tmp = self.nonterms(self.nonterm_emb, softmax=None)
        rule = self.nonterms(self.nonterm_emb, softmax=softmax)
        # Unary
        # self.unary_tmp = self.terms(self.term_emb, softmax=None)
        unary = self.terms(self.term_emb, softmax=softmax)

        rule = rule.reshape(self.NT, self.NT_T, self.NT_T)

        # for gradient conflict by using gradients of rules
        # if self.training:
        #     root.retain_grad()
        #     rule.retain_grad()
        #     # unary.retain_grad()
        #     self.root_tmp.retain_grad()
        #     self.rule_tmp.retain_grad()
        #     # self.unary_tmp.retain_grad()

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

    def forward(self, input, partition=False, soft=False, **kwargs):
        words = input["word"]

        # Calculate rule distributions
        self.rules = self.get_grammar(input)
        rules = self.batchify(self.rules, words)
        rules["word"] = input["word"]

        if partition:
            result = self.pcfg(rules, lens=input["seq_len"], topk=1)
            self.pf = self.part(rules, lens=input["seq_len"], mode=self.mode)
            if soft:
                return -result["partition"].mean(), self.pf.mean()
            result["partition"] = result["partition"] - self.pf
        else:
            result = self.pcfg(
                rules,
                lens=input["seq_len"],
                dropout=self.dropout,
            )

        # result["partition"] = result["partition"] / (2 * input["seq_len"] - 1)
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
            self.rules = self.get_grammar()

    def decode(self, rules, lens, decode_type, label=False, tree=None):
        if decode_type == "viterbi":
            result = self.pcfg(
                rules,
                lens=lens,
                viterbi=True,
                mbr=False,
                label=label,
                tree=tree,
            )
        elif decode_type == "mbr":
            result = self.pcfg(
                rules,
                lens=lens,
                viterbi=False,
                mbr=True,
                label=label,
                tree=tree,
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
        tree=None,
        **kwargs,
    ):
        self.check_rule_update(rule_update)
        rules = self.batchify(self.rules, input["word"])
        result = self.decode(
            rules, input["seq_len"], decode_type, label=label, tree=tree
        )

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

    def sample_trees(self, n=1, rule_update=False):
        self.check_rule_update(rule_update)
        logits = self.rules["root"][0]
        root_dist = Categorical(logits=logits)
        root_idx = root_dist.sample((1,))
        root_prob = logits[root_idx]

        symbol_stack = [root_idx]

        def get_children(symbol_idx):
            if symbol_idx < self.NT:
                logits = self.rules["rule"].flatten(1)
            else:
                logits = self.rules["unary"]
            logits = logits[symbol_idx][0]
            dist = Categorical(logits=logits)
            n_idx = dist.sample((1,))
            n_prob = logits[n_idx]
            n_idx = torch.unravel_index(n_idx, logits.shape)
            return n_idx, n_prob

        n_idx, n_prob = get_children(root_idx)
        for i in n_idx:
            idx, prob = get_children(i)

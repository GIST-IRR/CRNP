from abc import abstractmethod
from argparse import ArgumentError
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.nn.utils.parametrizations import orthogonal, weight_norm

from ..modules.res import ResLayer, Bilinear_ResLayer, default_residual_config
from ..modules.norm import MeanOnlyLayerNorm


def normalize(x, dim=-1):
    return x - x.mean(dim, keepdim=True)


class Rule_Parameterizer(nn.Module):

    def __init__(
        self,
        dim,
        n_parent,
        n_child,
        h_dim=None,
        shared_child=None,
        activation="relu",
        norm="rms",
        mlp_mode="standard",
        first_layer_norm=False,
        last_layer_norm=False,
        first_weight_norm=False,
        last_weight_norm=False,
        last_layer="linear",
        last_layer_bias=True,
        elementwise_affine=True,
        num_res_blocks=2,
        residual=ResLayer,
        residual_config=default_residual_config,
    ):
        super().__init__()
        self.dim = dim
        self.h_dim = h_dim if h_dim is not None else dim
        self.n_parent = n_parent
        self.n_child = n_child
        self.mlp_mode = mlp_mode
        self.last_layer = last_layer

        if self.mlp_mode in ["standard", "mlp"]:
            self.rule_mlp = nn.Sequential()
            # First layer
            fl = nn.Linear(self.dim, self.h_dim)
            if first_weight_norm:
                fl = weight_norm(fl)
            self.rule_mlp.append(fl)
            # Layer Norm
            if first_layer_norm:
                self.rule_mlp.append(
                    nn.LayerNorm(
                        self.h_dim, elementwise_affine=elementwise_affine
                    )
                )
                self.rule_mlp.append(nn.LeakyReLU())
                self.rule_mlp.append(nn.Linear(self.h_dim, self.h_dim))
            # Residual Blocks
            for _ in range(num_res_blocks):
                if self.mlp_mode == "standard":
                    self.rule_mlp.append(
                        residual(
                            self.h_dim,
                            self.h_dim,
                            **residual_config,
                        )
                    )
                elif self.mlp_mode == "mlp":
                    if norm is not None:
                        self.rule_mlp.append(norm(self.h_dim))
                    if activation is not None:
                        self.rule_mlp.append(activation())
                    self.rule_mlp.append(nn.Linear(self.h_dim, self.h_dim))
            # Last Layer Norm
            if last_layer_norm:
                self.rule_mlp.append(
                    nn.LayerNorm(
                        self.h_dim, elementwise_affine=elementwise_affine
                    )
                )
                # self.rule_mlp.append(nn.LeakyReLU())
            # Last Layer
            if last_layer == "linear":
                ll = nn.Linear(self.h_dim, self.n_child, bias=last_layer_bias)
                if last_weight_norm:
                    ll = weight_norm(ll)
                self.rule_mlp.append(ll)
            elif last_layer == "normalized":
                self.ll = nn.Parameter(torch.empty(self.n_child, self.h_dim))
                nn.init.xavier_uniform_(self.ll)

        elif self.mlp_mode == "single":
            if last_layer == "linear":
                l = nn.Linear(self.dim, self.n_child, bias=last_layer_bias)
                if last_weight_norm:
                    l = weight_norm(l)
                self.rule_mlp = nn.Sequential(l)
            elif last_layer == "normalized":
                self.ll = nn.Parameter(torch.empty(self.n_child, self.dim))
                nn.init.xavier_uniform_(self.ll)

        elif self.mlp_mode == None:
            self.register_module("rule_mlp", None)

        if shared_child:
            self.rule_mlp[-1].weight = shared_child

    def set_shared_child(self, shared_child):
        self.rule_mlp[-1].weight = shared_child
        self.n_child = shared_child.shape[0]

    def forward(self, parent_emb, softmax="log_softmax"):
        if hasattr(self, "rule_mlp"):
            rule_prob = self.rule_mlp(parent_emb)
        if self.last_layer == "normalized":
            ll = F.normalize(self.ll, p=2, dim=-1)
            if hasattr(self, "rule_mlp"):
                rule_prob = rule_prob @ ll.T
            else:
                rule_prob = parent_emb @ ll.T

        if softmax == "log_softmax":
            rule_prob = rule_prob.log_softmax(-1)
        elif softmax == "softmax":
            rule_prob = rule_prob.softmax(-1)

        return rule_prob


class Linear_Rule_Parameterizer(nn.Module):
    def __init__(
        self,
        dim,
        n_parent,
        n_child,
        h_dim=None,
        shared_child=None,
        activation="relu",
        norm="rms",
        mlp_mode="standard",
        first_layer_norm=False,
        last_layer_norm=False,
        first_weight_norm=False,
        last_weight_norm=False,
        last_layer="linear",
        last_layer_bias=True,
        elementwise_affine=True,
        num_res_blocks=2,
        residual=ResLayer,
        residual_config=default_residual_config,
    ):
        super().__init__()
        self.dim = dim
        self.h_dim = h_dim if h_dim is not None else dim
        self.n_parent = n_parent
        self.n_child = n_child
        self.mlp_mode = mlp_mode
        self.last_layer = last_layer

        if self.mlp_mode == "standard":
            self.rule_mlp = nn.Sequential()
            # First layer
            fl = nn.Linear(self.dim, self.h_dim)
            if first_weight_norm:
                fl = weight_norm(fl)
            self.rule_mlp.append(fl)
            # Layer Norm
            if first_layer_norm:
                self.rule_mlp.append(
                    nn.LayerNorm(
                        self.h_dim, elementwise_affine=elementwise_affine
                    )
                )
                self.rule_mlp.append(nn.LeakyReLU())
                self.rule_mlp.append(nn.Linear(self.h_dim, self.h_dim))
            # Residual Blocks
            for _ in range(num_res_blocks):
                self.rule_mlp.append(nn.RMSNorm(self.h_dim))
                self.rule_mlp.append(nn.GELU())
                self.rule_mlp.append(nn.Linear(self.h_dim, self.h_dim))
            # Last Layer Norms
            if last_layer_norm:
                self.rule_mlp.append(
                    nn.RMSNorm(
                        self.h_dim, elementwise_affine=elementwise_affine
                    )
                )
                # self.rule_mlp.append(nn.LeakyReLU())
            # Last Layer
            if last_layer == "linear":
                ll = nn.Linear(self.h_dim, self.n_child, bias=last_layer_bias)
                if last_weight_norm:
                    ll = weight_norm(ll)
                self.rule_mlp.append(ll)
            elif last_layer == "normalized":
                self.ll = nn.Parameter(torch.empty(self.n_child, self.h_dim))
                nn.init.xavier_uniform_(self.ll)

        elif self.mlp_mode == "single":
            l = nn.Linear(self.dim, self.n_child, bias=last_layer_bias)
            if last_weight_norm:
                l = weight_norm(l)
            self.rule_mlp = nn.Sequential(l)

        elif self.mlp_mode == None:
            self.register_module("rule_mlp", None)

        if shared_child:
            self.rule_mlp[-1].weight = shared_child

    def set_shared_child(self, shared_child):
        self.rule_mlp[-1].weight = shared_child
        self.n_child = shared_child.shape[0]

    def forward(self, parent_emb, softmax="log_softmax"):
        if hasattr(self, "rule_mlp"):
            rule_prob = self.rule_mlp(parent_emb)
        if self.last_layer == "normalized":
            ll = F.normalize(self.ll, p=2, dim=-1)
            if hasattr(self, "rule_mlp"):
                rule_prob = rule_prob @ ll.T
            else:
                rule_prob = parent_emb @ ll.T

        if softmax == "log_softmax":
            rule_prob = rule_prob.log_softmax(-1)
        elif softmax == "softmax":
            rule_prob = rule_prob.softmax(-1)

        return rule_prob


class UnaryRule_parameterizer(nn.Module):
    def __init__(
        self,
        dim,
        n_parent,
        n_child,
        h_dim=None,
        parent_emb=None,
        child_emb=None,
        activation="relu",
        mlp_mode="standard",
        num_res_blocks=2,
        scale=False,
        norm=None,
        last_norm=None,
        first_layer_norm=False,
        last_layer_norm=False,
        first_weight_norm=False,
        last_weight_norm=False,
        temp=1,
        last_layer=True,
        last_layer_bias=True,
        elementwise_affine=True,
        residual="standard",
        res_version=1,
        res_add_first=False,
        res_norm_first=False,
        res_dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.h_dim = h_dim if h_dim is not None else dim
        self.n_parent = n_parent
        self.n_child = n_child

        self.scale = scale
        self.mlp_mode = mlp_mode
        self.temp = temp

        self.parent_emb = parent_emb
        self.child_emb = child_emb

        self.shared_parent = parent_emb is not None
        self.shared_child = child_emb is not None

        if not self.shared_parent:
            self.parent_emb = nn.Parameter(
                torch.randn(self.n_parent, self.h_dim)
            )
        # if not self.shared_child:
        #     self.register_parameter("child_emb", None)

        if residual == "standard":
            residual = ResLayer
        elif residual == "bilinear":
            residual = Bilinear_ResLayer

        if mlp_mode == "standard":
            self.rule_mlp = nn.Sequential()
            # First layer
            fl = nn.Linear(self.dim, self.h_dim)
            if first_weight_norm:
                fl = weight_norm(fl)
            self.rule_mlp.append(fl)
            # Layer Norm
            if first_layer_norm:
                self.rule_mlp.append(
                    nn.LayerNorm(
                        self.h_dim, elementwise_affine=elementwise_affine
                    )
                )
                self.rule_mlp.append(nn.LeakyReLU())
            # Residual Blocks
            for _ in range(num_res_blocks):
                self.rule_mlp.append(
                    residual(
                        self.h_dim,
                        self.h_dim,
                        activation=activation,
                        norm=norm,
                        elementwise_affine=elementwise_affine,
                        add_first=res_add_first,
                        norm_first=res_norm_first,
                        dropout=res_dropout,
                        version=res_version,
                    )
                )
            # Last Layer Norm
            if last_layer_norm:
                self.rule_mlp.append(
                    nn.LayerNorm(
                        self.h_dim, elementwise_affine=elementwise_affine
                    )
                )
                # self.rule_mlp.append(nn.LeakyReLU())
            # Last Layer
            if last_layer:
                ll = nn.Linear(self.h_dim, self.n_child, bias=last_layer_bias)
                if last_weight_norm:
                    ll = weight_norm(ll)
                self.rule_mlp.append(ll)
        elif mlp_mode == "single":
            l = nn.Linear(self.dim, self.n_child, bias=last_layer_bias)
            if first_weight_norm:
                l = weight_norm(l)
            self.rule_mlp = l
        elif mlp_mode == "cosine similarity":
            self.rule_mlp = nn.CosineSimilarity(dim=-1)
            if not self.shared_child:
                self.child_emb = nn.Parameter(
                    torch.randn(self.n_child, self.h_dim)
                )
        elif mlp_mode == None:
            self.register_module("rule_mlp", None)

        if self.shared_child:
            if mlp_mode == "standard":
                self.rule_mlp[-1].weight = child_emb
            elif mlp_mode == "single":
                self.rule_mlp.weight = child_emb

        if last_norm == "batch":
            self.norm = nn.BatchNorm1d(self.n_child)
        elif last_norm == "layer":
            self.norm = nn.LayerNorm(self.n_child, elementwise_affine=False)
            self.norm_std = nn.Parameter(torch.ones(self.n_parent, 1))
        elif last_norm == "mo-layer":
            self.norm = MeanOnlyLayerNorm(
                self.n_child, elementwise_affine=False
            )
        else:
            self.register_parameter("norm", None)

    def set_parent_symbol(self, parent):
        self.parent_emb = parent
        self.n_parent = parent.shape[0]

    def set_child_symbol(self, child):
        self.child_emb = child
        self.n_child = child.shape[0]
        if self.mlp_mode != "cosine similarity" and child is not None:
            self.rule_mlp[-1].weight = child

    def forward(self, parent_emb=None, softmax="log_softmax"):
        if parent_emb is None:
            parent_emb = self.parent_emb

        if self.mlp_mode == "cosine similarity":
            rule_prob = self.rule_mlp(
                parent_emb.unsqueeze(1), self.child_emb.unsqueeze(0)
            )
            rule_prob = rule_prob * self.temp
        else:
            rule_prob = self.rule_mlp(parent_emb)

        if self.norm is not None:
            rule_prob = self.norm(rule_prob)
            # rule_prob = self.norm_std * rule_prob

        if self.scale:
            rule_prob = rule_prob / np.sqrt(rule_prob.size(-1))

        if softmax == "log_softmax":
            rule_prob = rule_prob.log_softmax(-1)
        elif softmax == "softmax":
            rule_prob = rule_prob.softmax(-1)

        return rule_prob


class Nonterm_parameterizer(nn.Module):
    def __init__(
        self,
        dim,
        NT,
        T,
        h_dim=None,
        temperature=1.0,
        nonterm_emb=None,
        term_emb=None,
        mlp_mode="standard",
        num_res_blocks=2,
        compose_fn="compose",
        scale=False,
        norm=None,
        last_norm=None,
        temp=1,
        activation="relu",
        last_layer_bias=True,
        elementwise_affine=True,
        residual="standard",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.h_dim = h_dim if h_dim is not None else dim
        self.NT = NT
        self.T = T
        self.NT_T = self.NT + self.T

        self.scale = scale
        # self.softmax = softmax

        self.temperature = temperature

        self.mlp_mode = mlp_mode
        self.compose_fn = compose_fn
        self.temp = temp

        self.nonterm_emb = nonterm_emb
        self.term_emb = term_emb

        self.shared_nonterm = self.nonterm_emb is not None
        self.shared_term = self.term_emb is not None

        if not self.shared_nonterm:
            self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.h_dim))

        if residual == "standard":
            residual = ResLayer
        elif residual == "bilinear":
            residual = Bilinear_ResLayer

        if mlp_mode == "standard":
            if self.shared_nonterm and self.shared_term:
                if compose_fn == "compose":
                    self.children_compose = nn.Linear(self.dim * 2, self.dim)
                elif compose_fn == "expose":
                    self.parent_expose = nn.Linear(self.dim, self.dim * 2)
            else:
                self.rule_mlp = nn.Sequential(
                    nn.Linear(self.dim, self.h_dim),
                    *[
                        residual(
                            self.h_dim,
                            self.h_dim,
                            activation=activation,
                            norm=norm,
                            elementwise_affine=elementwise_affine,
                        )
                        for _ in range(num_res_blocks)
                    ],
                    nn.Linear(
                        self.h_dim,
                        (self.NT_T) ** 2,
                        bias=last_layer_bias,
                    ),
                )
                self.register_parameter("children_compose", None)
        elif mlp_mode == "single":
            # self.rule_mlp = nn.Sequential(
            #     weight_norm(nn.Linear(self.dim, self.NT_T**2))
            # )
            self.rule_mlp = nn.Sequential(nn.Linear(self.dim, self.NT_T**2))
        elif mlp_mode == "cosine similarity":
            self.rule_mlp = nn.CosineSimilarity(dim=-1)
            if not self.shared_term:
                self.term_emb = nn.Parameter(torch.randn(self.T, self.h_dim))
            if compose_fn == "compose":
                self.children_compose = nn.Linear(self.dim * 2, self.dim)
            elif compose_fn == "compose_mlp":
                self.children_compose = nn.Sequential(
                    nn.Linear(self.dim * 2, self.dim),
                    nn.ReLU(),
                    nn.Linear(self.dim, self.dim),
                )
            elif compose_fn == "expose":
                self.parent_expose = nn.Linear(self.dim, self.dim * 2)
        elif mlp_mode == None:
            self.register_module("rule_mlp", None)

        if last_norm == "batch":
            self.norm = nn.BatchNorm1d(self.NT_T**2)
        elif last_norm == "layer":
            self.norm = nn.LayerNorm(self.NT_T**2, elementwise_affine=False)
            self.norm_std = nn.Parameter(torch.ones(self.NT, 1))
        elif last_norm == "mo-layer":
            self.norm = MeanOnlyLayerNorm(
                self.NT_T**2, elementwise_affine=False
            )
        else:
            self.register_parameter("norm", None)

    def set_nonterm_symbol(self, nonterm):
        self.nonterm_emb = nonterm
        self.NT = nonterm.shape[0]
        self.NT_T = self.NT + self.T

    def set_term_symbol(self, term):
        self.term_emb = term
        self.T = term.shape[0]
        self.NT_T = self.NT + self.T

    def get_children_emb(self):
        children_emb = torch.cat([self.nonterm_emb, self.term_emb])
        children_emb = torch.cat(
            [
                children_emb.unsqueeze(1).expand(-1, self.NT_T, -1),
                children_emb.unsqueeze(0).expand(self.NT_T, -1, -1),
            ],
            dim=2,
        )
        # children_emb = self.children_compose(children_emb)
        return children_emb

    def forward(self, parent_emb=None, softmax="log_softmax", reshape=False):
        if parent_emb is None:
            parent_emb = self.nonterm_emb

        def setup_emb():
            nonterm_emb = parent_emb
            children_emb = self.get_children_emb()
            if self.compose_fn == "compose":
                children_emb = self.children_compose(children_emb)
            elif self.compose_fn == "expose":
                nonterm_emb = self.parent_expose(nonterm_emb)
            return nonterm_emb, children_emb

        if self.mlp_mode == "cosine similarity":
            nonterm_emb, children_emb = setup_emb()
            nonterm_prob = self.rule_mlp(
                nonterm_emb[:, None, None, ...],
                children_emb[None, ...],
            )
            nonterm_prob = nonterm_prob * self.temp
            nonterm_prob = nonterm_prob.reshape(-1, self.NT_T**2)
        else:
            if self.shared_nonterm and self.shared_term:
                nonterm_emb, children_emb = setup_emb()
                children_emb = children_emb.reshape(self.NT_T**2, -1)
                nonterm_prob = nonterm_emb @ children_emb.T
            else:
                nonterm_prob = self.rule_mlp(parent_emb)

        if self.norm is not None:
            nonterm_prob = self.norm(nonterm_prob)
            # nonterm_prob = self.norm_std * nonterm_prob

        if self.scale:
            nonterm_prob = nonterm_prob / np.sqrt(nonterm_prob.size(-1))

        if softmax == "log_softmax":
            nonterm_prob = nonterm_prob / self.temperature
            nonterm_prob = nonterm_prob.log_softmax(-1)
        elif softmax == "softmax":
            nonterm_prob = nonterm_prob / self.temperature
            nonterm_prob = nonterm_prob.softmax(-1)

        if reshape:
            nonterm_prob = nonterm_prob.reshape(self.NT, self.NT_T, self.NT_T)

        return nonterm_prob


class PCFG_module(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._no_initialize = []

    # Module tools
    def _initialize(self, mode="xavier_uniform", value=0.0):
        if mode == "no_init":
            return
        elif mode == "orthogonal":
            nn.init.orthogonal_(self.root_emb)
            nn.init.orthogonal_(self.nonterm_emb)
            nn.init.orthogonal_(self.term_emb)
            return
        # Original Method
        if self.activation == "gelu":
            g = 1.0
        else:
            g = nn.init.calculate_gain(self.activation)

        for n, p in self.named_parameters():
            if "weight" not in n and "emb" not in n:
                continue
            if "parametrizations" in n:
                continue
            if n in self._no_initialize:
                continue
            if mode == "xavier_uniform":
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=g)
            elif mode == "xavier_normal":
                if p.dim() > 1:
                    nn.init.xavier_normal_(p, gain=g)
            elif mode == "uniform":
                if p.dim() > 1:
                    nn.init.uniform_(p)
            elif mode == "normal":
                if p.dim() > 1:
                    nn.init.normal_(p)
            # elif mode == "orthogonal":
            #     if p.dim() > 1:
            #         nn.init.orthogonal_(p)
            elif mode == "constant":
                # # Init with constant 0.0009
                n = n.split(".")[0]
                if n == "terms":
                    nn.init.constant_(p, 0.001)
                else:
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
            elif mode == "mean":
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                val = p.mean()
                nn.init.constant_(p, val)

    def _set_configs(self, cfgs):
        self.cfgs = cfgs
        # number of symbols
        self.NT = getattr(cfgs, "NT", 30)
        self.T = getattr(cfgs, "T", 60)
        self.NT_T = self.NT + self.T
        self.V = getattr(cfgs, "V", 10003)

    def named_parameters_without(self, key="terms"):
        for name, param in self.named_parameters():
            module_name = name.split(".")[0]
            if module_name != key:
                yield param

    # Grammar tools
    @property
    def rules(self):
        if getattr(self, "_rules", None) is None:
            self._rules = self.get_grammar()
        return self._rules

    @rules.setter
    def rules(self, rule):
        self._rules = rule

    @abstractmethod
    def get_grammar(self):
        raise NotImplementedError

    def clear_grammar(self):
        # This function is used when the network is updated
        # Updated network will have different rules
        self.rules = None

    @property
    def metrics(self):
        if getattr(self, "_metrics", None) is None:
            self._metrics = None
        return self._metrics

    def clear_metrics(self):
        self._metrics = None

    # What?
    def batch_dot(self, x, y):
        return (x * y).sum(-1, keepdims=True)

    num_trees_cache = {}

    def num_trees(self, len):
        if isinstance(len, torch.Tensor):
            len = len.item()
        if len == 1 or len == 2:
            return 1
        else:
            if len in self.num_trees_cache:
                num = self.num_trees_cache[len]
            else:
                num = 0
                for i in range(1, len):
                    num += self.num_trees(i) * self.num_trees(len - i)
                self.num_trees_cache[len] = num
            return num

    def max_entropy(self, num):
        return math.log(num)

    # Entropy
    def _entropy(self, rule, batch=False, reduce="none", probs=False):
        if rule.dim() == 2:
            rule = rule.unsqueeze(1)
        elif rule.dim() == 3:
            pass
        elif rule.dim() == 4:
            rule = rule.reshape(*rule.shape[:2], -1)
        else:
            raise ArgumentError(
                f"Wrong size of rule tensor. The allowed size is (2, 3, 4), but given tensor is {rule.dim()}"
            )

        b, n_parent, n_children = rule.shape
        if batch:
            ent = rule.new_zeros((b, n_parent))
            for i in range(b):
                for j in range(n_parent):
                    ent[i, j] = dist.categorical.Categorical(
                        logits=rule[i, j]
                    ).entropy()
        else:
            rule = rule[0]
            ent = rule.new_zeros((n_parent,))
            for i in range(n_parent):
                ent[i] = dist.categorical.Categorical(logits=rule[i]).entropy()

        if reduce == "mean":
            ent = ent.mean(-1)
        elif reduce == "sum":
            ent = ent.sum(-1)

        if probs:
            emax = self.max_entropy(n_children)
            ent = (emax - ent) / emax

        return ent

    @torch.no_grad()
    def entropy_root(self, batch=False, probs=False, reduce="none"):
        return self._entropy(
            self.rules["root"], batch=batch, probs=probs, reduce=reduce
        )

    @torch.no_grad()
    def entropy_terms(self, batch=False, probs=False, reduce="none"):
        return self._entropy(
            self.rules["unary"], batch=batch, probs=probs, reduce=reduce
        )

    # Parameter Update
    def update_depth(self, depth):
        self.depth = depth

    def update_dropout(self, rate):
        self.apply_dropout = self.init_dropout * rate

    def clear_rules_grad(self):
        for k, v in self.rules.items():
            if k == "kl":
                continue
            v.grad = None

    def get_grad(self):
        grad = []
        for p in self.parameters():
            grad.append(p.grad.reshape(-1))
        return torch.cat(grad)

    def set_grad(self, grad):
        total_num = 0
        for p in self.parameters():
            shape = p.grad.shape
            num = p.grad.numel()
            p.grad = p.grad + grad[total_num : total_num + num].reshape(*shape)
            total_num += num

    def get_rules_grad(self, flatten=False):
        b = 0
        grad = []
        for i, (k, v) in enumerate(self.rules.items()):
            if k == "kl":
                continue
            if i == 0:
                b = v.shape[0]
            grad.append(v.grad)
        if flatten:
            grad = [g.reshape(b, -1) for g in grad]
        return grad

    def get_X_Y_Z(self, rule):
        NTs = slice(0, self.NT)
        return rule[:, :, NTs, NTs]

    def get_X_Y_z(self, rule):
        NTs = slice(0, self.NT)
        Ts = slice(self.NT, self.NT + self.T)
        return rule[:, :, NTs, Ts]

    def get_X_y_Z(self, rule):
        NTs = slice(0, self.NT)
        Ts = slice(self.NT, self.NT + self.T)
        return rule[:, :, Ts, NTs]

    def get_X_y_z(self, rule):
        Ts = slice(self.NT, self.NT + self.T)
        return rule[:, :, Ts, Ts]

    def get_rules_grad_category(self):
        b = 0
        grad = {}
        for i, (k, v) in enumerate(self.rules.items()):
            if k == "kl":
                continue
            if i == 0:
                b = v.shape[0]
            g = v.grad
            if k == "rule":
                g = g.reshape(b, g.shape[1], -1)
            grad[k] = g
        return grad

    def backward_rules(self, grad):
        total_num = 0
        for k, v in self.rules.items():
            if k == "kl":
                continue
            shape = v.shape
            num = v[0].numel()
            v.backward(
                grad[:, total_num : total_num + num].reshape(*shape),
                retain_graph=True,
            )
            total_num += num

    def backward_rules_category(self, grad):
        for k, v in grad.items():
            if k == "rule":
                v = v.reshape(*self.rules[k].shape)
            self.rules[k].backward(v, retain_graph=True)

    def term_from_unary(self, word, term, smooth=0.0):
        n = word.shape[1]
        b = term.shape[0]
        term = term.unsqueeze(1).expand(b, n, self.T, self.V)

        # indices = word[..., None, None].expand(b, n, self.T, 1)
        # return torch.gather(term, 3, indices).squeeze(3)

        # # Smoothing
        word = F.one_hot(word, num_classes=self.V)
        smooth_weight = word * (1 - smooth) + smooth / self.V
        term = term + smooth_weight.unsqueeze(2).log()
        term = term.logsumexp(-1)

        return term

    def soft_backward(
        self,
        loss,
        z_l,
        optimizer,
        dambda=1.0,
        target="rule",
        mode="projection",
    ):
        def batch_dot(x, y):
            return (x * y).sum(-1, keepdims=True)

        def projection(x, y):
            scale = batch_dot(x, y) / batch_dot(y, y)
            return scale * y, scale

        loss = loss.mean()
        z_l = z_l.mean()
        # Get dL_w
        loss.backward(retain_graph=True)
        if target == "rule":
            g_loss = self.get_rules_grad()  # main vector
            # g_loss = self.get_rules_grad_category()
            # self.save_rule_heatmap(g_loss[-1][0], dirname='figure', filename='loss_gradient.png', abs=False, symbol=False)
            self.clear_rules_grad()
        elif target == "parameter":
            g_loss = self.get_grad()
            g_loss_norm = batch_dot(g_loss, g_loss).sqrt()
        optimizer.zero_grad()
        # Get dZ_l
        z_l.backward(retain_graph=True)
        if target == "rule":
            g_z_l = self.get_rules_grad()
            # g_z_l = self.get_rules_grad_category()
            # self.save_rule_heatmap(g_z_l[-1][0], dirname='figure', filename='z_gradient.png', abs=False, symbol=False)
            self.clear_rules_grad()
        elif target == "parameter":
            g_z_l = self.get_grad()
            g_z_l_norm = batch_dot(g_z_l, g_z_l).sqrt()
        optimizer.zero_grad()

        # if target == 'parameter':
        #     g_rule = self.get_rules_grad()
        #     self.save_rule_heatmap(g_rule[-1][0], dirname='figure', filename='rule_gradient.png', abs=False, symbol=False)

        # tmp
        # TODO: remove unused computing
        # loss.backward(retain_graph=True)
        # grad_output = torch.tensor(dambda)
        # z_l.backward(grad_output, retain_graph=True)
        # tmp_g_z_l = self.get_grad()
        # optimizer.zero_grad()

        if mode == "both":
            if target == "rule":
                g_r = [g_l + dambda * g_z for g_l, g_z in zip(g_loss, g_z_l)]
                # self.save_rule_heatmap(g_r[-1][0], dirname='figure', filename='rule_gradient.png', abs=False, symbol=False)
            elif target == "parameter":
                g_r = g_loss + dambda * g_z_l
        elif mode == "projection":
            g_proj, proj_scale = projection(g_z_l, g_loss)
            g_orth = g_z_l - g_proj
            g_proj_norm = batch_dot(g_proj, g_proj).sqrt()
            g_orth_norm = batch_dot(g_orth, g_orth).sqrt()
            g_r = g_loss + g_proj + dambda * g_orth
            # g_r = g_loss + dambda * g_z_l
            # g_r = {}
            # for k, v in dambda.items():
            #     if g_z_l[k].dim() == 3:
            #         v = v[None, :, None]
            #     g_r[k] = g_loss[k] + v * g_z_l[k]
        elif mode == "orthogonal":
            # oproj_{dL_w}{dZ_l} = dZ_l - proj_{dL_w}{dZ_l}
            g_oproj = g_z_l - projection(g_z_l, g_loss)
            # dL_BCLs = dL_w + oproj_{dL_w}{dZ_l}
            g_r = g_loss + g_oproj

        # Re-calculate soft BCL
        if target == "rule":
            # self.backward_rules_category(g_r)
            # b = g_loss['root'].shape[0]
            # g_loss = torch.cat([g.reshape(b, -1) for g in g_loss.values()], dim=-1)
            # g_z_l = torch.cat([g.reshape(b, -1) for g in g_z_l.values()], dim=-1)
            # g_r = torch.cat([g.reshape(b, -1) for g in g_r.values()], dim=-1)
            self.backward_rules(g_r)
        elif target == "parameter":
            # grad_norm = g_orth_norm.mean()
            # grad_norm.backward()
            self.set_grad(g_r)

        return {
            "g_loss": g_loss,
            "g_z_l": g_z_l,
            "g_r": g_r,
            # 'proj_scale': proj_scale,
            "g_loss_norm": g_loss_norm,
            "g_z_l_norm": g_z_l_norm,
            # 'g_proj_norm': g_proj_norm,
            # 'g_orth_norm': g_orth_norm
        }

from argparse import ArgumentError
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.nn.utils.parametrizations import _Orthogonal

from ..modules.res import ResLayer

import math


class UnaryRule_parameterizer(nn.Module):
    def __init__(
        self,
        dim,
        n_parent,
        n_child,
        h_dim=None,
        activation="relu",
        parent_emb=None,
        child_emb=None,
        orthogonal=False,
        mlp_mode="standard",
        softmax=True,
        norm=None,
        temp=1,
    ):
        super().__init__()
        self.dim = dim
        self.h_dim = h_dim if h_dim is not None else dim
        self.n_parent = n_parent
        self.n_child = n_child

        self.softmax = softmax
        self.mlp_mode = mlp_mode
        self.temp = temp

        if parent_emb is None:
            self.parent_emb = nn.Parameter(
                torch.randn(self.n_parent, self.h_dim)
            )
        else:
            self.parent_emb = parent_emb

        if child_emb is None:
            self.child_emb = nn.Parameter(
                torch.randn(self.n_child, self.h_dim)
            )
        else:
            self.child_emb = child_emb

        if orthogonal:
            self.orth = _Orthogonal()

        if mlp_mode == "standard":
            self.rule_mlp = nn.Sequential(
                nn.Linear(self.dim, self.h_dim),
                ResLayer(self.h_dim, self.h_dim, activation=activation),
                ResLayer(self.h_dim, self.h_dim, activation=activation),
                nn.Linear(self.h_dim, self.n_child),
            )
        elif mlp_mode == "single":
            self.rule_mlp = nn.Linear(self.dim, self.n_child, bias=False)
        elif mlp_mode == "cosine similarity":
            self.rule_mlp = nn.CosineSimilarity(dim=-1)
        elif mlp_mode == None:
            self.register_module("rule_mlp", None)

        if mlp_mode != "cosine similarity" and child_emb is not None:
            self.rule_mlp[-1].weight = child_emb

        if norm == "batch":
            self.norm = nn.BatchNorm1d(self.n_child)
        elif norm == "layer":
            self.norm = nn.LayerNorm(self.n_child)
        else:
            self.register_parameter("norm", None)

    def forward(self):
        if self.mlp_mode == "cosine similarity":
            rule_prob = self.rule_mlp(
                self.parent_emb.unsqueeze(1), self.child_emb.unsqueeze(0)
            )
            rule_prob = rule_prob * self.temp
        else:
            rule_prob = self.rule_mlp(self.parent_emb)

        if self.norm is not None:
            rule_prob = self.norm(rule_prob)

        if self.softmax:
            rule_prob = rule_prob.log_softmax(-1)
        return rule_prob


class Term_parameterizer(UnaryRule_parameterizer):
    def __init__(
        self,
        dim,
        T,
        V,
        h_dim=None,
        activation="relu",
        term_emb=None,
        word_emb=None,
        orthogonal=False,
        mlp_mode="standard",
        softmax=True,
        norm=None,
        temp=1,
    ):
        super(Term_parameterizer, self).__init__(
            dim=dim,
            n_parent=T,
            n_child=V,
            h_dim=h_dim,
            activation=activation,
            parent_emb=term_emb,
            child_emb=word_emb,
            orthogonal=orthogonal,
            mlp_mode=mlp_mode,
            softmax=softmax,
            norm=norm,
            temp=temp,
        )


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
        compose_fn="linear",
        no_rule_layer=False,
        softmax=True,
        norm=None,
        temp=1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.h_dim = h_dim if h_dim is not None else dim
        self.NT = NT
        self.T = T
        self.NT_T = self.NT + self.T

        self.softmax = softmax

        self.temperature = temperature

        self.mlp_mode = mlp_mode
        self.temp = temp

        if nonterm_emb is None:
            self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.h_dim))
        else:
            self.nonterm_emb = nonterm_emb

        if term_emb is None:
            self.term_emb = nn.Parameter(torch.randn(self.T, self.h_dim))
        else:
            self.term_emb = term_emb

        if not no_rule_layer:
            if mlp_mode == "standard":
                self.rule_mlp = nn.Linear(self.dim, (self.NT_T) ** 2)
                self.register_parameter("children_compose", None)
            elif mlp_mode == "cosine similarity":
                self.rule_mlp = nn.CosineSimilarity(dim=-1)
                if compose_fn == "linear":
                    self.children_compose = nn.Linear(self.dim * 2, self.dim)
                elif compose_fn == "mlp":
                    self.children_compose = nn.Sequential(
                        nn.Linear(self.dim * 2, self.dim),
                        nn.ReLU(),
                        nn.Linear(self.dim, self.dim),
                    )
            elif mlp_mode == None:
                self.register_module("rule_mlp", None)

        else:
            self.register_parameter("rule_mlp", None)

        if norm == "batch":
            self.norm = nn.BatchNorm1d(self.NT_T**2)
        elif norm == "layer":
            self.norm = nn.LayerNorm(self.NT_T**2)
        else:
            self.register_parameter("norm", None)

    def forward(self, reshape=False):
        if self.mlp_mode == "cosine similarity":
            children_emb = torch.cat([self.nonterm_emb, self.term_emb])
            children_emb = torch.cat(
                [
                    children_emb.unsqueeze(1).expand(-1, self.NT_T, -1),
                    children_emb.unsqueeze(0).expand(self.NT_T, -1, -1),
                ],
                dim=2,
            )
            children_emb = self.children_compose(children_emb)
            nonterm_prob = self.rule_mlp(
                self.nonterm_emb[:, None, None, ...],
                children_emb[None, ...],
            )
            nonterm_prob = nonterm_prob * self.temp
            nonterm_prob = nonterm_prob.reshape(-1, self.NT_T**2)
        else:
            nonterm_prob = self.rule_mlp(self.nonterm_emb)

        if self.norm is not None:
            nonterm_prob = self.norm(nonterm_prob)

        if self.softmax:
            nonterm_prob = nonterm_prob / self.temperature
            nonterm_prob = nonterm_prob.log_softmax(-1)

        if reshape:
            nonterm_prob = nonterm_prob.reshape(self.NT, self.NT_T, self.NT_T)

        return nonterm_prob


class Root_parameterizer(UnaryRule_parameterizer):
    def __init__(
        self,
        dim,
        ROOT,
        NT,
        h_dim=None,
        root_emb=None,
        nonterm_emb=None,
        orthogonal=False,
        mlp_mode="standard",
        activation="relu",
        softmax=True,
        norm=None,
        temp=1,
    ):
        super(Root_parameterizer, self).__init__(
            dim=dim,
            n_parent=ROOT,
            n_child=NT,
            h_dim=h_dim,
            activation=activation,
            parent_emb=root_emb,
            child_emb=nonterm_emb,
            orthogonal=orthogonal,
            mlp_mode=mlp_mode,
            softmax=softmax,
            norm=norm,
            temp=temp,
        )


class PCFG_module(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._no_initialize = []

    def _initialize(self, mode="xavier_uniform", value=0.0):
        # Original Method
        for n, p in self.named_parameters():
            if n in self._no_initialize:
                continue
            if mode == "xavier_uniform":
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)
            elif mode == "xavier_normal":
                if p.dim() > 1:
                    torch.nn.init.xavier_normal_(p)
            elif mode == "uniform":
                if p.dim() > 1:
                    torch.nn.init.uniform_(p)
            elif mode == "normal":
                if p.dim() > 1:
                    torch.nn.init.normal_(p)
            elif mode == "orthogonal":
                if p.dim() > 1:
                    torch.nn.init.orthogonal_(p)
            elif mode == "constant":
                # # Init with constant 0.0009
                n = n.split(".")[0]
                if n == "terms":
                    torch.nn.init.constant_(p, 0.001)
                else:
                    if p.dim() > 1:
                        torch.nn.init.xavier_uniform_(p)
            elif mode == "mean":
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)
                val = p.mean()
                torch.nn.init.constant_(p, val)
        # # Init with mean of each layer
        # for n, p in self.named_parameters():
        #     if p.dim() > 1:
        #         torch.nn.init.xavier_uniform_(p)
        #     if n.split(".")[0] == "terms":
        #         val = p.mean()
        #         torch.nn.init.constant_(p, val)
        # # Init with mean of each layer for all
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         torch.nn.init.xavier_uniform_(p)
        #     val = p.mean()
        #     torch.nn.init.constant_(p, val)
        # # Init with mean of each layer for all & Fix terms
        # for n, p in self.named_parameters():
        #     if p.dim() > 1:
        #         torch.nn.init.xavier_uniform_(p)
        #     if n.split(".")[0] == "terms":
        #         val = p.mean()
        #         torch.nn.init.constant_(p, val)
        #         p.requires_grad = False
        # # Init with mean of each layer for nonterminal
        # for n, p in self.named_parameters():
        #     if p.dim() > 1:
        #         torch.nn.init.xavier_uniform_(p)
        #     if n.split(".")[0] != "terms":
        #         val = p.mean()
        #         torch.nn.init.constant_(p, val)
        #     else:
        #         p.requires_grad = False
        # # Init with mean of each layer for nonterminal & Fix terms
        # for n, p in self.named_parameters():
        #     if p.dim() > 1:
        #         torch.nn.init.xavier_uniform_(p)
        #     val = p.mean()
        #     torch.nn.init.constant_(p, val)
        #     if n.split(".")[0] != "terms":
        #         p.requires_grad = False
        #
        # for n, p in self.named_parameters():
        #     if p.dim() > 1:
        #         torch.nn.init.xavier_uniform_(p)
        #     if n.split(".")[0] == "terms":
        #         val = p.mean()
        #         torch.nn.init.constant_(p, val)
        #         p.requires_grad = False

    def clear_grammar(self):
        # This function is used when the network is updated
        # Updated network will have different rules
        self.rules = None

    def withoutTerm_parameters(self):
        for name, param in self.named_parameters():
            module_name = name.split(".")[0]
            if module_name != "terms":
                yield param

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

    def cos_sim(self, x, log=False):
        b, cat = x.shape[:2]
        cs = x.new_zeros(cat, cat).fill_diagonal_(1).expand(b, -1, -1).clone()
        for i in range(cat):
            if i == cat - 1:
                continue
            u = x[:, i : i + 1].expand(-1, cat - i - 1, -1)
            o = x[:, i + 1 : cat]
            if not log:
                u = u.exp()
                o = o.exp()
            cosine_score = F.cosine_similarity(u, o, dim=2)
            cs[:, i, i + 1 : cat] = cosine_score
            cs[:, i + 1 : cat, i] = cosine_score
        return cs

    def kl_div(self, x):
        b, cat = x.shape[:2]
        kl = x.new_zeros(b, cat, cat)
        x = x.reshape(b, cat, -1)
        for i in range(cat):
            t = x[:, i : i + 1].expand(-1, cat, -1)
            kl_score = F.kl_div(t, x, log_target=True, reduction="none")
            kl_score = kl_score.sum(-1)
            kl[:, i] = kl_score
        # reverse ratio of kl score
        # mask = nkl.new_ones(nkl.shape[1:]).fill_diagonal_(0)
        # weight = 1 - (nkl / nkl.sum((1, 2), keepdims=True))
        # weight = (mask * weight).detach()
        # nkl = (weight * nkl).mean((1, 2))
        # nkl = nkl.mean()
        return kl

    def cos_sim_mean(self, x):
        cat = x.shape[1]
        x = x.tril(diagonal=-1)
        x = x.flatten(start_dim=1)
        x = x.abs()
        x = x.sum(-1)
        x = x / (cat * (cat - 1) / 2)
        return x

    def cos_sim_max(self, x):
        x = x.tril(diagonal=-1)
        x = x.flatten(start_dim=1)
        x = x.abs()
        return x.max(-1)[0]

    def js_div(self, x, y, log_target=False):
        raise NotImplementedError

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

    def update_depth(self, depth):
        self.depth = depth

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

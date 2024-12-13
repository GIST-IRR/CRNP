import torch
import torch.nn as nn
from parser.model.NeuralPCFG import NeuralPCFG
from parser.model.PCFG_module import (
    UnaryRule_parameterizer,
    Nonterm_parameterizer as NTP,
)
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class Root_parameterizer(UnaryRule_parameterizer):
    def __init__(self, dim, z_dim, NT) -> None:
        super().__init__(dim + z_dim, 1, NT, h_dim=dim, mlp_mode="standard")
        self.z_dim = z_dim

    def forward(self, z):
        b = z.shape[0]
        parent_emb = self.parent_emb.expand(b, self.h_dim)
        parent_emb = torch.cat([parent_emb, z], -1)

        rule_prob = self.rule_mlp(parent_emb).log_softmax(-1)
        return rule_prob


class Term_parameterizer(UnaryRule_parameterizer):
    def __init__(self, dim, z_dim, T, V) -> None:
        super().__init__(dim + z_dim, T, V, h_dim=dim, mlp_mode="standard")
        self.z_dim = z_dim

    def forward(self, z):
        b = z.shape[0]
        parent_emb = self.parent_emb.unsqueeze(0).expand(b, -1, -1)
        z_expand = z.unsqueeze(1).expand(b, self.n_parent, self.z_dim)
        parent_emb = torch.cat([parent_emb, z_expand], -1)

        rule_prob = self.rule_mlp(parent_emb).log_softmax(-1)
        return rule_prob


class Nonterm_parameterizer(NTP):
    def __init__(self, dim, z_dim, NT, T) -> None:
        super().__init__(dim + z_dim, NT, T, h_dim=dim, mlp_mode="standard")
        self.z_dim = z_dim

    def forward(self, z):
        b = z.shape[0]
        nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
            b, self.NT, self.h_dim
        )
        z_expand = z.unsqueeze(1).expand(b, self.NT, self.z_dim)
        nonterm_emb = torch.cat([nonterm_emb, z_expand], -1)

        rule_prob = self.rule_mlp(nonterm_emb).log_softmax(-1)
        rule_prob = rule_prob.reshape(b, self.NT, self.NT_T, self.NT_T)
        return rule_prob


class Encoder(nn.Module):
    def __init__(self, V, w_dim, h_dim, z_dim) -> None:
        super().__init__()
        self.V = V
        self.w_dim = w_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.enc_emb = nn.Embedding(self.V, self.w_dim)

        self.enc_rnn = nn.LSTM(
            self.w_dim,
            self.h_dim,
            bidirectional=True,
            num_layers=1,
            batch_first=True,
        )

        self.enc_out = nn.Linear(self.h_dim * 2, self.z_dim * 2)

    def forward(self, x, len):
        x_embbed = self.enc_emb(x)
        x_packed = pack_padded_sequence(
            x_embbed, len.cpu(), batch_first=True, enforce_sorted=False
        )
        h_packed, _ = self.enc_rnn(x_packed)
        padding_value = float("-inf")
        output, lengths = pad_packed_sequence(
            h_packed, batch_first=True, padding_value=padding_value
        )
        h = output.max(1)[0]
        out = self.enc_out(h)
        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim :]
        return mean, lvar


class CompoundPCFG(NeuralPCFG):
    def _set_configs(self, cfgs):
        super()._set_configs(cfgs)
        self.z_dim = getattr(cfgs, "z_dim")
        self.w_dim = getattr(cfgs, "w_dim")
        self.h_dim = getattr(cfgs, "h_dim")

    def _init_grammar(self):
        self.nonterms = Nonterm_parameterizer(
            self.s_dim, self.z_dim, self.NT, self.T
        )
        self.terms = Term_parameterizer(self.s_dim, self.z_dim, self.T, self.V)
        self.root = Root_parameterizer(self.s_dim, self.z_dim, self.NT)

        self.enc = Encoder(self.V, self.w_dim, self.h_dim, self.z_dim)

    def forward(self, input, evaluating=False):
        x = input["word"]
        b, n = x.shape[:2]
        seq_len = input["seq_len"]

        def kl(mean, logvar):
            result = -0.5 * (
                logvar - torch.pow(mean, 2) - torch.exp(logvar) + 1
            )
            return result

        # mean, lvar = enc(x)
        # z = mean

        mean, lvar = self.enc(x, seq_len)
        z = mean
        # z = torch.cat([mean, lvar], -1)

        if not evaluating:
            z = mean.new(b, mean.size(1)).normal_(0, 1)
            z = (0.5 * lvar).exp() * z + mean

        root, unary, rule = self.root(z), self.terms(z), self.nonterms(z)

        # for gradient conflict by using gradients of rules
        if self.training:
            root.retain_grad()
            # unary.retain_grad()
            rule.retain_grad()

        return {
            "unary": unary,
            "root": root,
            "rule": rule,
            "kl": kl(mean, lvar).sum(1),
        }

    def batchify(self, rules, words):

        b, n = words.shape[:2]
        unary = rules["unary"]
        unary = unary.gather(
            -1, words.unsqueeze(1).expand(b, self.T, words.shape[-1])
        ).transpose(-1, -2)

        return {
            "unary": unary,
            "root": rules["root"],
            "rule": rules["rule"],
            "kl": rules["kl"],
        }

    def loss(self, input, partition=False, max_depth=0, soft=False, **kwargs):
        res = super().loss(
            input,
            partition=partition,
            max_depth=max_depth,
            soft=soft,
            reduction=None,
        )
        return (res + self.rules["kl"]).mean()

    def evaluate(self, input, decode_type, depth=0, label=False, **kwargs):
        rules = self.forward(input, evaluating=True)
        rules = self.batchify(rules, input["word"])
        result = self.decode(rules, input["seq_len"], decode_type, label)

        if depth > 0:
            result["depth"] = self.part(
                rules, depth, mode="length", depth_output="full"
            )
            result["depth"] = result["depth"].exp()

        if "kl" in rules:
            result["partition"] -= rules["kl"]
        return result

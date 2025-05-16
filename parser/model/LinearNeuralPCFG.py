import torch
import torch.nn as nn
from parser.model import FNPCFG
from .PCFG_module import Linear_Rule_Parameterizer as Rule_Parameterizer


class LinearNeuralPCFG(FNPCFG):
    # def _set_configs(self, args):
    #     super()._set_configs(args)
    #     self.norm = getattr(args, "norm", "rms")

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

import torch
import torch.nn as nn

from .PFTNPCFG import Parse_Focusing
from .FNPCFG import FNPCFG


class FSPTNPCFG(Parse_Focusing, FNPCFG):
    "Fully-Shared Parse-foucsed TNPCFG"

    def __init__(self, args):
        super(FSPTNPCFG, self).__init__(args)
        self._setup_parse_focusing(args)
        self.sim = nn.CosineSimilarity(dim=-1)

    def _embedding_sharing(self):
        super()._embedding_sharing()

    def loss(self, input, partition=False, soft=False, label=False, **kwargs):
        loss = super().loss(
            input, partition=partition, soft=soft, label=label, **kwargs
        )
        nt_sim = self.sim(
            self.nonterm_emb.unsqueeze(1), self.nonterm_emb.unsqueeze(0)
        )
        t_sim = self.sim(
            self.term_emb.unsqueeze(1), self.term_emb.unsqueeze(0)
        )
        nt_sim = nt_sim.square().sqrt().mean()
        t_sim = t_sim.square().sqrt().mean()

        return loss + nt_sim + t_sim
        # return loss

    @property
    def root_emb(self):
        return self.root.root_emb

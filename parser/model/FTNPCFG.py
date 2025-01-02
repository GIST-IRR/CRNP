from ..pcfgs.tdpcfg import Fastest_TDPCFG
from ..model.TNPCFG import TNPCFG


class FTNPCFG(TNPCFG):
    def __init__(self, args):
        super(FTNPCFG, self).__init__(args)
        self.pcfg = Fastest_TDPCFG()

    def _set_configs(self, args):
        super()._set_configs(args)
        self.r = getattr(args, "r_dim", 1000)
        self.rank_proj = getattr(args, "rank_proj", True)
        self.init = getattr(args, "init", "xavier_normal")

    def forward(self, input=None, **kwargs):
        root = self.root()
        unary = self.terms()
        head, left, right = self.nonterms(softmax="softmax")

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

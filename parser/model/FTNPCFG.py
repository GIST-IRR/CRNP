from ..pcfgs.tdpcfg import Fastest_TDPCFG
from ..model.TNPCFG import TNPCFG


class FTNPCFG(TNPCFG):
    def __init__(self, args):
        super(FTNPCFG, self).__init__(args)
        self.pcfg = Fastest_TDPCFG()

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

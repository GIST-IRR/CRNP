from parser.model.FNPCFG import FNPCFG
from parser.model.PFTNPCFG import Parse_Focusing


class PFNPCFG(Parse_Focusing, FNPCFG):
    """Parse focused C-PCFG"""

    def __init__(self, args):
        super(PFNPCFG, self).__init__(args)
        self._setup_parse_focusing(args)

    def loss(self, input, partition=False, max_depth=0, soft=False, **kwargs):
        res = super().loss(
            input,
            partition=partition,
            max_depth=max_depth,
            soft=soft,
            reduction=None,
        )
        return res

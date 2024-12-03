from parser.model.TNPCFG import TNPCFG
from ..parse_foucsing.SplittedLabeledParseFocusing import (
    SplittedLabeledParseFocusing,
)


class SLPFTNPCFG(SplittedLabeledParseFocusing, TNPCFG):
    """Splitted Labeled Parse-Focused TN-PCFG"""

    def _set_arguments(self, args):
        self.symbol_split = getattr(args, "symbol_split", 2)
        args.NT = self.symbol_split * args.NT
        args.T = self.symbol_split * args.T
        super(SLPFTNPCFG, self)._set_arguments(args)

    def __init__(self, args):
        self._setup_parse_focusing(args)
        args.NT = len(self.idx2nt)
        args.T = len(self.idx2t)
        super(SLPFTNPCFG, self).__init__(args)

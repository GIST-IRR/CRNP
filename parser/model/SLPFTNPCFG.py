from parser.model.TNPCFG import TNPCFG
from ..parse_foucsing.SplittedLabeledParseFocusing import (
    SplittedLabeledParseFocusing,
)


class SLPFTNPCFG(SplittedLabeledParseFocusing, TNPCFG):
    """Splitted Labeled Parse-Focused TN-PCFG"""

    def _set_configs(self, cfgs):
        self.symbol_split = getattr(cfgs, "symbol_split", 2)
        cfgs.NT = self.symbol_split * cfgs.NT
        cfgs.T = self.symbol_split * cfgs.T
        super()._set_configs(cfgs)

    def __init__(self, args):
        self._setup_parse_focusing(args)
        super(SLPFTNPCFG, self).__init__(args)

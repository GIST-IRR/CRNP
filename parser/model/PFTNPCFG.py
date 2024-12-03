from ..model.FTNPCFG import FTNPCFG
from ..parse_foucsing.ParseFocusing import ParseFocusing


class PFTNPCFG(ParseFocusing, FTNPCFG):
    """Parse focused FTN-PCFG"""

    def __init__(self, args):
        self._setup_parse_focusing(args)
        super(PFTNPCFG, self).__init__(args)

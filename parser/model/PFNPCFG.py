from parser.model.FNPCFG import FNPCFG
from ..parse_foucsing.ParseFocusing import ParseFocusing


class PFNPCFG(ParseFocusing, FNPCFG):
    """Parse focused FN-PCFG"""

    def __init__(self, args):
        super(PFNPCFG, self).__init__(args)
        self._setup_parse_focusing(args)

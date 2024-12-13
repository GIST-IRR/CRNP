from parser.model.CompoundPCFG import CompoundPCFG
from ..parse_foucsing.ParseFocusing import ParseFocusing


class PFCPCFG(ParseFocusing, CompoundPCFG):
    """Parse focused C-PCFG"""

    def __init__(self, args):
        super(PFCPCFG, self).__init__(args)
        self._setup_parse_focusing(args)

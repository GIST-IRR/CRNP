from parser.model.TNPCFG import TNPCFG
from ..parse_foucsing.LabeldParseFocusing import LabeledParseFocusing


class LPFTNPCFG(LabeledParseFocusing, TNPCFG):
    """Labeled Parse-Focused TN-PCFG"""

    def __init__(self, args):
        self._setup_parse_focusing(args)
        super(LPFTNPCFG, self).__init__(args)

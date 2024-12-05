from ..model.FNPCFG import FNPCFG
from ..parse_foucsing.LabeldParseFocusing import LabeledParseFocusing


class LPFNPCFG(LabeledParseFocusing, FNPCFG):
    def __init__(self, args):
        self._setup_parse_focusing(args)
        super(LPFNPCFG, self).__init__(args)

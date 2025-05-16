from .LinearNeuralPCFG import LinearNeuralPCFG
from ..parse_foucsing.ParseFocusing import ParseFocusing


class PFLinearNPCFG(ParseFocusing, LinearNeuralPCFG):
    def __init__(self, args):
        super(PFLinearNPCFG, self).__init__(args)
        self._setup_parse_focusing(args)

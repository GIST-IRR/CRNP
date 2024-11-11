from ..pcfgs.pcfg import Faster_PCFG
from ..model.NeuralPCFG import NeuralPCFG


class FNPCFG(NeuralPCFG):
    """FGG-NPCFG"""

    def __init__(self, args):
        super(FNPCFG, self).__init__(args)
        self.pcfg = Faster_PCFG()

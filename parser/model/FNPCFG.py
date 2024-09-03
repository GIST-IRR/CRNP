from ..pcfgs.pcfg import Faster_PCFG
from ..model.NeuralPCFG import NeuralPCFG
from .PCFG_module import (
    Term_parameterizer,
    Nonterm_parameterizer,
    Root_parameterizer,
)


class FNPCFG(NeuralPCFG):
    """FGG-NPCFG"""

    def __init__(self, args):
        super(FNPCFG, self).__init__(args)
        self.pcfg = Faster_PCFG()

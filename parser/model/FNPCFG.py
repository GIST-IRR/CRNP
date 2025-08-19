from ..pcfgs.pcfg import Faster_PCFG, PCFG
from ..model.NeuralPCFG import NeuralPCFG


class FNPCFG(NeuralPCFG):
    """FGG-NPCFG"""

    def __init__(self, args):
        super(FNPCFG, self).__init__(args)
        self.pcfg = Faster_PCFG()

    def evaluate(
        self,
        input,
        decode_type="mbr",
        label=False,
        rule_update=False,
        tree=None,
        **kwargs,
    ):
        if decode_type == "viterbi":
            self.pcfg = PCFG()
        result = super().evaluate(
            input,
            decode_type=decode_type,
            label=label,
            rule_update=rule_update,
            tree=tree,
            **kwargs,
        )
        return result

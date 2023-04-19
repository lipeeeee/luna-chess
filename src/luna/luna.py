"""
    Luna-Chess main logic
"""

from .luna_NN import LunaNN
from .luna_eval import LunaEval
import chess

class Luna():
    """Luna_chess engine main class"""

    def __init__(self, verbose=False) -> None:
        """If on initialization there is no pre-saved model we create one and train it, to then save it"""
        self.verbose = verbose
        self.luNNa = LunaNN(cuda=True, verbose=verbose, epochs=100, save_after_each_epoch=True)
        self.luna_eval = LunaEval()
        self.board = chess.Board()
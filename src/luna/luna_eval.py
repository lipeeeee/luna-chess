"""
    Basically the "Brain" of luna,
    LunaEval to evaluate a certain board position,
    
    this could potentially be moved somewhere else such as LunaNN but 
    this is a more correct approach
"""

import chess
import chess.engine
import chess.pgn
from .luna_state import LunaState
from .luna_NN import LunaNN
import torch

MAXVAL = 100000
class LunaEval():
    """Luna-Chess trained position evaluator()"""
    
    def __init__(self, verbose=False) -> None:
        self.model = LunaNN(verbose=verbose, epochs=100, save_after_each_epoch=True)
        assert self.model.model_exists()
        
        self.reset()

    def __call__(self, s:LunaState) -> float:
        """override object call to make it so we can evaluate positions by: LunaEval(state) -> float"""
        brd = LunaState.serialize_board(s.board)

        output = self.model(torch.tensor(brd, device="cuda").float())
        self.count += 1
        
        return float(output.data[0][0])

    def reset(self) -> None:
        """Eval Counter, for printing purposes"""
        self.count = 0


#No neural net eval
class oldLunaEval():
    """Luna-Chess custom position evaluator(NO NEURAL NET)"""
    values = {  
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
        }
    
    def __init__(self, verbose=False) -> None:
        if verbose: print(f"[LUNAEVAL] Defining base evaluation")
        self.reset()
        self.memo = {}

    def reset(self) -> None:
        self.count = 0

    def __call__(self, s: LunaState):
        self.count += 1
        key = s.key()
        if key not in self.memo:
            self.memo[key] = self.value(s)
        return self.memo[key]

    def value(self, s: LunaState):
        """Calculate a board's value"""
        b = s.board
        
        # game over values
        if b.is_game_over():
            if b.result() == "1-0":
                return MAXVAL
            elif b.result() == "0-1":
                return -MAXVAL
        else:
            return 0

        val = 0.0
        # piece values
        pm = b.piece_map()
        for x in pm:
            tval = self.values[pm[x].piece_type]
            if pm[x].color == chess.WHITE:
                val += tval
            else:
                val -= tval

        # add a number of legal moves term
        bak = b.turn
        b.turn = chess.WHITE
        val += 0.1 * b.legal_moves.count()
        b.turn = chess.BLACK
        val -= 0.1 * b.legal_moves.count()
        b.turn = bak

        return val
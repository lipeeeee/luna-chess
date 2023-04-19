"""
    Luna-Chess position evaluator
"""

import chess
import chess.pgn

MAXVAL = 10000
class LunaEval():
    """Luna-Chess custom position evaluator"""
    values = {  
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
        }
    
    def __init__(self) -> None:
        self.reset()
        self.memo = {}

    def reset(self) -> None:
        self.count = 0

    def __call__(self, s):
        self.count += 1
        key = s.key()
        if key not in self.memo:
            self.memo[key] = self.value(s)
        return self.memo[key]

    def value(self, s: chess.pgn.Game):
        """Calculate a board's value"""
        b = s.board()
        
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
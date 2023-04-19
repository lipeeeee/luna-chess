"""
    Luna-Chess main logic
"""

from .luna_NN import LunaNN
from .luna_eval import LunaEval, MAXVAL
from .luna_state import LunaState
import time
import chess

class Luna():
    """Luna_chess engine main class"""

    def __init__(self, verbose=False, max_depth=5) -> None:
        """If on initialization there is no pre-saved model we create one and train it, to then save it"""
        self.verbose = verbose
        self.luna_eval = LunaEval(verbose=verbose)
        self.board_state = LunaState()
        self.max_depth = max_depth

    @property
    def board(self) -> chess.Board:
        return self.board_state.board

    def computer_minimax(self, s:LunaState, v:LunaEval, depth:int, a, b, big=False):
        """Perform minimax on a board_state"""

        if depth >= 5 or s.board.is_game_over():
            return v(s)
        
        # white is maximizing player
        turn = s.board.turn
        if turn == chess.WHITE:
            ret = -MAXVAL
        else:
            ret = MAXVAL
        if big:
            bret = []

        # can prune here with beam search
        isort = []
        for e in s.board.legal_moves:
            s.board.push(e)
            isort.append((v(s), e))
            s.board.pop()
        move = sorted(isort, key=lambda x: x[0], reverse=s.board.turn)

        # beam search beyond depth 3
        if depth >= 3:
            move = move[:10]

        for e in [x[1] for x in move]:
            s.board.push(e)
            tval = self.computer_minimax(s, v, depth+1, a, b)
            s.board.pop()
            if big:
                bret.append((tval, e))
            
            if turn == chess.WHITE:
                ret = max(ret, tval)
                a = max(a, ret)
                if a >= b:
                    break  # b cut-off
            else:
                ret = min(ret, tval)
                b = min(b, ret)
                if a >= b:
                    break  # a cut-off

        if big:
            return ret, bret
        else:
            return ret

    def explore_leaves(self, s:LunaState, v:LunaEval):
        ret = []
        start = time.time()
        v.reset()
        bval = v(s)
        cval, ret = self.computer_minimax(s, v, 0, a=-MAXVAL, b=MAXVAL, big=True)

        if self.verbose:
            eta = time.time() - start
            print("[EXPLORING] %.2f -> %.2f: explored %d nodes in %.3f seconds %d/sec" % (bval, cval, v.count, eta, int(v.count/eta)))

        return ret

    def computer_move(self, s:LunaState, v:LunaEval):
        # computer move
        move = sorted(self.explore_leaves(s, v), key=lambda x: x[0], reverse=s.board.turn)
        if len(move) == 0:
            return
        
        # TODO: Add a bit of randomness
        chosen_move = move[0][1]

        if self.verbose:
            print("[COMPUTER MOVE] Calculated Top 3:")        
            for i,m in enumerate(move[0:3]):
                print(f"  {m}")
            print(f"[COMPUTER MOVE] {s.board.turn} MOVING TO {chosen_move}")
        
        s.board.push(chosen_move)

    def new_game(self) -> None:
        """New game"""
        pass

    def is_game_over(self) -> bool:
        """Checks if game is over via checkmate or stalemate"""
        return self.board.is_game_over()

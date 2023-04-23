"""
    Luna-Chess main logic
"""

from .luna_eval import LunaEval, MAXVAL
from .luna_state import LunaState
from .luna_constants import SEARCH_DEPTH
import time
import chess
import chess.engine
import random

class Luna():
    """Luna_chess engine main class"""

    def __init__(self, verbose=False) -> None:
        """If on initialization there is no pre-saved model we create one and train it, to then save it"""
        self.verbose = verbose
        self.luna_eval = LunaEval(verbose=verbose)
        self.board_state = LunaState()

    @property
    def board(self) -> chess.Board:
        return self.board_state.board

    # Search algo: https://www.chessprogramming.org/Alpha-Beta
    def alpha_beta_pruning(self, s:LunaState, v:LunaEval, depth:int, alpha: int, beta: int, root=False):
        """Alpha-Beta Pruning Search, gets eval and tuple with moves and its eval"""
        if depth == 0 or s.board.is_game_over():
            return v(s)
        
        # get all legal moves from the current position for the current player
        turn = s.board.turn
        legal_moves = s.board.legal_moves
        
        if root:
            eval_tuple_list = []

        if turn == chess.WHITE:
            value = -MAXVAL
            for move in legal_moves:
                # get board after move and calculate it
                s.board.push(move)
                alpha_beta_iteration_value = self.alpha_beta_pruning(s, v, depth-1, alpha, beta)
                value = max(value, alpha_beta_iteration_value)
                alpha = max(alpha, value)
                if root:
                    eval_tuple_list.append((v(s), move))
                # get back to original board
                s.board.pop()

                # this will 
                if alpha >= beta:
                    break
        else:
            value = MAXVAL
            for move in legal_moves:
                # get board after move and calculate it
                s.board.push(move)
                alpha_beta_iteration_value = self.alpha_beta_pruning(s, v, depth-1, alpha, beta)
                value = min(value, alpha_beta_iteration_value)
                beta = min(beta, value)
                if root:
                    eval_tuple_list.append((v(s), move))
                # get back to original board
                s.board.pop()

                # this will 
                if beta <= alpha:
                    break
            
        if root:
            return value, eval_tuple_list
        else:
            return value

    def explore_leaves(self, s:LunaState, v:LunaEval) -> list:
        """
            Explore Neighbour states with alpha pruning and calculate values,
            this function returns the value of each neighbour after (DEPTH) moves
        """
        if self.verbose: start = time.time()
        bval = v(s)
        v.reset()
        ret, eval_tuple_list = self.alpha_beta_pruning(s, v, depth=SEARCH_DEPTH, alpha=-MAXVAL, beta=MAXVAL, root=True)

        if self.verbose:
            eta = time.time() - start
            print("[EXPLORING] %.2f: explored %d nodes in %.3f seconds %d/sec" % (bval, v.count, eta, int(v.count/eta)))
            print(f"[BOARD]\n{s.board}")

        return eval_tuple_list

    def computer_move(self, s:LunaState, v:LunaEval):
        """Logic for selecting and making move"""
        
        # Explore all neighbours and it's values
        move = sorted(self.explore_leaves(s, v), key=lambda x: x[0], reverse=s.board.turn)
        if len(move) == 0:
            return
        
        # Get Move from best value
        # TODO: Add a bit of randomness
        chosen_move = move[0][1]

        if self.verbose:
            print("[COMPUTER MOVE] Calculated Top 3:")        
            for i,m in enumerate(move[0:3]):
                print(f"  {m}")
            print(f"[COMPUTER MOVE] {s.board.turn} MOVING TO {chosen_move}")

        # make move        
        s.board.push(chosen_move)

    @staticmethod
    def random_board(max_depth=200) -> chess.Board:
        """Generate a random board position"""
        board = chess.Board()
        depth = random.randrange(0, max_depth)
        
        for _ in range(depth):
            all_moves = list(board.legal_moves)
            random_move = random.choice(all_moves)
            board.push(random_move)
            
            if board.is_game_over():
                break
        
        return board

    def new_game(self) -> None:
        """New game"""
        pass

    def is_game_over(self) -> bool:
        """Checks if game is over via checkmate or stalemate"""
        return self.board.is_game_over()

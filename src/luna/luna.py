"""
    Luna-Chess main logic
"""

from .luna_eval import LunaEval, MAXVAL
from .luna_state import LunaState
from .luna_constants import SEARCH_DEPTH
from .luna_utils import *
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

    def visualize_net(self) -> None:
        """Visual representation of the network"""
        import torchviz
        state = LunaState()
        y = self.luna_eval(state)
        
        torchviz.make_dot(y, params=dict(self.luna_eval.model.named_parameters()))

    def luna_stockfish_diff(self, num_tests=1_000_000) -> int:
        """Calculate difference in outputs between luna and stockfish"""
        ls = LunaState(self.random_board(random.randint(10, 100)))
        le = self.luna_eval
        
        diff_sum = 0
        init_stockfish()
        sucessful_i = 0
        for i in range(num_tests):
            # gen board
            while ls.board.is_game_over():
                ls = LunaState(self.random_board(random.randint(10, 100)))
            
            luna_res = le(ls)
            sf_res = stockfish(ls.board, 0)
            
            if sf_res is None:
                continue

            if luna_res >= sf_res:
                diff_sum += luna_res - sf_res
            else:
                diff_sum += sf_res - luna_res
            
            sucessful_i += 1

            # verbose logic
            if num_tests >= 1000:
                print(f"[DIFF {i}/{num_tests}] Luna - Stockfish; diff_sum: {diff_sum}; avg: {diff_sum/sucessful_i}")

        return diff_sum, (diff_sum/sucessful_i) #sum, avg

    def print_luna_vs_stockfish(self, times=200):
        """Print luna vs stockfish results"""
        init_stockfish()
        
        for i in range(times):
            rnd_board = self.random_board(100)
            print("[LUNA VS STOCKFISH "+ str(i+1) + "/" + str(times) + "]\
luna(" + str(self.luna_eval(LunaState(rnd_board))) + ") vs stockfish(" + str(stockfish(rnd_board, 0)) +");")
        
        close_stockfish()

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

    def is_game_over(self) -> bool:
        """Checks if game is over via checkmate or stalemate"""
        return self.board.is_game_over()

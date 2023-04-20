"""
    Luna-Chess, a chess engine

    Project Architecture    
    Wrapper(either html or anything else) ->
        Luna ->
            Luna_State ->
            Luna_Eval ->
                Luna_NN ->
                Luna_dataset ->

    by lipeeeee
"""

from luna import *
import chess
import sys

def main() -> int:
    """Entry Point"""
    luna_engine = Luna(verbose=True)
    while 1:
        board = luna_engine.random_board(200)
        #print(board)
        #print(board.fen())
        a = luna_engine.luna_eval.stockfish(board, 0)

    return 0


if __name__ == "__main__":
    sys.exit(main())
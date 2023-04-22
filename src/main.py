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
import random
import chess
import sys

def main() -> int:
    """Entry Point"""
    
    # le = LunaEval(True)
    # print(le(LunaState()))
    # init_stockfish()
    # print(stockfish(chess.Board(), 0))
    l = Luna(True)
    print(luna_stockfish_diff(l))

    return 0

def luna_stockfish_diff(l: Luna,num_tests=1_000_000) -> int:
    """Calculate difference in outputs between luna and stockfish"""
    ls = LunaState(l.random_board(random.randint(10, 100)))
    le = l.luna_eval
    
    diff_sum = 0
    init_stockfish()
    sucessful_i = 0
    for i in range(num_tests):
        # gen board
        while ls.board.is_game_over():
            ls = LunaState(l.random_board(random.randint(10, 100)))
        
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


if __name__ == "__main__":
    sys.exit(main())
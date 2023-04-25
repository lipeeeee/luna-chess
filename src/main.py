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
    #l = Luna(True)
    #print(luna_stockfish_diff(l))
    # b = Luna.random_board(30)
    # luna_state.LunaState(b)
    # bs = luna_state.LunaState.serialize_board(b)
    # c=0
    # for i in bs:
    #     print(f"I:{c}:\n{i}\n")
    #     c+=1

    # epsquare = None
    # while epsquare == None:
    #     b = Luna.random_board(60)
    #     epsquare = b.ep_square

    # print(epsquare)
    luna = Luna(True)
    luna.luna_stockfish_diff(1_000_000) 
    luna.visualize_net()

    return 0


if __name__ == "__main__":
    sys.exit(main())
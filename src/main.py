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
    #le = LunaEval(True)
    #s = LunaState(chess.Board())
    #print("PRED", le(s))
    ln = luna_NN.LunaNN()
    print(ln)
    return 0


if __name__ == "__main__":
    sys.exit(main())
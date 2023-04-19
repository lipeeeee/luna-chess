"""
    Zero Knowledge Chess Engine

    
    Wrapper(either html or anything else) ->
        Luna ->
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
    le = LunaEval(True)
    s = LunaState(chess.Board())
    print("PRED", le(s))

    return 0


if __name__ == "__main__":
    sys.exit(main())
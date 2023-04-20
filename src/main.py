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
    luna_s = LunaState()
    print(luna_s.better_serialize_board(luna_s.board)[12])

    return 0


if __name__ == "__main__":
    sys.exit(main())